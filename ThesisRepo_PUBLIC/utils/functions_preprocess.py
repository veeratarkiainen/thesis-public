import mne
import neurokit2 as nk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as sp
from scipy.signal import find_peaks

"""
Functions for signal preprocessing, event detection and initial summary statistics.

Includes saccade onset detection by adaptive velocity-based algorithm and an alternative acceleration-based method. 

FILTER for preprocessing
DETECT for event detection
STATS for summary statistics (mean, mode, sd, range)

by Veera Tarkiainen
"""

sfreq = 1000

#%% Filter signals

def filter_eog(eog_raw_signals, sfreq):
    filtered_eog_signals = []
    for i in range(eog_raw_signals.shape[0]):
        signal = eog_raw_signals[i]
        filtered_eog = nk.signal_filter(signal, lowcut=0.1, highcut=30, method='butterworth', sampling_rate=sfreq)
        #print("0.01")
        filtered_eog_signals.append(filtered_eog)
    return np.array(filtered_eog_signals).T # Shape: (channels, timepoints).T

def filter_ecg(ecg_raw_signals, sfreq):
    filtered_ecg_signals = []

    for i in range(ecg_raw_signals.shape[0]):  # Loop through each channel (row in the transposed array)
        signal = ecg_raw_signals[i]  # This is a 1D vector
        filtered_ecg = nk.ecg_clean(signal, sampling_rate=sfreq, method = "neurokit")
        b, a = sp.butter(4, 45, btype='low', fs=sfreq)
        filtered_ecg = sp.filtfilt(b, a, filtered_ecg)
        filtered_ecg_signals.append(filtered_ecg)

    return np.array(filtered_ecg_signals).T

def filter_rsp(rsp_raw, sfreq):
    filtered_rsp = nk.signal_filter(rsp_raw, lowcut=0.1, highcut=0.5, method='butterworth', sampling_rate=sfreq)
    return np.array(filtered_rsp).T

def remove_blinks(heog_clean_blinks, veog_blinks, window):

    n_samples = len(veog_blinks)
    mask = np.zeros(n_samples, dtype=bool)

    # Mark samples around blinks
    blink_indices = np.where(veog_blinks == 1)[0]

    for idx in blink_indices:
        start = max(0, idx - window)
        end = min(n_samples, idx + window)
        mask[start:end] = True

    # Interpolate the masked regions for both channels
    heog_no_blinks = heog_clean_blinks.copy()
    for ch in range(2):
        signal = heog_clean_blinks[:, ch]
        good = ~mask
        interp_signal = np.interp(np.arange(n_samples), np.where(good)[0], signal[good])
        heog_no_blinks[:,ch] = interp_signal

    return np.array(heog_no_blinks), blink_indices

def preprocess_veog(veog_signals, sfreq):
    veog_signals = -veog_signals
    # Optional manual filter
    veog_filtered = nk.signal_filter(veog_signals.flatten(), lowcut=0.05, highcut=10, method='butterworth',sampling_rate = sfreq)
    signals, _ = nk.eog_process(veog_filtered, sfreq)
    return signals # is a dataframe

def compute_leads(channel_x, channel_y):
    lead = channel_x - channel_y
    return lead

#%% Detect saccade onsets

def merge_saccades(saccades_df, min_merge_gap_ms = 100): # latency 150-250 ms
    """
    Merges consecutive saccades of the same direction that are close together.
    """
    if len(saccades_df) < 2:
        return saccades_df

    merged_saccades = []
    # Convert dataframe to a list of dictionaries for easier processing
    saccades = saccades_df.sort_values(by='start_index').to_dict('records')

    current_saccade = saccades[0]

    for next_saccade in saccades[1:]:
        # Calculate the time gap between the end of the current and start of the next saccade
        gap_ms = (next_saccade['start_time'] - current_saccade['end_time']) * 1000

        # Check for merge condition: same direction and small gap
        if (next_saccade['direction'] == current_saccade['direction'] and
            gap_ms <= min_merge_gap_ms):

            # Merge the next saccade into the current one
            current_saccade['end_time'] = next_saccade['end_time']
            current_saccade['end_index'] = next_saccade.get('end_index', next_saccade['start_index']) # Handle potential missing key
            current_saccade['duration_ms'] = (current_saccade['end_time'] - current_saccade['start_time']) * 1000

            # Update peak velocity if the next saccade's is larger
            if abs(next_saccade['peak_velocity']) > abs(current_saccade['peak_velocity']):
                current_saccade['peak_velocity'] = next_saccade['peak_velocity']
        else:
            # No merge, finalize the current saccade and move to the next
            merged_saccades.append(current_saccade)
            current_saccade = next_saccade

    # Add the last saccade
    merged_saccades.append(current_saccade)
    return pd.DataFrame(merged_saccades)

def threshold_saccades_by_amplitude(saccades_df, sampling_rate, eog_signal, min_amplitude_change = 250):
    """
    Filters out saccades that do not meet a minimum amplitude change criterion.
    """
    if saccades_df.empty:
        return saccades_df

    valid_saccades_indices = []
    for index, saccade in saccades_df.iterrows():
        # Get start and end indices from the saccade dataframe
        start_idx = int(saccade['start_index'])
        # The end_index for velocity corresponds to the start of the last sample,
        # so for amplitude we look at the next sample.
        end_idx = int(saccade['end_time'] * sampling_rate) # Recalculate end index from time

        if start_idx < 0 or end_idx >= len(eog_signal):
            continue # Skip if indices are out of bounds

        # Calculate the amplitude change from the EOG signal
        amplitude_start = eog_signal[start_idx]
        amplitude_end = eog_signal[end_idx]
        amplitude_change = abs(amplitude_end - amplitude_start)

        # Check if the change meets the threshold
        if amplitude_change >= min_amplitude_change:
            valid_saccades_indices.append(index)

    return saccades_df.loc[valid_saccades_indices].reset_index(drop=True)

def detect_saccades_robust(eog_signal, sampling_rate, threshold_factor, min_duration_ms, min_threshold_factor, min_saccades_required): # Adapted Nyström & Holmqvist 2010, Voloh et al. 2019
    """
    Detects saccades by a velocity-based statistically robust & adaptable algorithm
    """

    best_saccades_df = pd.DataFrame()
    best_threshold = None

    min_duration_samples = int((min_duration_ms / 1000) * sampling_rate)

    while threshold_factor >= min_threshold_factor:
        saccades = []

        # 1. Calculate the velocity of the EOG signal
        velocity = np.gradient(eog_signal) * sampling_rate

        # 2. Adaptive velocity threshold
        robust_std = 1.4826 * np.median(np.abs(velocity - np.median(velocity))) #/ 0.6745
        velocity_threshold = np.median(velocity) + threshold_factor * robust_std

        # 3. Identify points exceeding the threshold
        above_threshold = np.where(np.abs(velocity) > velocity_threshold)[0]
        if len(above_threshold) == 0:
            threshold_factor -= 1.0
            continue  # Don't exit early — try a lower threshold

        # 4. Group consecutive points into saccades
        diffs = np.diff(above_threshold)
        split_points = np.where(diffs > 1)[0] + 1
        saccade_groups = np.split(above_threshold, split_points)

        # 5. Process detected saccades
        for saccade in saccade_groups:
            if len(saccade) < min_duration_samples:
                continue

            start_index = saccade[0]
            end_index = saccade[-1]

            saccade_velocities = velocity[start_index:end_index + 1]
            peak_velocity_index = np.argmax(np.abs(saccade_velocities))
            peak_velocity = saccade_velocities[peak_velocity_index]

            duration_samples = end_index - start_index
            duration_ms = (duration_samples / sampling_rate) * 1000
            direction = 'right' if peak_velocity > 0 else 'left'

            saccade_info = {
                'start_index': start_index,
                'end_index': end_index,
                'start_time': start_index / sampling_rate,
                'end_time': end_index / sampling_rate,
                'duration_ms': duration_ms,
                'peak_velocity': peak_velocity,
                'direction': direction,
            }
            saccades.append(saccade_info)

        saccades_df = pd.DataFrame(saccades)

        # Threshold by latency and amplitude
        merged_saccades_df = merge_saccades(saccades_df)
        threshold_saccades_df = threshold_saccades_by_amplitude(merged_saccades_df, sampling_rate, eog_signal)

        print(f"[Threshold {threshold_factor:.1f}] Detected {len(threshold_saccades_df)} saccades.")

        # Early return if enough saccades found
        if len(threshold_saccades_df) >= min_saccades_required:
            print(f"Returning {len(threshold_saccades_df)} saccades from threshold {threshold_factor:.1f}")
            return threshold_saccades_df

        # Keep best result (most saccades found so far)
        if len(threshold_saccades_df) > len(best_saccades_df):
            best_saccades_df = threshold_saccades_df
            best_threshold = threshold_factor

        threshold_factor -= 1.0

    if best_threshold is not None:
        print(f"Returning best result with {len(best_saccades_df)} saccades from threshold {best_threshold:.1f}")
        return best_saccades_df

    else:
        print("No valid saccades found at any threshold.")
        return pd.DataFrame()  # return an empty DataFrame


def detect_saccades_acceleration(subject, eog_signal, times, threshold_factor, min_threshold_factor, min_saccades_required):

    best_saccades_df = pd.DataFrame()
    best_threshold = None

    right = 1
    left = -1

    visualize = True

    while threshold_factor >= min_threshold_factor:
        saccades = []

        # 3) derivatives
        t_vel = (times[:-1] + times[1:]) / 2
        vel = np.diff(eog_signal) / np.diff(times)  # V/s
        vel_z = (vel - vel.mean()) / (vel.std() + 1e-12)

        t_acc = (t_vel[:-1] + t_vel[1:]) / 2
        acc = np.diff(vel) / np.diff(t_vel)  # V/s^2
        acc_z = (acc - acc.mean()) / (acc.std() + 1e-12)

        # 4) peak detection on acceleration (second derivative)
        r_idx, r_mag = mne.preprocessing.peak_finder(acc_z, thresh=3, extrema=right)
        l_idx, l_mag = mne.preprocessing.peak_finder(acc_z, thresh=3, extrema=left)

        # Threshold the peaks with regard to their length in terms of the first derivative
        peak_dur_thresh = int(threshold_factor * sfreq)

        right_peak_keep_indices = [peak_idx for peak_idx, peak_loc in enumerate(r_idx)
                                   if np.all(vel_z[peak_loc:peak_loc + peak_dur_thresh] > 1)]
        right_idx_keep = r_idx[right_peak_keep_indices]

        left_peak_keep_indices = [peak_idx for peak_idx, peak_loc in enumerate(l_idx)
                                  if np.all(vel_z[peak_loc:peak_loc + peak_dur_thresh] < -1)]
        left_idx_keep = l_idx[left_peak_keep_indices]

        right_times = t_acc[right_idx_keep]
        left_times  = t_acc[left_idx_keep]

        # Visualize derivatives
        if visualize:
            start_t = 97000
            end_t = 102000
            fig, axs = plt.subplots(3, figsize=(30, 12))
            axs[0].plot(times[start_t:end_t], eog_signal[start_t:end_t])
            axs[0].set_ylabel('Horizontal EOG (V)')
            axs[1].plot(t_vel[start_t:end_t], vel_z[start_t:end_t])
            axs[1].set_ylabel('First derivative (V/s)')
            axs[2].plot(t_acc[start_t:end_t], acc_z[start_t:end_t])
            axs[2].set_ylabel('Second derivative (V/s^2)')
            for ax in axs:
                for p_r, p_l, pp_r, pp_l in zip(t_acc[right_idx_keep], t_acc[left_idx_keep], t_vel[right_idx_keep], t_vel[left_idx_keep]):
                    ax.axvline(x=p_r, color='r', linestyle='--', linewidth=2.0)
                    ax.axvline(x=p_l, color='r', linestyle='--', linewidth=2.0)
                    ax.axvline(x=pp_r, color='g', linestyle='--', lindewidth=2.0)
                    ax.axvline(x=pp_l, color='g', linestyle='--', lindewidth=2.0)
                    ax.set_xlim(times[start_t], times[end_t])
                    ax.grid()
                    start, end = ax.get_xlim()
                    ax.set_xticks(np.arange(start, end, 0.2))
            fig.suptitle(f'{subject}')
            fig.tight_layout()
            visualize = False

        rows = []
        rows += [{'direction': 'right', 'start_time': t} for t in right_times]
        rows += [{'direction': 'left',  'start_time': t} for t in left_times]
        saccades_df = pd.DataFrame(rows)

        print(f"[Threshold {threshold_factor:.1f}] Detected {len(saccades_df)} saccades.")

        # Early return if enough saccades found
        if len(saccades_df) >= min_saccades_required:
            print(f"Returning {len(saccades_df)} saccades from threshold {threshold_factor:.1f}")
            return saccades_df

        # Keep best result (most saccades found so far)
        if len(saccades_df) > len(best_saccades_df):
            best_saccades_df = saccades_df
            best_threshold = threshold_factor

        threshold_factor -= 0.01

    if best_threshold is not None:
        print(f"Returning best result with {len(best_saccades_df)} saccades from threshold {best_threshold:.1f}")
        return best_saccades_df

    else:
        print("No valid saccades found at any threshold.")
        return pd.DataFrame()  # return an empty DataFrame

#%% Detect respiration events

def detect_respiration_peaks_diff(signal, sampling_rate):

    velocity = np.diff(signal) * sampling_rate
    # Identify zero-crossings in derivative

    # Inhalation peaks: derivative changes from positive to negative
    inhalation_peaks = np.where((velocity[:-1] > 0) & (velocity[1:] <= 0))[0] + 1

    # Exhalation peaks: derivative changes from negative to positive
    exhalation_troughs = np.where((velocity[:-1] < 0) & (velocity[1:] >= 0))[0] + 1

    if len(inhalation_peaks) > 0:
        inh_prominence = 0.2 * (np.max(signal) - np.min(signal)) # change multiplier depending on how "big" changes you want to detect
        inh_peaks_refined, _ = find_peaks(signal, prominence=inh_prominence)
        inhalation_peaks = np.array([p for p in inhalation_peaks if p in inh_peaks_refined])

    if len(exhalation_troughs) > 0:
        exh_prominence = 0.2 * (np.max(signal) - np.min(signal))
        exh_troughs_refined, _ = find_peaks(-signal, prominence=exh_prominence)
        exhalation_troughs = np.array([p for p in exhalation_troughs if p in exh_troughs_refined])

    return inhalation_peaks, exhalation_troughs

#%% Get descriptive (signal-level) statistics

def blink_stats(signal, peaks, sampling_rate, window_ms=500):
    n = len(signal)
    window = int(window_ms * sampling_rate / 1000)
    onsets, offsets, durations = [], [], []

    for pk in np.asarray(peaks, dtype=int):
        start = max(0, pk - window)
        if pk <= start:
            continue
        onset = start + int(np.argmin(signal[start:pk]))

        end = min(n, pk + window)
        if end <= pk:
            continue
        offset = pk + int(np.argmin(signal[pk:end]))

        if offset <= onset:
            continue

        onsets.append(onset)
        offsets.append(offset)
        durations.append((offset - onset) / sampling_rate)

    # build arrays once (safe even if empty)
    onsets_arr    = np.array(onsets, dtype=int)
    offsets_arr   = np.array(offsets, dtype=int)
    durations_arr = np.array(durations, dtype=float)

    return {
        "Blink_counts":      len(peaks),
        "Blink_rate_bpm":    len(peaks) * 60 * sampling_rate / n,
        "Blink_onset_idx":   onsets_arr,
        "Blink_offset_idx":  offsets_arr,
        "Blink_durations_s": durations_arr,
    }

def saccade_stats(signal, onsets, sampling_rate):

    saccade_count = len(onsets)
    saccades_per_min = saccade_count * 60 * sampling_rate / len(signal)

    stats = {
        'Saccade_count': saccade_count,
        'Saccade_rate': saccades_per_min
    }
    return stats

def respiration_stats(time, inh_peaks, exh_peaks):

    t_inh = time[inh_peaks]
    t_exh = time[exh_peaks]

    insp_durs = []
    for t0 in t_inh:
        later = t_exh[t_exh > t0]
        if len(later) > 0:
            insp_durs.append(later[0] - t0)

    exp_durs = []
    for t0 in t_exh:
        later = t_inh[t_inh > t0]
        if len(later) > 0:
            exp_durs.append(later[0] - t0)

    # Convert to arrays
    insp = np.array(insp_durs)
    exp  = np.array(exp_durs)

    # Only full cycles (inh + next exh)
    num_cycles = min(len(insp), len(exp))

    # Slice to equal length
    insp = insp[:num_cycles]
    exp  = exp[:num_cycles]

    # Compute cycle durations and I:E ratios
    cycle_durs = insp + exp
    ratios     = insp / exp

    # Respiration rate (breaths per minute)
    respiration_rate = 60.0 / cycle_durs

    stats = {
        'Inhales_count':        len(inh_peaks),
        'Exhales_count':        len(exh_peaks),
        'Cycles_count':         num_cycles,
        'Respiration_rate_bpm': respiration_rate,
        'Inhale_durations_s':   insp,
        'Exhale_durations_s':   exp,
        'Cycle_durations_s':    cycle_durs,
        'Ratios_IE':            ratios,
    }
    return stats

def ecg_stats(time,  r_peaks, t_offsets):

    r_times = time[r_peaks]
    t_times = time[t_offsets]
    rr_intervals = np.diff(r_times)

    # Pair T_end(i) with R_i and R_{i+1}
    # Use as many beats as we can pair safely
    N = min(len(rr_intervals), len(t_times))

    if N == 0:
        return {
        'RR_intervals':           np.array([]),
        'RT_intervals':           np.array([]), # "systole"
        'TR_intervals':           np.array([]), # "diastole"
        'HR_bpm':                 np.array([])
        }

    R_i   = r_times[:N]
    R_ip1 = r_times[1:N+1]
    T_i   = t_times[:N]

    # RT
    RT = T_i - R_i # systole
    RR = R_ip1 - R_i

    # Keep only beats where T_end falls between its R and the next R
    valid = (RT > 0) & (RT < RR)

    RT      = RT[valid]
    RR      = RR[valid]
    hr_bpm  = 60.0 / RR
    TR    = RR - RT                       # diastole = RR - RT

    stats = {
        'RR_intervals':           RR,
        'RT_intervals':           RT, # "systole"
        'TR_intervals':           TR, # "diastole"
        'HR_bpm':                 hr_bpm
    }
    return stats