import os, gc
from utils.functions_preprocess import *
from utils.align_start_times import relative_differences_seconds
from scipy.signal import savgol_filter
import time, sys, platform
t=time.time(); print("Pandas import:", round(time.time()-t,2),"s")
print(sys.version); print(platform.platform())
import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)

"""
Functions to read raw data, preprocess signals and detect events with descriptive statistics by Veera Tarkiainen
"""

# Go through rows in subject list text file
print("Reading raw data...")
file_names=[]
with open('data\subject_list.txt') as subjects:
    for line in subjects:
        # Get subject number
        line = line.strip()
        line_data = line.split("_")
        sub = line_data[1]
        date = line_data[0]
        # Load EEG data
        eeg_file = os.path.join("data", date + "_" + sub, date + "_" + sub + ".vhdr")
        file_names.append(eeg_file)

def read_raw_data(file_names):
    sub_names = ["Sub0" + str(i) for i in range(len(file_names))]
    raw_datas = []
    sample_times = []

    for i in range(len(sub_names)):
        raw = mne.io.read_raw_brainvision(file_names[i], preload=True, verbose=False)

        ch_names = raw.ch_names[:7]  # First 7 channels
        raw_data = raw.get_data(picks=ch_names)  # Shape: (channels, time)
        samples = raw.times
        sfreq = raw.info["sfreq"]
        sample_times.append(samples)

        # Rename channels
        rename_dict = {
            raw.ch_names[0]: 'HEOG_r',
            raw.ch_names[1]: 'HEOG_l',
            raw.ch_names[2]: 'VEOG_r',
            raw.ch_names[3]: 'RA',
            raw.ch_names[4]: 'LL',
            raw.ch_names[5]: 'LA',
            raw.ch_names[6]: 'Respiration',
        }
        raw.rename_channels(rename_dict)
        new_ch_names = raw.ch_names[:7]

        # Set channel types
        raw.set_channel_types({
            'HEOG_r': 'eog', 'HEOG_l': 'eog', 'VEOG_r': 'eog',
            'RA': 'ecg', 'LL': 'ecg', 'LA': 'ecg', 'Respiration': 'resp'
        })

        print(raw.info)

        # Convert to DataFrame
        df = pd.DataFrame(raw_data.T, columns=new_ch_names, index=samples)
        df["Subject"] = sub_names[i]  # Add subject column
        df["Time"] = sample_times[i]
        raw_datas.append(df)
        raw_data_df = pd.concat(raw_datas, ignore_index=True)

    return raw_data_df, sample_times, sfreq
print("Raw data successfully read!")

print("Preprocessing raw data....")
def preprocess_data(df, time_difference):
    subjects = df["Subject"].unique()

    clean_signals = []
    veog_blinks= []

    # Define trial ranges in seconds
    base_trial_ranges = [(20, 80), (90, 270), (280, 460), (470, 650)]

    for i, subject in enumerate(subjects):
        subject_df = df[df["Subject"] == subject]
        time_shift = time_difference[i]

        current_trial_ranges = list(base_trial_ranges)
        for trial_idx, (start_sec, end_sec) in enumerate(current_trial_ranges):

            start = int((start_sec + time_shift) * sfreq)
            end = int((end_sec + time_shift) * sfreq)

            # Extract signals and convert units
            eog_signals = subject_df[['HEOG_r', 'HEOG_l']].iloc[start:end].values.T * 1e6 # in microvolts
            veog_signals = subject_df[['VEOG_r']].iloc[start:end].values.T * 1e6 # in microvolts
            ecg_signals = subject_df[['RA', 'LL', 'LA']].iloc[start:end].values.T * 1e3 # in millivolts
            resp_signals = subject_df[['Respiration']].iloc[start:end].values.T

            # filter signals
            heog_filt = filter_eog(eog_signals, sfreq)
            veog_filt = preprocess_veog(veog_signals, sfreq) # extract blinks
            ecg_filt = filter_ecg(ecg_signals, sfreq)
            resp_filt = filter_rsp(resp_signals, sfreq)

            resp_norm = (resp_filt - np.mean(resp_filt)) / np.std(resp_filt)
            resp_clean = savgol_filter(resp_norm, window_length=0.5*sfreq, polyorder=3, mode = "mirror") # 0.5 s

            # remove blinks
            heog_clean, blink_data = remove_blinks(heog_filt, veog_filt["EOG_Blinks"], window=250)

            # get clean leads
            heog_lead = compute_leads(heog_clean[:, 0], heog_clean[:, 1])
            ecg_lead = compute_leads(ecg_filt[:, 1], ecg_filt[:, 0])

            heog_lead_filt = savgol_filter(heog_lead, window_length=0.040*sfreq, polyorder=2, mode = "mirror") # 40ms

            #ecg_lead_norm = (ecg_lead - np.mean(ecg_lead)) / np.std(ecg_lead)

            # EDR for subjects with bad resp data
            if subject in ["Sub05", "Sub07"]:
                rpeaks, info = nk.ecg_peaks(ecg_lead, sfreq)
                ecg_rate = nk.ecg_rate(rpeaks, sfreq, desired_length=len(ecg_lead))
                edr = nk.ecg_rsp(ecg_rate, sampling_rate=sfreq)
                resp_clean = edr

            # Create DataFrames
            heog_clean_df = pd.DataFrame(heog_clean, columns=['HEOG_r', 'HEOG_l']) # don´t need?
            veog_clean_df = pd.DataFrame(veog_filt[["EOG_Clean"]].values, columns=['VEOG'])
            ecg_clean_df = pd.DataFrame(ecg_filt, columns=['RA', 'LL', 'LA']) # don´t need?
            resp_clean_df = pd.DataFrame(resp_clean, columns=['Respiration'])

            ecg_lead_df = pd.DataFrame(ecg_lead, columns = ['ECG_signal'])
            heog_lead_df = pd.DataFrame(heog_lead_filt, columns=['HEOG_signal'])

            time_col = subject_df["Time"].iloc[start:end].values
            time_col = time_col - time_col[0]

            clean_trial_df = pd.concat([heog_clean_df, veog_clean_df, ecg_clean_df, resp_clean_df, ecg_lead_df, heog_lead_df], axis=1)
            clean_trial_df['Subject'] = subject
            clean_trial_df['Trial'] = trial_idx + 1
            clean_trial_df['Time'] = time_col

            clean_signals.append(clean_trial_df)

            # save blink information too
            veog_blinks.append({
                "Blinks": blink_data,
                "Subject": subject,
                "Trial": trial_idx + 1})


    clean_signals_df = pd.concat(clean_signals, axis=0)
    veog_blinks_df = pd.DataFrame(veog_blinks)

    return clean_signals_df, veog_blinks_df

df_raw_data, sample_times, sfreq = read_raw_data(file_names)
df_clean_signals, df_veog_blinks = preprocess_data(df_raw_data, relative_differences_seconds)
print("Raw data successfully preprocessed and saved!")

trial_names = {
    1: "Control",
    2: "External",
    3: "Internal",
    4: "Self-paced"}

# Get all unique subjects
all_subjects = df_clean_signals["Subject"].unique()

#%% Blink detection

print("Blink detection...")
blink_rows = []
blink_summary = []

for subject in all_subjects:
    # Get trials for current subject
    subject_trials = df_clean_signals[df_clean_signals["Subject"] == subject]["Trial"].unique()

    for trial in subject_trials:
        # Filter data for this subject and trial
        trial_data = df_clean_signals[(df_clean_signals["Subject"] == subject) &
                                      (df_clean_signals["Trial"] == trial)]

        # Get the corresponding blink array from df_veog_blinks
        blink_data = df_veog_blinks[(df_veog_blinks["Subject"] == subject) &
                                    (df_veog_blinks["Trial"] == trial)]

        veog_signal = trial_data["VEOG"].values
        time = trial_data["Time"]

        blink_indices_raw = blink_data.iloc[0]["Blinks"]  # <- this extracts the actual list

        idx_raw = np.asarray(blink_data.iloc[0]["Blinks"], dtype=int)
        idx_raw = idx_raw[(idx_raw >= 0) & (idx_raw < len(veog_signal))]

        amps = veog_signal[idx_raw]

        q_low, q_high = 5, 95
        lo_q, hi_q = np.percentile(amps, [q_low, q_high])
        keep_q = (amps >= lo_q) & (amps >= 100) & (amps <= hi_q)
        blink_indices_q = blink_indices_raw[keep_q]

        if blink_indices_q.size == 0:
            continue

        stats_q = blink_stats(veog_signal, blink_indices_q, sfreq)
        blink_onset_indices_q = stats_q["Blink_onset_idx"]

        row = {
            'Subject': subject,
            'Trial': trial,
            **stats_q
        }

        blink_rows += [
            {"Subject": subject, "Trial": trial, "event": "blink_onset",
             "index": int(i), "time_sec": float(time[i])}
            for i in blink_onset_indices_q
        ]

        blink_summary.append(row)
        del trial_data, veog_signal, blink_data, stats_q, blink_onset_indices_q; gc.collect()

        blink_index_df = pd.DataFrame(blink_rows)
        blink_summary_df = pd.DataFrame(blink_summary)

blink_rate_stats = blink_summary_df.groupby("Trial")["Blink_rate_bpm"].agg(
    EBR_min_bpm = "min",
    EBR_max_bpm = "max",
    EBR_mean_bpm ="mean",
    EBR_sd_bpm ="std"
).round(2)

print("--- Eye Blink Rate (EBR) Statistics Grouped by Trial ---")
print(blink_rate_stats)

#%% Saccades

print("Saccade detection...")
final_saccades_df = pd.read_csv("results/saccades_review.csv")
kept_saccades = final_saccades_df[final_saccades_df['Keep']].copy()

saccade_rows = []
saccade_summary = []

for subject in all_subjects:
    # Get trials for current subject
    subject_trials = df_clean_signals[df_clean_signals["Subject"] == subject]["Trial"].unique()[1:]

    for trial in subject_trials:
        # Filter data for this subject and trial
        trial_data = df_clean_signals[(df_clean_signals["Subject"] == subject) &
                                      (df_clean_signals["Trial"] == trial)]

        eog_signal = trial_data["HEOG_signal"].values
        time = trial_data["Time"].values

        sacc = kept_saccades[(kept_saccades["Subject"] == subject) & (kept_saccades["Trial"] == trial)]
        right = sacc[sacc["direction"] == "right"]
        left  = sacc[sacc["direction"]== "left"]

        saccade_idx = sacc["start_index"]

        saccade_rows += [
            {"Subject": subject, "Trial": trial, "event": "saccade_onset",
             "index": int(i), "time_sec": float(time[i])}
            for i in saccade_idx
        ]

        stats = saccade_stats(eog_signal, saccade_idx, sfreq)

        row = {
            'Subject': subject,
            'Trial':   trial,
            # durations?
            **stats
        }
        saccade_summary.append(row)

        del trial_data, eog_signal, time, sacc, left, right, stats, saccade_idx; gc.collect()

    saccade_index_df = pd.DataFrame(saccade_rows)
    saccade_summary_df = pd.DataFrame(saccade_summary)

saccade_rate_stats = saccade_summary_df.groupby("Trial")["Saccade_rate"].agg(
    SR_min_bpm = "min",
    SR_max_bpm = "max",
    SR_mean_bpm ="mean",
    SR_sd_bpm ="std"
).round(2)

print("--- Saccade Rate (SR) Statistics Grouped by Trial ---")
print(saccade_rate_stats)

#%% Respiratory

print("Respiratory phase detection...")
rows = []
respiration_summary = []

for subject in all_subjects:

    subject_trials = df_clean_signals[df_clean_signals["Subject"] == subject]["Trial"].unique()

    for trial in subject_trials:
        trial_data = df_clean_signals[(df_clean_signals["Subject"] == subject) &
                                      (df_clean_signals["Trial"] == trial)]

        time = trial_data["Time"].values
        respiration_signal = trial_data["Respiration"].values

        # Call the functions
        inspiration_peaks, expiration_troughs = detect_respiration_peaks_diff(respiration_signal, sfreq)

        insp_peak_idx = np.asarray(inspiration_peaks, dtype=int)
        exp_trough_idx  = np.asarray(expiration_troughs, dtype=int)

        # Long rows (each index becomes a row)
        rows += [
            {"Subject": subject, "Trial": trial, "event": "inspiration_peaks",
             "index": int(i), "time_sec": float(time[i])}
            for i in insp_peak_idx
        ]
        rows += [
            {"Subject": subject, "Trial": trial, "event": "expiration_troughs",
             "index": int(i), "time_sec": float(time[i])}
            for i in exp_trough_idx
        ]

        stats = respiration_stats(time, inspiration_peaks, expiration_troughs)

        row = {
            'Subject': subject,
            'Trial':   trial,
            **stats
        }
        respiration_summary.append(row)

        del trial_data, respiration_signal, inspiration_peaks, expiration_troughs, stats

    resp_index_df = pd.DataFrame(rows)
    respiration_summary_df = pd.DataFrame(respiration_summary)

exploded_resp = respiration_summary_df.explode("Respiration_rate_bpm")

resp_rate_stats = exploded_resp.groupby("Trial")["Respiration_rate_bpm"].agg(
    RR_min_bpm = "min",
    RR_max_bpm = "max",
    RR_mean_bpm ="mean",
    RR_sd_bpm ="std"
).round(2)

print("--- Respiration Rate (RR) Statistics Grouped by Trial ---")
print(resp_rate_stats)

exploded_in = respiration_summary_df.explode("Inhale_durations_s")
exploded_ex = respiration_summary_df.explode("Exhale_durations_s")

inh_stats = exploded_in.groupby("Trial")["Inhale_durations_s"].agg(
    Inh_dur_min_s = "min",
    Inh_dur_max_s = "max",
    Inh_dur_mean_s = "mean",
    Inh_dur_sd_s = "std"
).round(2)

exp_stats = exploded_ex.groupby("Trial")["Exhale_durations_s"].agg(
    Ex_dur_min_s = "min",
    Ex_dur_max_s = "max",
    Ex_dur_mean_s = "mean",
    Ex_dur_sd_s = "std"
).round(2)

inh_ex_stats = pd.concat([inh_stats, exp_stats], axis=1).reset_index()

print("--- Inhale Statistics Grouped by Trial ---")
print(inh_stats)

print("--- Exhale Statistics Grouped by Trial ---")
print(exp_stats)

#%% Cardiac

print("Cardiac phase detection...")
ecg_rows = []
ecg_summary = []
r_peaks_count = 0
t_offsets_count = 0

# Loop over all subjects
for subject in all_subjects:
    # Get trials for current subject
    subject_trials = df_clean_signals[df_clean_signals["Subject"] == subject]["Trial"].unique()

    for trial in subject_trials:
        # Filter data for this subject and trial
        trial_data = df_clean_signals[(df_clean_signals["Subject"] == subject) &
                                      (df_clean_signals["Trial"] == trial)]

        ecg_signal = trial_data["ECG_signal"].values
        time = trial_data["Time"].values
        ecg_signals, info = nk.ecg_peaks(ecg_signal, sfreq)
        r_peaks = info["ECG_R_Peaks"] # samples at which occur

        wave_signals, waves = nk.ecg_delineate(ecg_signal, r_peaks, sfreq)
        #cardiac_phase = nk.ecg_phase(ecg_signal, r_peaks, waves, sfreq)

        raw_t = waves["ECG_T_Peaks"]
        raw_t_offset = waves["ECG_T_Offsets"]

        # 2) Convert to numpy arrays of floats (to catch NaNs), then filter & cast
        t_arr = np.array(raw_t, dtype=float)
        t_offset_arr = np.array(raw_t_offset, dtype=float)


        # keep only the finite (non-NaN, non-inf) entries, then to int
        t_peaks = list(t_arr[np.isfinite(t_arr)].astype(int))
        t_wave_offsets = list(t_offset_arr[np.isfinite(t_offset_arr)].astype(int))

        r_peaks_count += len(r_peaks)
        t_offsets_count += len(t_wave_offsets)

        ecg_rows += [
            {"Subject": subject, "Trial": trial, "event": "cardiac",
             "r_peak_index": int(i), "r_time_sec": float(time[i]), "t_wave_offset_index": int(j), "t_time_sec": float(time[j])}
            for i, j in zip(r_peaks, t_wave_offsets)
        ]

        stats = ecg_stats(time, r_peaks, t_wave_offsets)

        row = {
            'Subject': subject,
            'Trial':   trial,
            **stats
        }

        ecg_summary.append(row)

        del trial_data, ecg_signal, r_peaks, time, ecg_signals, info, wave_signals, waves, stats

    ecg_index_df = pd.DataFrame(ecg_rows)
    ecg_summary_df = pd.DataFrame(ecg_summary)

exploded_rr = ecg_summary_df.explode("RR_intervals")
exploded_hr = ecg_summary_df.explode("HR_bpm")
exploded_rt = ecg_summary_df.explode("RT_intervals")
exploded_tr = ecg_summary_df.explode("TR_intervals")

rr_interval_stats = (
    exploded_rr
    .groupby("Trial")["RR_intervals"]
    .agg(
        RRI_min_s="min",
        RRI_max_s="max",
        RRI_mean_s="mean",
        RRI_sd_s="std"
    )
    .round(2)
)

rt_interval_stats = (
    exploded_rt
    .groupby("Trial")["RT_intervals"]
    .agg(
        RT_min_s="min",
        RT_max_s="max",
        RT_mean_s="mean",
        RT_sd_s="std"
    )
    .round(2)
)

tr_interval_stats = (
    exploded_tr
    .groupby("Trial")["TR_intervals"]
    .agg(
        TR_min_s="min",
        TR_max_s="max",
        TR_mean_s="mean",
        TR_sd_s="std"
    )
    .round(2)
)

hr_stats = exploded_hr.groupby("Trial")["HR_bpm"].agg(
    HR_min_bpm = "min",
    HR_max_bpm = "max",
    HR_mean_bpm ="mean",
    HR_sd_bpm ="std"
).round(2)

trial_ecg_stats = pd.concat([rr_interval_stats, hr_stats, rt_interval_stats, tr_interval_stats], axis=1).reset_index()

print("--- ECG Statistics Grouped by Trial ---")
print(trial_ecg_stats)
