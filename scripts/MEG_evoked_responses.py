"""
Created on Wed Oct 15 16:14:05 2025

Create EOG epochs, evoked responses and visualize all pilots

A) Original data (no ICA exclusion)
B) Cleaned data (cardiac IC exclusion)
C) ICs only

Reduction in RMS as a function of time and PVE comparison

@author: tarkiav1
"""

import os, mne, gc
import numpy as np
import matplotlib.pyplot as plt
from mne.preprocessing import read_ica
from mne.channels import read_layout

ica_ecg_inds = {
    'pilot01': [1],
    'pilot02': [0],
    'pilot03': [7],
    'pilot04': [0]
}

ica_eog_inds = {
    'pilot01': [0, 2],
    'pilot02': [1, 2],
    'pilot03': [12],
    'pilot04': [1, 7],
}

folder_out = "/m/nbe/scratch/artefact_sync/tarkiav1/"

# Dirs
out_dir = "Evoked_responses/epochs_and_evokeds"
out_figs = "Evoked_responses/evoked_figures"

subjects = ["pilot01", "pilot02", "pilot03", "pilot04"]

out = "figures"  # or "figures"

if out == "covariance": # different parameters for noise covariance matrices in source estimation
    tmin = -2.0
    tmax = 0.5
    baseline = (-1.5, -0.05)  # for evokeds

elif out == "figures": # evoked responses visualization
    tmin = -0.5
    tmax = 0.5
    baseline = (-0.5, -0.05)  # for evokeds

# %% Get HEOG epochs and evokeds

# EOG saccade thresholds
saccade_peak_thresh = {
    'pilot01': 1.5e-7,  # V
    'pilot02': 1.5e-7,
    'pilot03': 1.0e-7,
    'pilot04': 1.5e-7,
}
saccade_peak_dur_thresh = {
    'pilot01': 0.03,  # s
    'pilot02': 0.05,
    'pilot03': 0.04,
    'pilot04': 0.03,
}
saccade_peak_deriv_thresh = {
    'pilot01': 1.4e-6,  # V/s
    'pilot02': 0.6e-6,
    'pilot03': 0.8e-6,
    'pilot04': 1.0e-6,
}

blink_thresh = {
    'pilot01': 5.0,
    'pilot02': 15.0,
    'pilot03': 10.0,
    'pilot04': 10.0,
}

cond_to_event_id = {
    'left': 10,
    'right': 11,
}

right = 1
left = -1

epochs = []

for pilot in subjects:

    heog_raw = mne.io.read_raw_fif(f"{folder_out}preprocessed_signals/{pilot}_heog_raw.fif", preload=True)
    veog_raw = mne.io.read_raw_fif(f"{folder_out}preprocessed_signals/{pilot}_veog_raw.fif", preload=True)
    raw = mne.io.read_raw_fif(f"{folder_out}preprocessed_signals/{pilot}_raw.fif", preload=True)
    info = raw.info

    sfreq = heog_raw.info['sfreq']
    heog = heog_raw.get_data()[0]  # shape (n_samples,)
    times = heog_raw.times  # seconds, shape (n_samples,)
    times_der = (times[:-1] + times[1:]) / 2

    # first derivative
    heog_der = (np.diff(heog) / np.diff(times) / sfreq)
    heog_der_z = (heog_der - heog_der.mean()) / heog_der.std()

    # second derivative
    heog_der2 = (np.diff(heog_der) / np.diff(times_der) / sfreq)
    times_der2 = (times_der[:-1] + times_der[1:]) / 2
    heog_der2_z = (heog_der2 - heog_der2.mean()) / heog_der2.std()

    # Find peaks from the second derivative of the horizontal EOG

    right = -1 if pilot == 'pilot04' else 1
    left = -1 * right

    right_peak_loc_n, right_peak_mag = mne.preprocessing.peak_finder(
        heog_der2_z,
        thresh=3,
        extrema=right)

    left_peak_loc_n, left_peak_mag = mne.preprocessing.peak_finder(
        heog_der2_z,
        thresh=3,
        extrema=left)

    if pilot == 'pilot04':
        heog_der_z *= -1
        heog_der2_z *= -1

    # Threshold the peaks with regard to their length in terms of the first derivative
    peak_dur_thresh = int(saccade_peak_dur_thresh[pilot] * sfreq)
    right_peak_keep_indices = [peak_idx for peak_idx, peak_loc in enumerate(right_peak_loc_n)
                               if np.all(heog_der_z[peak_loc:peak_loc + peak_dur_thresh] > 1)]
    right_peak_loc_n_keep = right_peak_loc_n[right_peak_keep_indices]

    left_peak_keep_indices = [peak_idx for peak_idx, peak_loc in enumerate(left_peak_loc_n)
                              if np.all(heog_der_z[peak_loc:peak_loc + peak_dur_thresh] < -1)]
    left_peak_loc_n_keep = left_peak_loc_n[left_peak_keep_indices]

    # Correct for the sample indexing of the system
    right_peak_loc = times_der2[right_peak_loc_n_keep] + heog_raw.first_samp / sfreq
    left_peak_loc = times_der2[left_peak_loc_n_keep] + heog_raw.first_samp / sfreq

    # Create annotations for left and right saccades
    annot = (mne.Annotations(right_peak_loc, 0.6, 'right', orig_time=heog_raw.info["meas_date"])
             + mne.Annotations(left_peak_loc, 0.6, 'left', orig_time=heog_raw.info["meas_date"]))

    ### DROP BAD EPOCHS ? ###
    # for onset, dur in map_bad[subject][task][run]:
    #    annot.append(onset + raw.first_samp / rsreq, dur, 'bad')
    # task/run in one concat

    ### ANNOTATE BLINKS ###

    veog_name = 'MEG1411' if pilot == 'pilot01' else veog_raw.info["ch_names"][0]

    blink_ch = (raw.copy().pick([veog_name]) if veog_name in raw.info["ch_names"]
                else veog_raw.copy().pick([veog_name]))

    # blink_ch.set_channel_types({veog_name: 'eog'})
    blink_ch.apply_function(lambda x: (x - x.mean()) / x.std(), picks=[veog_name], channel_wise=True)

    blink_events = mne.preprocessing.find_eog_events(blink_ch, event_id=998, ch_name=veog_name,
                                                     thresh=blink_thresh[pilot])
    onsets = blink_events[:, 0] / blink_ch.info['sfreq'] - 0.15

    if len(onsets):
        blink_annot = mne.Annotations(onsets, 0.3, 'bad blink', orig_time=blink_ch.info["meas_date"])
        annot += blink_annot

    heog_raw.set_annotations(annot)
    raw.set_annotations(annot)

    # Pick sensors for visualization
    layout = read_layout("Vectorview-all")

    left_occ = mne.read_vectorview_selection('Left-occipital', info=info)
    right_occ = mne.read_vectorview_selection('Right-occipital', info=info)
    picks_post = left_occ + right_occ

    picks_clean = [ch.replace(' ', '') for ch in picks_post]
    picks_final = [ch for ch in picks_clean if ch in raw.ch_names]  # intersect

    posterior_raw = raw.copy().pick(picks_final)

    # indices
    posterior_idx = mne.pick_channels(raw.ch_names, picks_final)
    posterior_info = mne.pick_info(info, sel=posterior_idx)

    ### ONLY MAG SNESORS?

    picks_mag = mne.pick_types(info, meg='mag')  # indices of all mags
    mag_names = [raw.ch_names[i] for i in picks_mag]

    posterior_mag_names = [ch for ch in picks_clean if ch in mag_names]

    posterior_raw_mag = raw.copy().pick(posterior_mag_names)
    posterior_idx_mag = mne.pick_channels(raw.ch_names, posterior_mag_names)
    posterior_info_mag = mne.pick_info(info, sel=posterior_idx_mag)

    eog_picks = mne.pick_types(raw.info, eog=True)
    if len(eog_picks) == 0:
        raise RuntimeError(f"No EOG channel in raws_combined[{pilot}]")
    eog_name = raw.info["ch_names"][eog_picks[0]]
    if eog_name in raw.info["bads"]:
        raw.info["bads"].remove(eog_name)

    # Create saccade events from annotations
    eog_events, event_id = mne.events_from_annotations(raw, event_id=cond_to_event_id)

    # Read ICA file
    ica = read_ica(f"{folder_out}ICA/ICA_sources/{pilot}_ica.fif")

    # debug
    print(">>>", pilot)
    print("raw types:", raw.get_channel_types(unique=True))  # ecg
    print("EOG in raw:", [raw.info["ch_names"][i] for i in mne.pick_types(raw.info, eog=True)])  # ecg

    # Picks MEG
    picks_mag = mne.pick_types(raw.info, meg='mag', eeg=False, eog=False, ecg=True)
    picks_eog_ecg = mne.pick_types(raw.info, eog=True, ecg=True)
    picks_all = np.r_[picks_mag, picks_eog_ecg]  # use same for ICA & non-ICA
    picks_epochs = np.unique(np.r_[picks_mag, picks_eog_ecg])

    # Filter MEG
    raw_filt = raw.copy()
    raw_filt.filter(1.0, 80, picks=picks_mag, phase="zero", method="iir", verbose=False)  # cover
    raw_filt.notch_filter(freqs=np.arange(50, 251, 50), picks=picks_mag, verbose=False)

    ecg_name = raw.info['ch_names'][picks_eog_ecg[0]]
    if ecg_name in raw.info['bads']:
        raw.info['bads'].remove(ecg_name)

    # A) ORIGINAL (no ICA exclusion)

    epochs_orig = mne.Epochs(raw_filt,
                             eog_events,
                             event_id=event_id,
                             tmin=tmin,  # -0.5
                             tmax=tmax,
                             baseline=None,  # or (-0.25, -0.05)
                             picks=picks_epochs,
                             preload=False,
                             reject_by_annotation=True,
                             verbose=False)

    print("epochs types:", epochs_orig.get_channel_types(unique=True))  # expect ['mag','ecg']

    picks_mag = mne.pick_types(epochs_orig.info, meg='mag', ecg=True)
    picks_eog = mne.pick_types(epochs_orig.info, meg=False, eog=True)
    picks_meg_eog = np.r_[picks_mag, picks_eog]

    epochs_orig.save(f"{folder_out}{out_dir}/{pilot}_eog-epo.fif", overwrite=True)

    epochs_left_orig = epochs_orig["left"]
    epochs_right_orig = epochs_orig["right"]

    ev_left = epochs_left_orig.average()
    ev_right = epochs_right_orig.average()
    ev_comb = epochs_orig.average()

    for ev in (ev_left, ev_right, ev_comb):
        ev.apply_baseline(baseline)

    ev_left.save(f"{folder_out}{out_dir}/{pilot}_evoked_eog_left-ave.fif", overwrite=True)
    ev_right.save(f"{folder_out}{out_dir}/{pilot}_evoked_eog_right-ave.fif", overwrite=True)
    ev_comb.save(f"{folder_out}{out_dir}/{pilot}_evoked_eog_combined-ave.fif", overwrite=True)

    del epochs_orig, ev_left, ev_right, ev_comb
    gc.collect()

    # B) ICA CLEANED - EOG

    raw_clean = raw_filt.copy()
    ica.exclude = ica_ecg_inds[pilot]
    ica.apply(raw_clean)

    epochs_ica = mne.Epochs(raw_clean,
                            eog_events,
                            event_id=event_id,
                            tmin=tmin,
                            tmax=tmax,
                            baseline=None,
                            picks=picks_epochs,
                            preload=False,
                            reject_by_annotation=True,
                            verbose=False)

    epochs_ica.save(f"{folder_out}{out_dir}/{pilot}_eog_ICA-epo.fif", overwrite=True)
    epochs_left_ica = epochs_ica["left"]
    epochs_right_ica = epochs_ica["right"]

    ev_left_ica = epochs_left_ica.average()
    ev_right_ica = epochs_right_ica.average()
    ev_comb_ica = epochs_ica.average()

    # Apply same baseline
    for ev in (ev_left_ica, ev_right_ica, ev_comb_ica):
        ev.apply_baseline(baseline)

    ev_left_ica.save(f"{folder_out}{out_dir}/{pilot}_evoked_eog_left_ICA-ave.fif", overwrite=True)
    ev_right_ica.save(f"{folder_out}{out_dir}/{pilot}_evoked_eog_right_ICA-ave.fif", overwrite=True)
    ev_comb_ica.save(f"{folder_out}{out_dir}/{pilot}_evoked_eog_combined_ICA-ave.fif", overwrite=True)
    del epochs_ica, ev_left_ica, ev_right_ica, ev_comb_ica, raw_clean

    # C) Get only ECG ICs

    keep = ica_ecg_inds[pilot]
    drop = [i for i in range(ica.n_components_) if i not in keep]

    raw_IC_only = raw_filt.copy()
    ica.exclude = drop
    ica.apply(raw_IC_only)

    epochs_IC = mne.Epochs(raw_IC_only,
                           eog_events,
                           event_id=event_id,
                           tmin=tmin, tmax=tmax,
                           baseline=None,
                           picks=picks_epochs,
                           preload=False,
                           reject_by_annotation=True
                           )

    print("epochs types:", epochs_IC.get_channel_types(unique=True))  # expect ['mag','ecg']

    epochs_IC.save(f"{folder_out}{out_dir}/{pilot}_eog_ICs-epo.fif", overwrite=True)
    epochs_IC_loaded = epochs_IC.copy().load_data()

    # IMPORTANT to add number of averages
    n_epochs_C = len(epochs_IC_loaded)

    epochs_left_C = epochs_IC_loaded['left']
    epochs_right_C = epochs_IC_loaded['right']

    n_epochs_left_C = len(epochs_left_C)
    n_epochs_right_C = len(epochs_right_C)

    evoked_left_data_C = epochs_left_C.get_data(picks=picks_meg_eog).mean(axis=0)
    info_sel_left_C = mne.pick_info(epochs_left_C.info, picks_meg_eog, copy=True)
    evoked_left_C = mne.EvokedArray(evoked_left_data_C, info_sel_left_C, tmin=epochs_IC_loaded.tmin,
                                    comment="left_saccades_ICs_only", nave=n_epochs_left_C)

    evoked_right_data_C = epochs_right_C.get_data(picks=picks_meg_eog).mean(axis=0)
    info_sel_right_C = mne.pick_info(epochs_right_C.info, picks_meg_eog, copy=True)
    evoked_right_C = mne.EvokedArray(evoked_right_data_C, info_sel_right_C, tmin=epochs_IC_loaded.tmin,
                                     comment="right_saccades_ICs_only", nave=n_epochs_right_C)

    evoked_comb_data_C = epochs_IC_loaded.get_data(picks=picks_meg_eog).mean(axis=0)
    info_sel_comb_C = mne.pick_info(epochs_IC_loaded.info, picks_meg_eog, copy=True)
    evoked_comb_C = mne.EvokedArray(evoked_comb_data_C, info_sel_comb_C, tmin=epochs_IC_loaded.tmin,
                                    comment="combined_saccades_ICs_only", nave=n_epochs_C)

    for ev in (epochs_left_C, epochs_right_C, evoked_comb_C):
        ev.apply_baseline(baseline)

    evoked_left_C.save(f"{folder_out}{out_dir}/{pilot}_evoked_eog_left_ICs-ave.fif", overwrite=True)
    evoked_right_C.save(f"{folder_out}{out_dir}/{pilot}_evoked_eog_right_ICs-ave.fif", overwrite=True)
    evoked_comb_C.save(f"{folder_out}{out_dir}/{pilot}_evoked_eog_combined_ICs-ave.fif", overwrite=True)

    del epochs_IC_loaded, epochs_IC, epochs_left_C, epochs_right_C, evoked_left_C, evoked_right_C, evoked_comb_C, ica, raw_IC_only
    del heog_raw, veog_raw, raw;
    gc.collect()

# %% Plot evoked responses

out_fig_dir = "Thesis/Figures/Evokeds"
os.makedirs(out_fig_dir, exist_ok=True)

# define better times?
times_dict = {
    "pilot01": [-0.05, 0.0, 0.104, 0.212],
    "pilot02": [-0.05, 0.0, 0.098, 0.252],
    "pilot03": [-0.05, 0.0, 0.143, 0.206],
    "pilot04": [-0.05, -0.01, 0.146, 0.239],  # change rightward their own
}

times_dict_all = {
    "pilot01": [-0.05, 0.0, 0.104, 0.212],
    "pilot02": [0.006, 0.101, 0.255],
    "pilot03": [-0.05, 0.088, 0.143, 0.209, 0.240],
    "pilot04": [-0.05, -0.002, 0.065, 0.160, 0.250],  # change rightward their own
}

times_dict_ICs = {
    "pilot01": [-0.05, 0.0, 0.104, 0.212],
    "pilot02": [-0.215, -0.043, -0.004, 0.007],
    "pilot03": [-0.05, -0.003, 0.084, 0.134],
    "pilot04": [-0.329, -0.005, -0.002, 0.080, 0.175, 0.443],  # change rightward their own
}


# default: times = [-0.10, -0.05, 0.00, 0.02, 0.12, 0.15, 0.20, 0.28]

def evoked_rms(evoked, picks, global_scalar=False):
    """
    If global_scalar=False: Returns RMS over sensors for each time point (array).
    If global_scalar=True:  Returns a single scalar RMS for the entire epoch.
    """
    data = evoked.data[picks]
    if global_scalar:
        # Standard deviation/RMS of all data points combined
        return np.sqrt(np.mean(data ** 2))
    else:
        # RMS across sensors at each time point
        return np.sqrt(np.mean(data ** 2, axis=0))


def plot_rms_comparison_onefig(evoked_orig, evoked_ica, *, title, outpath=None):
    # Pick mag channels
    picks_mag = mne.pick_types(evoked_orig.info, meg='mag')

    # 1. Get RMS arrays for plotting (Function of Time)
    rms_orig_arr = evoked_rms(evoked_orig, picks_mag, global_scalar=False)
    rms_ica_arr = evoked_rms(evoked_ica, picks_mag, global_scalar=False)

    # 2. Get Global Scalars for PVE reporting
    rms_orig_global = evoked_rms(evoked_orig, picks_mag, global_scalar=True)
    rms_ica_global = evoked_rms(evoked_ica, picks_mag, global_scalar=True)

    # Calculate Global PVE (Scalar)
    global_pve = (1 - (rms_ica_global ** 2 / rms_orig_global ** 2)) * 100

    # Plotting
    times = evoked_orig.times
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(times * 1e3, rms_orig_arr * 1e15, label='RMS before cardiac ICA', color='tab:red', lw=2)
    ax.plot(times * 1e3, rms_ica_arr * 1e15, label='RMS after cardiac ICA', color='tab:blue', lw=2)

    ax.axvline(0, color='k', ls='--', lw=2, label='Saccade onset')
    ax.set_xlabel('Time (ms)', fontsize=14, fontweight="bold")
    ax.set_ylabel('RMS (fT)', fontsize=14, fontweight="bold")  # Note: *1e15 converts Tesla to Femtotesla
    ax.set_title(f"{title}", fontsize=16, fontweight="bold")
    ax.legend(frameon=False, loc='upper left', fontsize=14)
    ax.grid(axis="both", alpha=0.3)

    ax.set_xticks([-400, -200, 0, 200, 400]);
    ax.set_yticks([0, 20, 40, 60, 80, 100, 120, 140, 160])
    ax.set_xticklabels(ax.get_xticks(), fontsize=14);
    ax.set_yticklabels(ax.get_yticks(), fontsize=14)

    plt.tight_layout()
    if outpath:
        fig.savefig(outpath, dpi=300)
    plt.show()

    # Return the scalars so the print statement works
    return rms_orig_global, rms_ica_global, global_pve


lock_txt = "MEG evoked responses time-locked to saccade onset"
sensor_txt = "Magnetometers"  # or "Occipital magnetometers" if you want


def title_evoked(pilot, direction, cleaning):
    # direction: "Leftward", "Rightward", "All"
    # cleaning:  "Original", "ICA-cleaned", "Cardiac IC only"
    return f"{lock_txt} {cleaning}"


for pilot in subjects:

    if out == "covariance":
        continue

    if pilot != "pilot02":
        continue

    e_left_orig = mne.read_evokeds(f"{folder_out}{out_dir}/{pilot}_evoked_eog_left-ave.fif", condition=0)
    e_right_orig = mne.read_evokeds(f"{folder_out}{out_dir}/{pilot}_evoked_eog_right-ave.fif", condition=0)

    e_left_ica = mne.read_evokeds(f"{folder_out}{out_dir}/{pilot}_evoked_eog_left_ICA-ave.fif", condition=0)
    e_right_ica = mne.read_evokeds(f"{folder_out}{out_dir}/{pilot}_evoked_eog_right_ICA-ave.fif", condition=0)

    e_left_C = mne.read_evokeds(f"{folder_out}{out_dir}/{pilot}_evoked_eog_left_ICs-ave.fif", condition=0)
    e_right_C = mne.read_evokeds(f"{folder_out}{out_dir}/{pilot}_evoked_eog_right_ICs-ave.fif", condition=0)

    times = times_dict[pilot]  # seconds
    time_inds = [(np.abs(e_left_orig.times - t)).argmin() for t in times]

    chs = np.isin(e_left_orig.ch_names, picks_final)

    mask = np.zeros(e_left_orig.data.shape, dtype=bool)
    for ti in time_inds:
        mask[chs, ti] = True

    mask_params = dict(markersize=10, markerfacecolor="r")

    # A) ORIGINAL

    fig1 = e_left_orig.plot_joint(
        times='peaks', picks="mag",
        title=title_evoked(pilot, "Leftward", "Original"),
        show=False)

    fig2 = e_right_orig.plot_joint(
        times='peaks', picks="mag",
        title=title_evoked(pilot, "Rightward", "Original"),
        show=False)

    fig3 = e_left_orig.plot_topomap(
        times=times, time_unit="s", mask=mask, mask_params=mask_params,
        show=False)

    # fig3.axes[0].set_title(title_evoked(pilot, "Leftward", "Original"), fontsize=12)

    fig4 = e_right_orig.plot_topomap(times=times, time_unit="s", show=False)
    # fig4.axes[0].set_title(title_evoked(pilot, "Rightward", "Original"), fontsize=12)

    # B) CLEANED

    fig5 = e_left_ica.plot_joint(
        times=times, picks="mag",
        title=title_evoked(pilot, "leftward", "ICA-cleaned"),
        show=False)

    fig6 = e_right_ica.plot_joint(
        times=times, picks="mag",
        title=title_evoked(pilot, "rightward", "ICA-cleaned"),
        show=False)

    fig7 = e_left_ica.plot_topomap(times=times, time_unit="s", show=False)
    # fig7.axes[0].set_title(title_evoked(pilot, "Leftward", "ICA-cleaned"), fontsize=12)

    fig8 = e_right_ica.plot_topomap(times=times, time_unit="s", show=False)
    # fig8.axes[0].set_title(title_evoked(pilot, "Rightward", "ICA-cleaned"), fontsize=12)

    # C) ICs only
    fig9 = e_left_C.plot_joint(
        times='peaks', picks="mag",
        title=title_evoked(pilot, "leftward", "Cardiac ICs only"),
        show=False)

    fig10 = e_right_C.plot_joint(
        times='peaks', picks="mag",
        title=title_evoked(pilot, "rightward", "Cardiac ICs only"),
        show=False)

    # fig11 = e_left_C.plot_topomap(times='peaks', time_unit="s", show=False)
    # fig11.axes[0].set_title(title_evoked(pilot, "Leftward", "Cardiac IC only"), fontsize=12)

    # fig12 = e_right_C.plot_topomap(times='peaks', time_unit="s", show=False)
    # fig12.axes[0].set_title(title_evoked(pilot, "Rightward", "Cardiac IC only"), fontsize=12)

    # COMBINED ORIGINAL AND CLEANED
    e_combined_orig = mne.read_evokeds(f"{folder_out}{out_dir}/{pilot}_evoked_eog_combined-ave.fif", condition=0)
    e_combined_ica = mne.read_evokeds(f"{folder_out}{out_dir}/{pilot}_evoked_eog_combined_ICA-ave.fif", condition=0)
    e_combined_ICs = mne.read_evokeds(f"{folder_out}{out_dir}/{pilot}_evoked_eog_combined_ICs-ave.fif", condition=0)

    fig13 = e_combined_orig.plot_joint(
        times=times_dict_all[pilot], picks="mag",
        title=title_evoked(pilot, "all", "before cardiac ICA"),
        show=False)

    fig14 = e_combined_ica.plot_joint(
        times=times_dict_all[pilot], picks="mag",
        title=title_evoked(pilot, "all", "after cardiac ICA"),
        show=False)

    fig15 = e_combined_ICs.plot_joint(
        times='peaks', picks="mag",
        title=title_evoked(pilot, "all", "for the cardiac component"),
        show=False)

    for fig in [fig13, fig14, fig15]:
        fig.canvas.draw()  # Force the renderer to calculate the colorbar
        # Optional: explicitly find the colorbar axes and force visibility
        for ax in fig.get_axes():
            if ax.get_label() == "<colorbar>":  # MNE labels colorbars this way
                ax.set_visible(True)

    # save figs
    fig1.savefig(f"{out_fig_dir}/{pilot}_left_joint.pdf", dpi=300, bbox_inches="tight")
    fig2.savefig(f"{out_fig_dir}/{pilot}_right_joint.pdf", dpi=300, bbox_inches="tight")
    fig3.savefig(f"{out_fig_dir}/{pilot}_left_topomap_with_sensors.pdf", dpi=300, bbox_inches="tight")
    fig4.savefig(f"{out_fig_dir}/{pilot}_right_topomap.pdf", dpi=300, bbox_inches="tight")
    fig5.savefig(f"{out_fig_dir}/{pilot}_left_ica_joint.pdf", dpi=300, bbox_inches="tight")
    fig6.savefig(f"{out_fig_dir}/{pilot}_right_ica_joint.pdf", dpi=300, bbox_inches="tight")
    fig7.savefig(f"{out_fig_dir}/{pilot}_left_ica_topomap.pdf", dpi=300, bbox_inches="tight")
    fig8.savefig(f"{out_fig_dir}/{pilot}_right_ica_topomap.pdf", dpi=300, bbox_inches="tight")
    fig9.savefig(f"{out_fig_dir}/{pilot}_left_ica_only_joint.pdf", dpi=300, bbox_inches="tight")
    fig10.savefig(f"{out_fig_dir}/{pilot}_right_ica_only_joint.pdf", dpi=300, bbox_inches="tight")
    # fig11.savefig(f"Thesis/Figures/{pilot}_left_ica_only_topomap.pdf", dpi=300, bbox_inches="tight")
    # fig12.savefig(f"Thesis/Figures/{pilot}_right_ica_only_topomap.pdf", dpi=300, bbox_inches="tight")
    fig13.savefig(f"{out_fig_dir}/{pilot}_combined_joint.pdf", dpi=300, bbox_inches="tight")
    fig14.savefig(f"{out_fig_dir}/{pilot}_combined_ica_joint.pdf", dpi=300, bbox_inches="tight")
    fig15.savefig(f"{out_fig_dir}/{pilot}_combined_ica_only.pdf", dpi=300, bbox_inches="tight")

    rms_o, rms_i, pve = plot_rms_comparison_onefig(
        e_combined_orig,
        e_combined_ica,
        title="RMS of evoked responses before and after cardiac ICA",
        outpath=f"{out_fig_dir}/{pilot}_rms_comparison.pdf"
    )

    print(f"{pilot} global PVE={pve:.2f}%")

    del e_left_orig, e_right_orig, e_left_ica, e_right_ica, e_combined_orig, e_combined_ica
    gc.collect()