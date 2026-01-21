"""
Created on Sat Nov  1 21:31:42 2025

Perform ICA and save results of all pilots, pick ecg-related ICs

@author: tarkiav1
"""

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

# %%

import mne, os, gc
import numpy as np
from mne.preprocessing import ICA

folder_out = "/m/nbe/scratch/artefact_sync/tarkiav1/"
subjects = ["pilot01", "pilot02", "pilot03", "pilot04"]
out_ica = "ICA/ICA_sources"
os.makedirs(out_ica, exist_ok=True)

for pilot in subjects:

    raw = mne.io.read_raw_fif(f"{folder_out}preprocessed_signals/{pilot}_raw.fif", preload=True)

    filt_ica = (raw.copy()
                .pick(['mag', 'ecg', 'eog'])
                .load_data()
                .filter(1.0, 80, phase="zero", method="iir"))

    # Fit ICA
    ica = ICA(n_components=60, method='fastica', max_iter="auto", random_state=97)  # n_components eg. 30 enough?
    ica.fit(filt_ica, decim=2, reject_by_annotation=True)  # fastica, components = 60
    ica

    ica_fname = f"{folder_out}{out_ica}/{pilot}_ica.fif"
    ica.save(ica_fname, overwrite=True)

    # Save sources
    ica_sources = ica.get_sources(raw)
    ica_sources.save(f"{folder_out}{out_ica}/{pilot}_ica_sources_raw.fif", overwrite=True)

    var_ratio = ica.get_explained_variance_ratio(raw)
    print(f"Variance explained: {var_ratio}")

    ica.exclude = ica_ecg_inds[pilot]
    removed_var = ica.get_explained_variance_ratio(raw, components=ica.exclude)

    print("Variance removed:")
    for ch_type, v in removed_var.items():
        print(f"{ch_type}: {100 * v:.2f}%")

    print(f"Finished ICA for {pilot}")

    del filt_ica, raw;
    gc.collect()

# %% Plot for subject

raw = mne.io.read_raw_fif(f"{folder_out}preprocessed_signals/pilot04_raw.fif", preload=True)
ecg_raw = mne.io.read_raw_fif(f"{folder_out}preprocessed_signals/pilot04_ecg_raw.fif", preload=True)

ecg_picks = mne.pick_types(raw.info, ecg=True)
if len(ecg_picks) == 0:
    raise RuntimeError(f"No ECG channel in raws_combined[{pilot}]")
ecg_name = raw.info["ch_names"][ecg_picks[0]]
if ecg_name in raw.info["bads"]:
    raw.info["bads"].remove(ecg_name)

ecg_events, _, _ = mne.preprocessing.find_ecg_events(raw, ch_name=ecg_name, event_id=999, verbose=False)

picks_mag = mne.pick_types(raw.info, meg='mag', eeg=False, eog=False, ecg=False)
picks_ecg = mne.pick_types(raw.info, ecg=True)
picks_epochs = np.unique(np.r_[picks_mag, picks_ecg])

raw_filt = raw.copy()
raw_filt.filter(1.0, 80, picks=picks_mag, phase="zero", method="iir", verbose=False)  # cover
raw_filt.notch_filter(freqs=np.arange(50, 251, 50), picks=picks_mag, verbose=False)

ecg_epochs = mne.Epochs(
    raw_filt,
    ecg_events,
    event_id=999,
    tmin=-0.3, tmax=0.8,
    baseline=None,
    picks=picks_epochs,
    preload=False,  # preload = True to load data into main memory but Kernel dies
    reject_by_annotation=True,
)

read_ica = mne.preprocessing.read_ica("/m/nbe/scratch/artefact_sync/tarkiav1/ICA/ICA_sources//pilot04_ica.fif")
fig_comp = read_ica.plot_components(picks=range(4), show=False)
fig_comp.savefig("Thesis/Figures/pilot04_ICA_comp.pdf", dpi=300, bbox_inches="tight")
read_sources = mne.io.read_raw_fif("/m/nbe/scratch/artefact_sync/tarkiav1/ICA/ICA_sources//pilot04_ica_sources_raw.fif")
fig_sources = read_sources.plot()
# fig_sources.savefig("Thesis/Figures/pilot04_ICA_src.pdf", dpi=300, bbox_inches="tight")
# fig_prop = read_ica.plot_properties(raw, picks=[0], show=False)

read_ica.plot_properties(ecg_epochs, picks=[0])