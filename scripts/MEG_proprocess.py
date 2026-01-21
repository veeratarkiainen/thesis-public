#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  1 20:05:00 2025

Preprocess EOG and ECG signals of all pilots

Save into raws_combined with correct channel types

@author: tarkiav1
"""

import os, mne, gc
import numpy as np
from mne.filter import filter_data

# %% Extract raw data

folder_out = "/m/nbe/scratch/artefact_sync/tarkiav1/"
data_folder = "/m/nbe/scratch/artefact_sync/triux/" #data available on Triton cluster, confidential

subjects = ["pilot01", "pilot02", "pilot03", "pilot04"]
dates = ["240216", "240625", "240625", "250127"]

# Dictionary to store combined raw objects for each subject
raws_combined = {}

for subj, date in zip(subjects, dates):

    subject_folder = f"/{subj}/{date}/raw/"

    run_files = [
        f"{data_folder}{subject_folder}{subj}_saccadelight_run-0{i}_raw_tsss_mc.fif"
        for i in range(1, 4)
    ]

    if subj == "pilot04":
        run_files = [
            f"{data_folder}{subject_folder}{subj}_task-saccadelight_run-0{i}_raw_tsss_mc.fif"
            for i in range(1, 4)
        ]

    # Read and concatenate runs
    raws = [mne.io.read_raw_fif(f, preload=True) for f in run_files]
    raw_combined = mne.concatenate_raws(raws)

    # Store the concatenated data
    raws_combined[subj] = raw_combined

out_preprocess = "preprocessed_signals"

# %% EOG Preprocessing

# 1) Define EOG channels per subject

eog_channel_names = {
    "pilot01": ["EOG002"],  # EOG002
    "pilot02": ["BIO003"],
    "pilot03": ["BIO003"],
    "pilot04": ["BIO002"],
}

veog_channel_names = {
    "pilot01": ["EOG001"],
    "pilot02": ["BIO002"],
    "pilot03": ["BIO002"],
    "pilot04": ["BIO001"],
}

ecg_channel_names = {
    "pilot02": ["BIO001"],
    "pilot03": ["BIO001"],
    "pilot04": ["BIO003"],
}


def preprocess_eog(
        raw: mne.io.BaseRaw,
        eog_names: list[str],
        l_freq: float,
        h_freq: float
):
    eog_raw = raw.copy().pick(eog_names)
    eog_raw.load_data()
    eog_raw.set_channel_types({ch: "eog" for ch in eog_names})
    eog_raw.filter(l_freq=l_freq, h_freq=h_freq, phase="zero", method="iir", picks='eog')

    return eog_raw


HEOG_signals = {}
VEOG_signals = {}

# Set channel types to Raw
for subj, raw in raws_combined.items():
    all_eog = eog_channel_names[subj] + veog_channel_names[subj]
    raw.set_channel_types({ch: "eog" for ch in all_eog})

for subj, raw in raws_combined.items():
    HEOG_signals[subj] = preprocess_eog(raw, eog_channel_names[subj], 0.1, 15.0)
    HEOG_signals[subj].save(f"{folder_out}{out_preprocess}/{subj}_heog_raw.fif", overwrite=True)
    del HEOG_signals[subj];
    gc.collect()

    VEOG_signals[subj] = preprocess_eog(raw, veog_channel_names[subj], 1.0, 80.0)
    VEOG_signals[subj].save(f"{folder_out}{out_preprocess}/{subj}_veog_raw.fif", overwrite=True)
    del VEOG_signals[subj];
    gc.collect()


# %% ECG Preprocessing

def preprocess_ecg(subj: str,
                   raw: mne.io.BaseRaw,
                   ecg_names: list[str],
                   l_freq: float = float,
                   h_freq: float = float
                   ):
    ecg_raw = raw.copy().pick(ecg_names)
    ecg_raw.load_data()  # small object (1–2 chs), safe to preload
    ecg_raw.set_channel_types({ch: "ecg" for ch in ecg_names})
    ecg_raw.filter(l_freq=l_freq, h_freq=h_freq, phase="zero", method="iir", picks="ecg")

    if subj == "pilot03":
        ecg_raw.apply_function(lambda data: -data, picks="ecg", channel_wise=True)

    return ecg_raw


# 1) pilot01: synthesize ECG from EOG, add as channel, then preprocess

raw_p1 = raws_combined["pilot01"]
eog_ch = "EOG002"

if "ECG_FROM_EOG" not in raw_p1.info["ch_names"]:
    tmp = raw_p1.copy().pick([eog_ch]).load_data()
    eog_ts = tmp.get_data()[0]
    sfreq = tmp.info["sfreq"]
    del tmp;
    gc.collect()

    ecg_like = mne.filter.filter_data(eog_ts, sfreq, l_freq=5.0, h_freq=35., phase="zero", method="iir", verbose=False)
    info = mne.create_info(["ECG_FROM_EOG"], sfreq, ch_types=["ecg"])
    ecg_signal = mne.io.RawArray(ecg_like[np.newaxis, :], info)

    # add a single tiny RawArray → minimal overhead
    raw_p1.add_channels([ecg_signal], force_update_info=True)
    raw_p1.set_channel_types({'ECG_FROM_EOG': 'ecg'})
    raws_combined["pilot01"] = raw_p1  # keep in master
    del ecg_like, ecg_signal;
    gc.collect()

raws_combined["pilot01"].save(f"{folder_out}{out_preprocess}/pilot01_raw.fif", overwrite=True)

# 3) Build one dict with filtered ECG for all pilots

ECG_signals = {}

# pilot01 via the synthetic ECG channel
ECG_signals["pilot01"] = preprocess_ecg("pilot01",
                                        raws_combined["pilot01"], ecg_names=["ECG_FROM_EOG"], l_freq=0.1, h_freq=45)
ECG_signals["pilot01"].save(f"{folder_out}{out_preprocess}/pilot01_ecg_raw.fif", overwrite=True)

# raws_combined["pilot01"].save(f"{out_preprocess}/pilot01_ecg_raw.fif", overwrite=True)

del ECG_signals["pilot01"], raws_combined["pilot01"];
gc.collect()

# 2) Other pilots: known ECG channel names
for subj, ecg_names in ecg_channel_names.items():
    raw = raws_combined[subj].set_channel_types({ch: "ecg" for ch in ecg_names})
    ECG_signals[subj] = preprocess_ecg(subj, raws_combined[subj], ecg_names, l_freq=0.1, h_freq=45)
    ECG_signals[subj].save(f"{folder_out}{out_preprocess}/{subj}_ecg_raw.fif", overwrite=True)
    raws_combined[subj].save(f"{folder_out}{out_preprocess}/{subj}_raw.fif", overwrite=True)

    del ECG_signals[subj], raws_combined[subj];
    gc.collect()