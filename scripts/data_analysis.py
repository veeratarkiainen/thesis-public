from scripts.preprocessing import df_clean_signals, saccade_index_df, blink_index_df, ecg_index_df, resp_index_df
from utils.functions_histograms import *
from utils.functions_stat_testing import *

import numpy as np
import pandas as pd
import gc

"""
Time & phase domain analysis of eye movements in cardiac and respiratory cycles by Veera Tarkiainen 
"""

def debug_invalid_reasons(event_idx, r_idx, t_offsets_idx):
    e = np.asarray(event_idx, dtype=int)
    r = np.unique(np.asarray(r_idx, dtype=int))
    t = np.unique(np.asarray(t_offsets_idx, dtype=int))

    n_beats = len(r) - 1
    # Build T per beat (T in (R_i, R_{i+1}))
    T_per_beat = np.full(n_beats, np.nan)
    k = 0
    for i in range(n_beats):
        while k < len(t) and t[k] <= r[i]:
            k += 1
        if k < len(t) and t[k] < r[i+1]:
            T_per_beat[i] = t[k]
            k += 1

    i_prev = np.searchsorted(r, e, side='right') - 1
    outside = (i_prev < 0) | (i_prev >= n_beats)
    inside  = ~outside
    idx     = i_prev[inside]
    noT     = np.isnan(T_per_beat[idx])

    return {
        "total_events": len(e),
        "outside_any_RR": int(np.sum(outside)),
        "inside_but_missing_T": int(np.sum(noT)),
        "inside_with_T": int(np.sum(~noT)),
    }

def classify_events_systole_diastole_with_fallback(event_idx, r_idx, t_offsets_idx,
                                                   alpha_clip=(0.25, 0.45)):
    e = np.asarray(event_idx, dtype=int)
    r = np.unique(np.asarray(r_idx, dtype=int))
    t = np.unique(np.asarray(t_offsets_idx, dtype=int))

    labels = np.full(e.shape, 'invalid', dtype=object)
    beat_i = np.full(e.shape, -1, dtype=int)

    if e.size == 0 or r.size < 2:
        return pd.DataFrame({"event_idx": e, "beat": beat_i, "phase": labels}), {
            "n_events": int(e.size),
            "n_systole": 0, "n_diastole": 0, "n_invalid": int(e.size),
            "n_systole_equal_len": np.nan, "n_diastole_equal_len": np.nan,
            "scale_systole": np.nan, "scale_diastole": np.nan,
            "alpha_used": np.nan
        }

    # One T_end per beat: first T with R_i < T < R_{i+1}
    n_beats = len(r) - 1
    T_per_beat = np.full(n_beats, np.nan, float)
    k = 0
    for i in range(n_beats):
        while k < len(t) and t[k] <= r[i]:
            k += 1
        if k < len(t) and t[k] < r[i+1]:
            T_per_beat[i] = t[k]
            k += 1

    # Median alpha (fraction of RR to T), clipped; fill missing T_i = R_i + alpha*RR_i
    RR = r[1:] - r[:-1]
    with_T = ~np.isnan(T_per_beat)
    alpha = np.full(n_beats, np.nan, float)
    if np.any(with_T):
        alpha[with_T] = (T_per_beat[with_T] - r[:-1][with_T]) / np.maximum(RR[with_T], 1e-9)
        alpha_med = float(np.clip(np.nanmedian(alpha[with_T]), *alpha_clip))
    else:
        alpha_med = 0.35
    T_per_beat = np.where(with_T, T_per_beat, r[:-1] + alpha_med * RR)

    # Assign events to beats: R_i <= e < R_{i+1}
    i_prev = np.searchsorted(r, e, side='right') - 1
    valid  = (i_prev >= 0) & (i_prev < n_beats)
    ee     = e[valid]
    idx    = i_prev[valid]
    Ri     = r[idx]
    Rip1   = r[idx + 1]
    Ti_evt = T_per_beat[idx]

    # Labels
    is_syst = (ee >= Ri)   & (ee < Ti_evt)
    is_dias = (ee >= Ti_evt) & (ee < Rip1)

    # Phase durations per beat (same units as indices)
    syst_len_beats = np.maximum(T_per_beat - r[:-1], 0.0)   # shape (n_beats,)
    dias_len_beats = np.maximum(r[1:]      - T_per_beat, 0.0)

    # Totals across all beats
    T_syst_total = float(np.nansum(syst_len_beats))
    T_dias_total = float(np.nansum(dias_len_beats))
    T_total      = T_syst_total + T_dias_total
    T_equal      = 0.5 * T_total  # target equal duration per phase

    # Raw counts
    N_syst = int(is_syst.sum())
    N_dias = int(is_dias.sum())

    # Equal-length–adjusted counts (expected counts if both phases had T_equal duration)
    scale_syst = (T_equal / T_syst_total) if T_syst_total > 0 else np.nan
    scale_dias = (T_equal / T_dias_total) if T_dias_total > 0 else np.nan
    N_syst_equal = float(N_syst * scale_syst) if np.isfinite(scale_syst) else np.nan
    N_dias_equal = float(N_dias * scale_dias) if np.isfinite(scale_dias) else np.nan

    # Build per-event labels dataframe
    labels_valid = np.full(ee.shape, 'invalid', dtype=object)
    labels_valid[is_syst] = 'systole'
    labels_valid[is_dias] = 'diastole'
    labels[valid] = labels_valid
    beat_i[valid] = idx

    labels_df = pd.DataFrame({"event_idx": e, "beat": beat_i, "phase": labels})

    summary = {
        "n_events":             int(len(e)),
        "n_systole":            N_syst,
        "n_diastole":           N_dias,
        "n_invalid":            int(np.sum(labels == 'invalid')),
        "scale_systole":        scale_syst,
        "scale_diastole":       scale_dias,
        "n_systole_equal_len":  int(N_syst_equal),
        "n_diastole_equal_len": int(N_dias_equal),
        "alpha_used":           alpha_med,
    }
    return labels_df, summary

labels_s, summary_s = classify_events_systole_diastole_with_fallback(saccade_index_df["index"], ecg_index_df["r_peak_index"], ecg_index_df["t_wave_offset_index"])
print(f"Saccade-cardiac event classification results: {summary_s}")

labels_b, summary_b = classify_events_systole_diastole_with_fallback(blink_index_df["index"], ecg_index_df["r_peak_index"], ecg_index_df["t_wave_offset_index"])
print(f"Blink-cardiac event classification results: {summary_b}")

del labels_s, labels_b, summary_s, summary_b; gc.collect()

#%% R peak - saccade distances

def distance_prev_rpeak_before_saccade(saccades, r_peaks):

    r = np.asarray(r_peaks).astype(float)
    s = np.asarray(saccades).astype(float)

    #sort
    r = np.sort(r)
    s_idx = np.argsort(s)
    s_sorted = s[s_idx]

    # index of last R-peak strictly before each saccade
    idx_before = np.searchsorted(r, s_sorted, side = 'right') -1

    # build outputs in sorted order
    prev_idx_sorted = idx_before.copy()
    prev_vals_sorted = np.full_like(s_sorted, np.nan, dtype=float)
    dist_sorted = np.full_like(s_sorted, np.nan, dtype=float)

    valid = idx_before >= 0
    prev_vals_sorted[valid] = r[idx_before[valid]]
    dist_sorted[valid] = s_sorted[valid] - prev_vals_sorted[valid]

    # unsort back to original saccade order
    inv = np.empty_like(s_idx)
    inv[s_idx] = np.arange(len(s_idx))
    prev_idx = prev_idx_sorted[inv]
    prev_values = prev_vals_sorted[inv]

    prev_idx[~np.isfinite(prev_values)] = -1
    dist = dist_sorted[inv]
    dist_ms = dist * 1000

    return dist_ms

def distance_prev_saccade_before_rpeak(saccades, r_peaks):

    r = np.asarray(r_peaks, dtype=float)
    s = np.asarray(saccades, dtype=float)

    # sort saccades; keep R original order but compute on a sorted copy to vectorize
    s_sorted = np.sort(s)
    r_sorted_idx = np.argsort(r)
    r_sorted = r[r_sorted_idx]

    # index of last saccade strictly before each R peak
    idx_before_sorted = np.searchsorted(s_sorted, r_sorted, side='right') - 1
    # map back to original R order
    idx_before = np.empty_like(idx_before_sorted)
    idx_before[r_sorted_idx] = idx_before_sorted

    # build outputs aligned with original R order
    prev_idx = idx_before.copy()
    prev_vals = np.full_like(r, np.nan, dtype=float)
    dist = np.full_like(r, np.nan, dtype=float)

    valid = idx_before >= 0
    prev_vals[valid] = s_sorted[idx_before[valid]]
    dist[valid] = r[valid] - prev_vals[valid]   # R - previous saccade

    # No preceding saccade → index = -1
    prev_idx[~valid] = -1

    dist_ms = dist * 1000

    return dist_ms

def calculate_distances(df_clean, cardiac_index_df, event_index_df, event_name):

    subjects = sorted(df_clean["Subject"].unique().tolist())
    records = []

    for subject in subjects:
        subj_trials = sorted(df_clean.loc[df_clean["Subject"] == subject, "Trial"].unique().tolist())
        for j, trial in enumerate(subj_trials):

            data = df_clean[(df_clean["Subject"] == subject) & (df_clean["Trial"] == trial)].sort_values("Time")
            t    = data["Time"].to_numpy(dtype=float)
            L    = len(t)

            # r peaks
            c_rows = cardiac_index_df[(cardiac_index_df["Subject"] == subject) &
                                      (cardiac_index_df["Trial"] == trial)]
            if "event" in c_rows.columns:
                c_rows = c_rows[c_rows["event"].str.contains("cardiac", case=False, na=False)]

            r_idx = c_rows["r_peak_index"].to_numpy(dtype=int)
            r_idx = r_idx[(r_idx >= 0) & (r_idx < L)]
            r_times = t[r_idx]

            # events
            e_rows = event_index_df[(event_index_df["Subject"] == subject) &
                                    (event_index_df["Trial"] == trial) &
                                    (event_index_df["event"] == event_name)]
            e_idx = e_rows["index"].to_numpy(dtype=int)
            e_idx = e_idx[(e_idx >= 0) & (e_idx < L)]
            e_times = t[e_idx]

            distances_rpeak_prev_saccade = distance_prev_rpeak_before_saccade(e_times, r_times)
            distances_saccade_prev_rpeak = distance_prev_saccade_before_rpeak(e_times, r_times)

            records.append({
                "Subject": subject,
                "Trial": trial,
                "distance_r_prev_saccade": distances_rpeak_prev_saccade,
                "distance_saccade_prev_rpeak": distances_saccade_prev_rpeak
            })

    df = pd.DataFrame(records)
    return df

distances = calculate_distances(df_clean_signals, ecg_index_df, saccade_index_df, "saccade_onset")
exploded = distances.explode("distance_r_prev_saccade")

r_to_saccade_distance_stats = exploded.groupby("Trial")["distance_r_prev_saccade"].agg(
    Distance_min_ms = "min",
    Distance_max_ms = "max",
    Distance_mean_ms ="mean",
    Distance_std_ms ="std"
).round(2)

print("--- Distance from R peak to Saccade Statistics Grouped by Trial ---")
print(r_to_saccade_distance_stats)

del distances; gc.collect()

#%%

print("Saccade onsets in cardiac cycle...")
saccade_cardiac_stats_polar, saccade_cardiac_stats_timedom, saccade_cardiac_counts_timedom, saccade_cardiac_hist_info, saccade_cardiac_stats_pooled, saccade_cardiac_stats_mean = histograms_cardiac(
    df_clean_signals,
    ecg_index_df,
    saccade_index_df,
    "saccade_onset",
    "Sacc_vel",)

exploded_hist = saccade_cardiac_stats_polar.explode("p_value")

p_stats = exploded_hist.groupby("Trial")["p_value"].agg(
    P_value_min = "min",
    P_value_max = "max",
    P_value_mean ="mean",
    P_value_sd ="std"
).round(4)

print("--- Rayleigh P-value Statistics Grouped by Trial ---")
print(p_stats)

print(saccade_cardiac_counts_timedom)

#%%

print("Blink onsets in cardiac cycle...")
blink_cardiac_stats_polar, blink_cardiac_stats_timedom, blink_cardiac_counts_timedom, blink_cardiac_hist_info, blink_cardiac_stats_pooled, blink_cardiac_stats_mean  = histograms_cardiac(
    df_clean_signals,
    ecg_index_df,
    blink_index_df,
    "blink_onset",
    "Blink",
)

exploded_hist = blink_cardiac_stats_polar.explode("p_value")

p_stats = exploded_hist.groupby("Trial")["p_value"].agg(
    P_value_min = "min",
    P_value_max = "max",
    P_value_mean ="mean",
    P_value_sd ="std"
).round(4)

print("--- Rayleigh P-value Statistics Grouped by Trial ---")
print(p_stats)

#%%

print("Saccade onsets in respiratory cycle...")

saccade_resp_stats_polar, resp_saccade_phase_df = polar_histograms_respiration(df_clean_signals, resp_index_df, saccade_index_df, "saccade_onset", "Sacc_vel")

exploded_hist = saccade_resp_stats_polar.explode("P_value")

p_stats = exploded_hist.groupby("Trial")["P_value"].agg(
    P_value_min = "min",
    P_value_max = "max",
    P_value_mean ="mean",
    P_value_sd ="std"
).round(4)

print("--- Rayleigh P-value Statistics Grouped by Trial ---")
print(p_stats)

#%%

print("Blink onsets in respiratory cycle...")

blink_resp_stats_polar, resp_blink_phase_df  = polar_histograms_respiration(df_clean_signals, resp_index_df, blink_index_df, "blink_onset", "Blink")

exploded_hist = blink_resp_stats_polar.explode("P_value")

p_stats = exploded_hist.groupby("Trial")["P_value"].agg(
    P_value_min = "min",
    P_value_max = "max",
    P_value_mean ="mean",
    P_value_sd ="std"
).round(4)

print("--- Rayleigh P-value Statistics Grouped by Trial ---")
print(p_stats)

#%% Additional statistical testing

print("Time domain results (cardiac cycle only):\n")
perform_two_way_repeated_measures(saccade_cardiac_counts_timedom, label ="Saccade-cardiac")
perform_two_way_repeated_measures(blink_cardiac_counts_timedom, label="Blink-cardiac")

print("Phase domain results for the cardiac cycle:\n")
perform_watson_williams(saccade_cardiac_stats_polar, label = "Saccade-cardiac")
perform_watson_williams(blink_cardiac_stats_polar, label = "Blink-cardiac")

perform_friedman_test(saccade_cardiac_stats_polar, label = "Saccade-cardiac")
perform_friedman_test(blink_cardiac_stats_polar, label = "Blink_cardiac")

print("Phase domain results for the respiratory cycle:\n")
perform_watson_williams(saccade_resp_stats_polar, label="Saccade_respiratory")
perform_watson_williams(blink_resp_stats_polar, label="Blink_respiratory")

perform_friedman_test(saccade_resp_stats_polar, label = "Saccade-respiratory")
perform_friedman_test(blink_resp_stats_polar, label = "Blink-respiratory")

