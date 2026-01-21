import os
from pathlib import Path
import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt
from scipy.stats import circmean, circstd, circvar
from utils.histogram_helpers import *

"""
Functions for time & phase domain histogram visualization. Results individually and at group-level 
by pooled counts and bootstrapped grand average by Veera Tarkiainen 
"""

BASE_DIR = r"C:\Users\Veera\Aalto\Thesis"
os.chdir(BASE_DIR)
output_dir = 'results/figures'
os.makedirs(output_dir, exist_ok=True)

def histograms_cardiac(
    df_clean,
    cardiac_index_df,
    event_index_df,
    event_name,
    sub_folder,
    num_bins_polar = 12,
    bootstrap_B = 2000,
    bootstrap_alpha = 0.05,
    seed = 0,
):

    all_trials  = sorted(df_clean["Trial"].unique().tolist())
    subjects    = sorted(df_clean["Subject"].unique().tolist())
    trial_names = {1:"Control", 2:"External", 3:"Internal", 4:"Self-paced"}

    # Polar bins (fixed)
    polar_edges   = np.linspace(0, 2*np.pi, num_bins_polar + 1)
    polar_centers = (polar_edges[:-1] + polar_edges[1:]) / 2
    bin_w = 2*np.pi / num_bins_polar

    # Collectors for summaries
    trial_hist_polar = {trial: [] for trial in all_trials}   # per-subject polar counts
    trial_delta_ms_raw = {trial: [] for trial in all_trials} # per-subject Δt arrays for R-locked
    trial_r_t = {trial: [] for trial in all_trials}          # per-subject {r,t}

    # DataFrames
    circ_stats = []
    timedom_stats = []
    counts_timedom = []
    histogram_info_polar = []

    fontsize = 24

    # SUBJECT FIGURES
    for subject in subjects:
        # skip specific subjects with bad data
        if event_name == "saccade_onset" and subject == "Sub07":
            continue
        if event_name == "blink_onset" and subject == "Sub01":
            continue

        subj_trials = sorted(df_clean.loc[df_clean["Subject"] == subject, "Trial"].unique().tolist())

        plt.close('all')
        fig = plt.figure(figsize=(5.5 * len(subj_trials), 9.2))
        axs = np.empty((2, len(subj_trials)), dtype=object)
        for j in range(len(subj_trials)):
            axs[0, j] = fig.add_subplot(2, len(subj_trials), 1 + j, projection='polar')
            axs[1, j] = fig.add_subplot(2, len(subj_trials), 1 + len(subj_trials) + j)

        for j, trial in enumerate(subj_trials):
            ax_p = axs[0, j]
            ax_t = axs[1, j]

            data = df_clean[(df_clean["Subject"] == subject) & (df_clean["Trial"] == trial)].sort_values("Time")
            t    = data["Time"].to_numpy(dtype=float)
            L    = len(t)

            # cardiac indices
            c_rows = cardiac_index_df[(cardiac_index_df["Subject"] == subject) &
                                      (cardiac_index_df["Trial"] == trial)]
            if "event" in c_rows.columns:
                c_rows = c_rows[c_rows["event"].str.contains("cardiac", case=False, na=False)]

            r_idx = c_rows["r_peak_index"].to_numpy(dtype=int)
            t_idx = c_rows["t_wave_offset_index"].to_numpy(dtype=int)
            r_idx = r_idx[(r_idx >= 0) & (r_idx < L)]
            t_idx = t_idx[(t_idx >= 0) & (t_idx < L)]
            r_times = t[r_idx]
            t_offs  = t[t_idx]

            # store for later trial-level template
            trial_r_t[trial].append({'r': r_times, 't': t_offs})

            # subject-specific R-locked bins (ok for per-subject panels)
            wins_subj, RR_med_ms, T_med_ms, alpha_med, s_frac, d_frac, phase_lbls, scale_syst, scale_dias = new_windows_relative_to_cycle(r_times, t_offs, K=6)

            if not wins_subj:
                trial_hist_polar[trial].append(np.zeros(num_bins_polar, int))
                ax_p.set_axis_off()
                ax_p.set_title(f"{trial_names.get(trial, f'Trial {trial}')} - Not enough events", fontsize=fontsize, pad=24)
                ax_t.set_axis_off()
                ax_t.set_title(f"{trial_names.get(trial, f'Trial {trial}')} - Not enough events", fontsize=fontsize, pad=24)
                continue

            K_subj = len(wins_subj)
            edges_subj = np.array([wins_subj[0][0]] + [hi for (_, hi) in wins_subj], float)
            edges_subj = sanitize_edges(edges_subj)
            labels_subj = [f"{int(round(a))}–{int(round(b))} ms" for (a, b) in wins_subj]

            # events
            e_rows = event_index_df[(event_index_df["Subject"] == subject) &
                                    (event_index_df["Trial"] == trial) &
                                    (event_index_df["event"] == event_name)]
            e_idx = e_rows["index"].to_numpy(dtype=int)
            e_idx = e_idx[(e_idx >= 0) & (e_idx < L)]

            if e_idx.size < 5 or r_times.size < 2:
                trial_hist_polar[trial].append(np.zeros(num_bins_polar, int))
                ax_p.set_axis_off()
                ax_p.set_title(f"{trial_names.get(trial, f'Trial {trial}')} - Not enough events", fontsize=fontsize, pad=24)
                ax_t.set_axis_off()
                ax_t.set_title(f"{trial_names.get(trial, f'Trial {trial}')} - Not enough events", fontsize=fontsize)
                continue

            e_times = t[e_idx]

            # POLAR PER SUBJECT
            theta = phase_theta_rt(e_times, r_times, t_offs)
            # scale by scaling factors
            scale_weights = np.where(theta < np.pi, scale_syst, scale_dias)
            hist_polar, _ = np.histogram(theta, bins=polar_edges, weights=scale_weights)
            trial_hist_polar[trial].append(hist_polar)
            n_systole  = int((theta <  np.pi).sum())
            n_diastole = int((theta >= np.pi).sum())
            n_polar = n_systole + n_diastole

            if hist_polar.sum() > 0:
                # Circular stats
                mu = circmean(theta, high=2*np.pi, low=0)
                circ_var = circvar(theta, high=2*np.pi, low=0)
                circ_std = circstd(theta, high=2*np.pi, low=0)

                N = theta.size
                C = np.mean(np.cos(theta))           # mean cos
                S = np.mean(np.sin(theta))           # mean sin
                R = np.hypot(C, S)                   # || \bar r ||, the mean resultant length
                Rn = R*N
                z,p = rayleigh_test(Rn, N)

                # Circular "mode" as modal *bin* center (robust for continuous data)
                mode_bin = np.argmax(hist_polar)
                md = polar_centers[mode_bin]

                circ_stats.append({
                "Subject": subject,
                "Trial": trial,
                "Circular_mean_rad": mu,
                "Circular_mean_deg": float(np.degrees(mu)),
                "Circular_std": float(circ_std),
                "Circular_var": float(circ_var),
                "Resultant_vector_length": float(R),
                "Mode_rad": md,
                "Mode_deg": float(np.degrees(md)),
                "N_events": n_polar,
                "N_systole": n_systole,
                "N_diastole": n_diastole,
                "Scale_systole": scale_syst,
                "Scale_diastole": scale_dias,
                "alpha": alpha_med,
                "p_value": p
            })

            for k in range(len(hist_polar)):
                histogram_info_polar.append({
                    "Subject":   subject,
                    "Trial":     trial,
                    "Event_index": e_idx,
                    "Theta": theta,
                    "Polar_edges": polar_edges,
                })

                # PLOTTING POLAR HISTOGRAMS
                bars = ax_p.bar(polar_centers, hist_polar, width=bin_w, edgecolor="black", alpha=1.0)
                for b in bars:
                    b.set_facecolor("white")

                bars[mode_bin].set_facecolor("grey")
                bars[mode_bin].set_edgecolor("grey")

                rmx = max(hist_polar.max(), 1)

                """if p < 0.01:
                    ax_p.text(1.25, 0.02, f"Rayleigh p={p:.3g}", transform=ax_p.transAxes, ha='right', va='bottom', fontsize=14)
                else:
                    ax_p.text(1.25, 0.02, f"Rayleigh p={p:.2g}", transform=ax_p.transAxes, ha='right', va='bottom', fontsize=14)"""

                ax_p.annotate("", xy=(mu, rmx), xytext=(mu, 0.0),
                              arrowprops=dict(arrowstyle="-|>", color="red", lw=2))

            ax_p.set_title(f"{trial_names.get(trial, f'Trial {trial}')} (events = {n_polar})", fontsize=fontsize, pad=24)
            ax_p.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
            ax_p.set_xticklabels(["0", "π/2", "π", "3π/2"], fontsize=14)
            ax_p.set_yticklabels([]); ax_p.grid(True)

            print(f"{subject}-{event_name}-{trial}: Rayleigh p={p:.3g}")

            # Time domain
            delta_ms = delta(e_times, r_times)
            trial_delta_ms_raw[trial].append(delta_ms.copy())

            counts_subj, _ = np.histogram(delta_ms, bins=edges_subj)
            N_systole  = int(np.sum(counts_subj * s_frac))
            N_diastole = int(np.sum(counts_subj * d_frac))

            K_subj = len(edges_subj) - 1
            centers_subj = 0.5 * (edges_subj[:-1] + edges_subj[1:])

            mean_ms = float(np.mean(delta_ms))
            x_mean = np.interp(mean_ms, centers_subj, np.arange(K_subj))
            x_T = np.interp(T_med_ms, centers_subj, np.arange(K_subj))
            mode_bin = int(np.argmax(counts_subj))
            x_mode = float(mode_bin)

            # variance, std?
            timedom_stats.append({
                "Subject": subject, "Trial": trial,
                "Mean": x_mean,
                "Mode": x_mode,
                "N_events": len(counts_subj),
                "N_systole": N_systole,
                "N_diastole": N_diastole,
                "Scale_systole": scale_syst,
                "Scale_diastole": scale_dias
            })

            centers_subj = 0.5 * (edges_subj[:-1] + edges_subj[1:])
            for k in range(len(counts_subj)):

                counts_timedom.append({
                    "Subject":   subject,
                    "Trial":     trial,
                    "bin_idx":   k,                            # 0..5
                    "count":     int(counts_subj[k]),         # raw count (not normalized)
                    "bin_lo_ms": float(edges_subj[k]),
                    "bin_hi_ms": float(edges_subj[k+1]),
                    "center_ms": float(centers_subj[k]),
                    "label":     labels_subj[k],              # e.g., "0–137 ms"
                })

            n_rlocked = int(counts_subj.sum())
            bars = ax_t.bar(np.arange(K_subj), counts_subj, width=0.8, edgecolor="black", alpha=1.0)

            for b in bars:
                b.set_facecolor("white")
            bars[mode_bin].set_facecolor("grey")
            bars[mode_bin].set_edgecolor("grey")

            # PLOTTING TIME DOMAIN HISTOGRAMS
            err = ax_t.errorbar(np.arange(K_subj), counts_subj, yerr=np.sqrt(np.maximum(counts_subj, K_subj)),fmt='none', lw=1.0, color = 'black', capsize=5, capthick=1.0, label="Poisson error")

            ax_t.axvline(x_T, ls='--', lw=2, color='black', label="T wave offset (end of systole)")

            ax_t.set_title(f"{trial_names.get(trial, f'Trial {trial}')}", fontsize=fontsize)
            ax_t.set_xticks(np.arange(K_subj))
            ax_t.set_xticklabels(labels_subj, rotation=20, fontsize=14)

            # plot count only once
            if event_name == "blink_onset":
                if j == 0:
                    ax_t.set_ylabel("Count", fontsize=14)
                    ax_t.tick_params(axis="y", labelsize=14)

                else:
                    ax_t.set_ylabel("")     # remove label
                    ax_t.tick_params(axis="y", labelsize=14)

            if event_name == "saccade_onset":
                if j == 1:
                    ax_t.set_ylabel("Count", fontsize=14)
                    ax_t.tick_params(axis="y", labelsize=14)
                else:
                    ax_t.set_ylabel("")     # remove label
                    ax_t.tick_params(axis="y", labelsize=14)

            ax_t.grid(axis="y", alpha=0.3)
            ax_t.set_ylim(bottom=0)

        """handles_p = make_polar_row_handles()
        put_row_legend(fig, axs[0, :], handles_p)

        handles_t = make_timedom_row_handles()
        put_row_legend(fig, axs[1, :], handles_t)"""

        fig.suptitle(f"Subject {subject} — {event_name.replace('_',' ')} in cardiac cycle (phase & time domain)",fontsize=20, fontweight="bold", y=0.99)

        filename = f"{output_dir}/histograms/Cardiac/{sub_folder}/{subject}_{event_name}.png"
        plt.savefig(filename)
        print(f"Figure saved to {filename}!")
        #plt.show()
        plt.close()

    circ_stats_df = pd.DataFrame(circ_stats)
    timedom_stats_df = pd.DataFrame(timedom_stats)
    counts_timedom_df = pd.DataFrame(counts_timedom)
    histogram_info_polar_df = pd.DataFrame(histogram_info_polar)

    """if sub_folder == "Sacc_vel":
        filename = f"{out_preprocess}/{event_name}_cardiac_vel.pkl"

    elif sub_folder == "Sacc_acc":
        filename = f"{out_preprocess}/{event_name}_cardiac_acc.pkl"

    elif sub_folder == "Blink":
        filename = f"{out_preprocess}/{event_name}_cardiac_acc.pkl"

    # Save to file
    with open(filename, "wb") as f:
        pickle.dump({
            f"{event_name}_cardiac_polar": circ_stats_df,
            f"{event_name}_cardiac_time": counts_timedom_df,
        }, f)

    print(f"Saved cardiac data → {filename}")"""

    # POOLED PHASE DOMAIN
    plot_trials = []

    # Build one set of edges per trial from ALL subjects
    trial_windows_ms   = {}
    trial_bin_edges_ms = {}
    trial_bin_labels   = {}
    trial_T_star       = {}

    summary_stats_pooled = []
    summary_stats_mean = []

    for trial in all_trials:
        wins, RR_star, T_star, alpha_star, s_frac, d_frac, phase_lbls = new_windows_summary(trial_r_t[trial])
        trial_T_star[trial] = T_star

        summary_edges = np.array([wins[0][0]] + [hi for (_, hi) in wins], float)
        summary_edges = sanitize_edges(summary_edges)
        trial_windows_ms[trial]   = wins
        trial_bin_edges_ms[trial] = summary_edges
        trial_bin_labels[trial]   = [f"{int(round(a))}–{int(round(b))}" for (a,b) in zip(summary_edges[:-1], summary_edges[1:])]

        Hpol = np.asarray(trial_hist_polar.get(trial, []), dtype=float)
        has_polar = (Hpol.size > 0) and (Hpol.sum() > 0)

        td = trial_delta_ms_raw.get(trial, [])
        has_time = (td is not None) and any(len(arr) > 0 for arr in td)

        if has_polar or has_time:
            plot_trials.append(trial)

    nT = len(plot_trials)

    fig_pool, axs_pool = plt.subplots(2, nT, subplot_kw={'projection':'polar'}, squeeze=False,
                                      figsize=(5.6*nT, 2*6.3))
    # bottom row cartesian
    for j in range(nT):
        fig_pool.delaxes(axs_pool[1, j])
        axs_pool[1, j] = fig_pool.add_subplot(2, nT, nT + j + 1)

    for j, trial in enumerate(plot_trials):
        # pooled polar density
        Hpol = np.asarray(trial_hist_polar.get(trial, []), dtype=float)
        entry = {"trial": trial, "polar": None, "timedom": None}
        ax = axs_pool[0, j]

        if Hpol.size and Hpol.sum() > 0:
            pooled_counts = Hpol.sum(axis=0)
            tot_polar = int(pooled_counts.sum())
            density = (pooled_counts / max(tot_polar, 1)) / bin_w

            mu, R = circ_from_density(density, polar_centers, bin_w)
            mode_bin = int(np.argmax(pooled_counts))
            md = float(polar_centers[mode_bin])
            rmx = max(density.max(), 1e-9)

            N = pooled_counts.size
            C = np.mean(np.cos(pooled_counts))           # mean cos
            S = np.mean(np.sin(pooled_counts))           # mean sin
            R = np.hypot(C, S)                   # || \bar r ||, the mean resultant length
            Rn = R*N
            z, p_value = rayleigh_test(Rn, N)

            syst_mask = polar_centers < np.pi
            n_systole  = int(pooled_counts[syst_mask].sum())
            n_diastole = int(pooled_counts[~syst_mask].sum())

            entry["polar"] = {
            "pooled_counts": pooled_counts.copy(),
            "pooled_density": density.copy(),
            "bin_edges": polar_edges.copy(),
            "bin_centers": polar_centers.copy(),
            "bin_width": float(bin_w),
            "N_total": tot_polar,
            "circ_mean": float(mu),
            "resultant_length": float(R),
            "mode_angle": md,
            "phase_counts": {"systole": n_systole, "diastole": n_diastole},
            "p_value": p_value
        }

            # PLOTTING POOLED PHASE
            bars = ax.bar(polar_centers, density, width=bin_w, linewidth=2, edgecolor="black", alpha=1.0)

            for b in bars:
                b.set_facecolor("white")

            bars[mode_bin].set_facecolor("grey")
            bars[mode_bin].set_edgecolor("grey")

            """if p_value < 0.01:
                ax.text(1.20, 0.00, f"p={p_value:.3g}", transform=ax.transAxes, ha='right', va='bottom', fontsize=20)

            else:
                ax.text(1.20, 0.00, f"p={p_value:.2g}", transform=ax.transAxes, ha='right', va='bottom', fontsize=20)"""

            ax.annotate("", xy=(mu, rmx), xytext=(mu, 0.0),
                        arrowprops=dict(arrowstyle="-|>", color="red", lw=4))

            ax.set_title(f"{trial_names.get(trial, f'Trial {trial}')} (events={tot_polar})", fontsize=fontsize, pad=30)
            ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
            ax.set_xticklabels(["0", "π/2", "π", "3π/2"], fontsize=20)
            ax.set_yticklabels([]); ax.grid(True)

            print(f"Pooled-{event_name}-{trial}: Rayleigh p={p_value:.3g}")

        """else:
            tot_polar = 0
            ax.set_axis_off()
            ax.set_title(f"{trial_names.get(trial, f'Trial {trial}')} - Not enough events", fontsize=fontsize, pad=24)"""

        # POOLED TIME DOMAIN
        ax = axs_pool[1, j]
        edges  = trial_bin_edges_ms[trial]
        centers = 0.5 * (edges[:-1] + edges[1:])
        labels = trial_bin_labels[trial]
        K = len(edges) - 1

        if trial_delta_ms_raw[trial]:
            C5 = [np.histogram(arr, bins=edges)[0] for arr in trial_delta_ms_raw[trial]]
            C5 = np.vstack(C5) if len(C5) else np.zeros((0, K), int)
            pooled5 = C5.sum(axis=0)
            tot_r = pooled5.sum()

            entry["timedom"] = {
                "summary": pooled5.copy(),
                "bin_edges_ms": edges.copy(),
                "bin_labels": list(labels),
                "N_total": tot_r,
            }

            bars = ax.bar(np.arange(K), pooled5, width=0.8, linewidth=2, edgecolor="black", alpha=1.0)

            for b in bars:
                b.set_facecolor("white")

            if tot_r > 0:
                mode_bin = int(np.argmax(pooled5))
                bars[mode_bin].set_facecolor("grey")
                bars[mode_bin].set_edgecolor("grey")

            err = ax.errorbar(np.arange(K), pooled5, yerr=np.sqrt(np.maximum(pooled5, 0)),
                        fmt='none', lw=1.0, color='black', capsize=5, capthick=1.0, label="Poisson error")

            T_star = trial_T_star.get(trial, np.nan)
            if np.isfinite(T_star):
                centers = 0.5 * (edges[:-1] + edges[1:])
                x_T = np.interp(T_star, centers, np.arange(K))
                ax.axvline(x_T, ls='--', lw=2, color='black', label="T wave offset (end of systole)")

            ax.set_title(f"{trial_names.get(trial, f'Trial {trial}')}", fontsize=fontsize)
            ax.set_xticks(np.arange(K))
            ax.set_xticklabels(labels, rotation=20, fontsize=18)

            if j == 0:
                ax.set_ylabel("Count", fontsize=fontsize)
                ax.tick_params(axis="y", labelsize=18)

            else:
                ax.set_ylabel("")     # remove label
                ax.tick_params(axis="y", labelsize=18)

            ax.grid(axis="y", alpha=0.3)

        """else:
            ax.set_axis_off()
            ax.set_title(f"{trial_names.get(trial, f'Trial {trial}')} - Not enough events", fontsize=fontsize)"""

        summary_stats_pooled.append(entry)

    fig_pool.supxlabel("Time (ms)", fontsize=fontsize)
    fig_pool.suptitle(f"Pooled average of {event_name.replace('_',' ')}s in cardiac cycle (phase & time domain)",fontsize=24, fontweight="bold", y=0.98)

    """handles_p = make_polar_row_handles()
    put_row_legend(fig_pool, axs_pool[0, :], handles_p)

    handles_t = make_timedom_row_handles()
    put_row_legend(fig_pool, axs_pool[1, :], handles_t)"""

    filename = f"{output_dir}/histograms/Cardiac/{sub_folder}/Summary_pooled.pdf"
    plt.savefig(filename, bbox_inches="tight")
    plt.show()
    print(f"Figure {filename} saved!")

    summary_stats_pooled_df = pd.DataFrame(summary_stats_pooled)

    # SUBJECT MEAN PHASE DOMAIN
    fig_mean, axs_mean = plt.subplots(2, nT, subplot_kw={'projection':'polar'}, squeeze=False,
                                      figsize=(5.6*nT, 2*6.3))
    # bottom row cartesian
    for j in range(nT):
        fig_mean.delaxes(axs_mean[1, j])
        axs_mean[1, j] = fig_mean.add_subplot(2, nT, nT + j + 1)

    for j, trial in enumerate(plot_trials):
        entry = {"trial": trial, "polar": None, "timedom": None}
        # subject-mean polar density
        Hpol = np.array(trial_hist_polar[trial], dtype=float) if trial_hist_polar[trial] else np.zeros((0, num_bins_polar))
        ax = axs_mean[0, j]

        if Hpol.size and Hpol.sum() > 0:
            Ni = Hpol.sum(axis=1)
            valid = Ni > 0
            per_subj_dens = (Hpol[valid] / Ni[valid][:, None]) / bin_w if np.any(valid) else np.zeros((0, num_bins_polar))
            mean_dens = per_subj_dens.mean(axis=0) if per_subj_dens.size else np.zeros(num_bins_polar)
            lo, hi = bootstrap_ci_subject_mean(per_subj_dens, B=bootstrap_B, alpha=bootstrap_alpha, seed=seed) \
                     if per_subj_dens.size else (np.zeros(num_bins_polar), np.zeros(num_bins_polar))

            bars = ax.bar(polar_centers, mean_dens, width=bin_w, linewidth=2, edgecolor="black", alpha=1.0, label="Mean subj. density")

            mean_counts = Hpol.sum(axis=0)
            tot_polar_m = int(mean_counts .sum())

            probs = mean_dens * bin_w  # integrates to ~1
            syst_mask_m = polar_centers < np.pi
            prob_systole  = int(probs[syst_mask_m].sum())
            prob_diastole = int(probs[~syst_mask_m].sum())

            mu_m, R_m = circ_from_density(mean_dens, polar_centers, bin_w)

            #(mu_lo, mu_hi), (R_lo, R_hi) = bootstrap_ci_circ_metrics(
            #    per_subj_dens, polar_centers, bin_w, B=bootstrap_B, alpha=bootstrap_alpha, seed=seed
            #)

            mode_bin_m = int(np.argmax(mean_dens))
            md_m = float(polar_centers[mode_bin_m])
            rmx = max(mean_dens.max(), 1e-9)

            N = mean_counts.size
            C = np.mean(np.cos(mean_counts))           # mean cos
            S = np.mean(np.sin(mean_counts))           # mean sin
            R = np.hypot(C, S)                   # || \bar r ||, the mean resultant length
            Rn = R*N
            z, p_value = rayleigh_test(Rn, N)

            #angles = np.repeat(polar_centers, mean_counts.astype(int))  # pooled events, not grand-average
            #p_value = rayleigh_test_from_angles(angles)

            entry["polar"] = {
            "mean_density": mean_dens.copy(),
            "bin_edges": polar_edges.copy(),
            "bin_centers": polar_centers.copy(),
            "bin_width": float(bin_w),
            "circ_mean": float(mu_m),
            "resultant_length": float(R_m),
            "mode_angle": md_m,
            "phase_prob": {"systole": prob_systole, "diastole": prob_diastole},
            "p_value": p_value
        }

            for b in bars:
                b.set_facecolor("white")

            bars[mode_bin_m].set_facecolor("grey")
            bars[mode_bin_m].set_edgecolor("grey")

            """if p_value < 0.01:
                ax.text(1.20, 0.00, f"p={p_value:.3g}", transform=ax.transAxes, ha='right', va='bottom', fontsize=20)

            else:
                ax.text(1.20, 0.00, f"p={p_value:.2g}", transform=ax.transAxes, ha='right', va='bottom', fontsize=20)"""

            ax.annotate("", xy=(mu_m, rmx), xytext=(mu_m, 0.0),
                        arrowprops=dict(arrowstyle="-|>", color="red", lw=4))

            ax.set_title(f"{trial_names.get(trial, f'Trial {trial}')} (events={tot_polar_m})", fontsize=fontsize, pad=30)
            ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
            ax.set_xticklabels(["0", "π/2", "π", "3π/2"], fontsize=20)
            ax.set_yticklabels([]); ax.grid(True)

            print(f"Mean-{event_name}-{trial}: Rayleigh p={p_value:.3g}")
            print(f"circular mean: {mu_m}, mode: {md_m}, RVL: {R_m}")

        """else:
            tot_polar_m = 0
            ax.set_axis_off()
            ax.set_title(f"{trial_names.get(trial, f'Trial {trial}')} - Not enough events", fontsize=fontsize, pad=24)"""

        # SUBJECT MEAN TIME DOMAIN
        ax = axs_mean[1, j]
        edges  = trial_bin_edges_ms[trial]
        labels = trial_bin_labels[trial]
        centers = 0.5 * (edges[:-1] + edges[1:])
        K = len(edges) - 1

        if trial_delta_ms_raw[trial]:
            per_subj_counts = [np.histogram(arr, bins=edges)[0] for arr in trial_delta_ms_raw[trial]]
            per_subj_counts = np.vstack(per_subj_counts)
            Ni = per_subj_counts.sum(axis=1, keepdims=True)
            valid = (Ni[:, 0] > 0)
            P5 = np.zeros_like(per_subj_counts, dtype=float)
            P5[valid] = per_subj_counts[valid] / Ni[valid]
            mean_prop = P5.mean(axis=0)
            lo5, hi5 = bootstrap_ci_subject_mean(P5[valid], B=bootstrap_B, alpha=bootstrap_alpha, seed=seed) \
                       if np.any(valid) else (np.zeros(K), np.zeros(K))

            bars = ax.bar(np.arange(K), mean_prop, width=0.8, linewidth=2, edgecolor="black", alpha=1.0, label="Mean subj. proportion")

            for b in bars:
                b.set_facecolor("white")

            mode_bin = int(np.argmax(mean_prop))

            bars[mode_bin].set_facecolor("grey")
            bars[mode_bin].set_edgecolor("grey")

            for xk, lo_, hi_ in zip(range(K), lo5, hi5):
                mid = (lo_ + hi_) / 2
                err = [[mid - lo_], [hi_ - mid]]   # asymmetric error bars

                err = ax.errorbar(
                    xk, mid,
                    yerr=np.array(err),
                    color='black',
                    lw=1.0,
                    capsize=5,
                    capthick=1.0,
                    label="95% CI (bootstrap)" if xk == 0 else None)

            entry["timedom"] = {
            "summary": mean_prop.copy(),
            "ci_lo": lo5.copy(),
            "ci_hi": hi5.copy(),
            "bin_edges_ms": edges.copy(),
            "bin_labels": list(labels),
            }

            T_star = trial_T_star.get(trial, np.nan)
            if np.isfinite(T_star):
                x_T = np.interp(T_star, centers, np.arange(K))
                ax.axvline(x_T, ls='--', lw=2, color='black', label="T offset")

            ax.set_ylim(0, np.max(hi5)*1.05)
            ax.set_title(f"{trial_names.get(trial, f'Trial {trial}')}", fontsize=fontsize)
            ax.set_xticks(np.arange(K))
            ax.set_xticklabels(labels, rotation=20, fontsize=18)

            if j == 0:
                ax.set_ylabel("Proportion", fontsize=fontsize)
                ax.tick_params(axis="y", labelsize=18)

            else:
                ax.set_ylabel("")     # remove label
                ax.tick_params(labelleft=False)  # hide tick labels

            ax.grid(axis="y", alpha=0.3)

        """else:
            ax.set_axis_off()
            ax.set_title(f"{trial_names.get(trial, f'Trial {trial}')} - Not enough events", fontsize=fontsize, pad=24)"""

        summary_stats_mean.append(entry)

    fig_mean.supxlabel("Time (ms)", fontsize=fontsize)
    fig_mean.suptitle(f"Grand average of {event_name.replace('_',' ')}s in cardiac cycle (phase & time domain)",fontsize=24, fontweight="bold", y=0.98)

    # need own handles
    """handles_p = make_polar_row_handles()
    put_row_legend(fig_mean, axs_mean[0, :], handles_p)

    handles_t = make_timedom_mean_row_handles()
    put_row_legend(fig_mean, axs_mean[1, :], handles_t)"""

    filename = f"{output_dir}/histograms/Cardiac/{sub_folder}/Summary_mean.pdf"
    plt.savefig(filename, bbox_inches="tight")
    plt.show()
    print(f"Figure {filename} saved!")

    summary_stats_mean_df = pd.DataFrame(summary_stats_mean)

    return circ_stats_df, timedom_stats_df, counts_timedom_df, histogram_info_polar_df, summary_stats_pooled_df, summary_stats_mean_df

#%%

def polar_histograms_respiration(
    df_clean,
    resp_index_df,
    event_index_df,
    event_name,
    sub_folder,
    num_bins=12,
    bootstrap_B=2000,
    bootstrap_alpha=0.05,
    seed=0,
):

    all_trials  = sorted(df_clean["Trial"].unique().tolist())
    subjects    = sorted(df_clean["Subject"].unique().tolist())
    trial_names = {1:"Control", 2:"External", 3:"Internal", 4:"Self-paced"}

    # fixed polar bins
    bin_edges   = np.linspace(0, 2*np.pi, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_w       = 2*np.pi / num_bins

    # collectors (for summaries)
    trial_histograms = {trial: [] for trial in all_trials}  # per-subject counts (S × bins)
    trial_angles     = {trial: [] for trial in all_trials}  # raw angles per subject (variable-length lists)
    circ_stats_rows  = []                                   # optional per-subject stats
    events = []

    fontsize = 24

    # Figures per subject
    for subject in subjects:

        if subject == "Sub05":
            continue

        if event_name == "saccade_onset" and subject == "Sub07":
            continue

        if event_name == "blink_onset" and subject == "Sub01": #or subject == "Sub04" or subject == "Sub02":
            continue

        subj_trials = sorted(df_clean.loc[df_clean["Subject"] == subject, "Trial"].unique().tolist())
        if len(subj_trials) == 0:
            continue

        # one row of polar panels (one per trial)
        plt.close('all')
        fig, axs = plt.subplots(1, len(subj_trials), subplot_kw={'projection':'polar'},
                                figsize=(5.5 * len(subj_trials), 6.0))
        if len(subj_trials) == 1:
            axs = np.array([axs], dtype=object)

        for ax, trial in zip(axs, subj_trials):
            # base vectors
            data = df_clean[(df_clean["Subject"] == subject) & (df_clean["Trial"] == trial)].sort_values("Time")
            t    = data["Time"].to_numpy(dtype=float)
            L    = len(t)

            # respiration landmarks (indices)
            rrows = resp_index_df[(resp_index_df["Subject"] == subject) & (resp_index_df["Trial"] == trial)]
            insp_peak = rrows[rrows["event"].str.contains("insp", case=False, na=False)]["index"].to_numpy(dtype=int)
            exp_trough = rrows[rrows["event"].str.contains("exp", case=False, na=False)]["index"].to_numpy(dtype=int)

            insp_peak  = insp_peak[(insp_peak >= 0) & (insp_peak < L)]
            exp_trough = exp_trough[(exp_trough >= 0) & (exp_trough < L)]

            exhale_times = t[insp_peak]
            inhale_times = t[exp_trough]
            scale_exh, scale_inh = scale_exhale_inhale(exhale_times, inhale_times)

            # map each sample to respiratory phase using NeuroKit (robust to missing samples)
            if insp_peak.size == 0 or exp_trough.size == 0:
                # nothing to map → empty panel
                trial_histograms[trial].append(np.zeros(num_bins, int))
                ax.set_axis_off()
                ax.set_title(f"{trial_names.get(trial, f'Trial {trial}')} - Not enough events", fontsize = fontsize, pad=24)
                continue

            rsp_out = nk.rsp_phase(peaks=insp_peak, troughs=exp_trough, desired_length=L)
            phase   = rsp_out["RSP_Phase"].to_numpy(dtype=float)            # 0=exhalation, 1=inhalation
            comp    = rsp_out["RSP_Phase_Completion"].to_numpy(dtype=float) # 0..1 within the current half-cycle

            # angle mapping: exhalation 0..π, inhalation π..2π
            theta = np.where(phase == 0, comp * np.pi, np.pi + comp * np.pi)
            theta = np.mod(theta, 2*np.pi)

            # events to bin
            erows = event_index_df[(event_index_df["Subject"] == subject) &
                                   (event_index_df["Trial"] == trial) &
                                   (event_index_df["event"] == event_name)]
            ev_idx = erows["index"].to_numpy(dtype=int)
            ev_idx = ev_idx[(ev_idx >= 0) & (ev_idx < L)]

            if ev_idx.size == 0:
                trial_histograms[trial].append(np.zeros(num_bins, int))
                ax.set_axis_off()
                ax.set_title(f"{trial_names.get(trial, f'Trial {trial}')} - Not enough events", fontsize=fontsize, pad=24)
                continue

            angles = theta[ev_idx]
            angles = angles[np.isfinite(angles)]

            if angles.size == 0:
                hist = np.zeros(len(bin_edges) - 1, dtype=float)

            else:
                in_exhale = angles < np.pi
                in_inhale = ~in_exhale

                scale_exh = 1.0 if not np.isfinite(scale_exh) else float(scale_exh)
                scale_inh = 1.0 if not np.isfinite(scale_inh) else float(scale_inh)
                weights = np.where(in_exhale, scale_exh, scale_inh).astype(float)
                hist, _ = np.histogram(angles, bins=bin_edges, weights=weights)

                events.append({
                "Subject": subject, "Trial": trial,
                "Event_idx": ev_idx,
                "In_exhale": in_exhale,
                "In_inhale": in_inhale
            })

            trial_histograms[trial].append(hist)
            trial_angles[trial].append(angles)

            # circular stats per subject (optional)
            cmean = circmean(angles, high=2*np.pi, low=0)
            cstd  = circstd(angles, high=2*np.pi, low=0)
            cvar  = circvar(angles, high=2*np.pi, low=0)

            N = angles.size
            C = np.mean(np.cos(angles))           # mean cos
            S = np.mean(np.sin(angles))           # mean sin
            R = np.hypot(C, S)                   # || \bar r ||, the mean resultant length
            Rn = R*N
            z, p_value = rayleigh_test(Rn, N)

            mode_bin = np.argmax(hist)
            md = bin_centers[mode_bin]

            circ_stats_rows.append({
                "Subject": subject, "Trial": trial,
                "Circular_mean_rad": cmean,
                "Circular_mean_deg": float(np.degrees(cmean)),
                "Circular_std": float(cstd),
                "Circular_var": float(cvar),
                "Resultant_vector_length": float(R),
                "Mode_rad": md,
                "Mode_deg": float(np.degrees(md)),
                "N_events": int(len(angles)),
                "P_value": p_value
            })

            # ---- plot per-subject panel (style aligned to cardiac figs) ----
            bars = ax.bar(bin_centers, hist, width=bin_w, linewidth=2, edgecolor="black", alpha=1.0)
            for b in bars:
                b.set_facecolor("white")

            bars[mode_bin].set_facecolor("grey")
            bars[mode_bin].set_edgecolor("grey")

            # highlight max bin
            m = np.zeros_like(hist); m[np.argmax(hist)] = hist.max()
            #ax.bar(bin_centers, m, width=bin_w, color="red", edgecolor="white", alpha=0.8)

            # circular mean arrow
            r_max = max(hist.max(), 1)

            """if p_value < 0.01:
                ax.text(1.30, 0.00, f"Rayleigh p={p_value:.3g}", transform=ax.transAxes, ha='right', va='bottom', fontsize=16)
            else:
                ax.text(1.30, 0.00, f"Rayleigh p={p_value:.2g}", transform=ax.transAxes, ha='right', va='bottom', fontsize=16)"""
            ax.annotate("", xy=(cmean, r_max), xytext=(cmean, 0.0),
                        arrowprops=dict(arrowstyle="-|>", color="red", lw=4))

            ax.set_title(f"{trial_names.get(trial, f'Trial {trial}')} (events={N})", fontsize=fontsize, pad=28)
            ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
            ax.set_xticklabels(["0", "π/2", "π", "3π/2"], fontsize=14)
            ax.set_yticklabels([]); ax.grid(True)

            print(f"{subject}-{event_name}-{trial}: Rayleigh p={p_value:.3g}")

        # figure title by event
        title_evt = "Saccade onsets" if event_name == "saccade_onset" else \
                    "Blink onsets"   if event_name == "blink_onset"   else event_name.replace("_"," ").title()
        fig.suptitle(f"Subject {subject} — {title_evt} in respiration cycle (scaled polar)", fontsize=20, fontweight="bold", y=1.02)

        handles_p = make_polar_row_handles()
        put_row_legend(fig, axs, handles_p)

        filename = f"{output_dir}/histograms/Respiration/{sub_folder}/{subject}_{event_name}.png"
        plt.savefig(filename)
        #plt.show()
        plt.close()
        print(f"Figure {filename} saved!")

    circ_stats_df = pd.DataFrame(circ_stats_rows)
    events_df = pd.DataFrame(events)

    # Define a custom file name per subject
    """filename = f"{out_preprocess}/{event_name}_respiratory.pkl"

    # Save to file
    with open(filename, "wb") as f:
        pickle.dump({
            f"{event_name}_respiratory_polar": circ_stats_df,
        }, f)

    print(f"Saved respiratory data → {filename}")"""

    # POOLED
    plot_trials = []
    summary_stats_pooled = []
    summary_stats_mean = []

    for trial in all_trials:
        H_list  = trial_histograms[trial]
        has_polar_data = (H_list) and (sum(h.sum() for h in H_list) >0)
        if has_polar_data:
            plot_trials.append(trial)
    nT = len(plot_trials)

    fig_pooled, axes_pooled = plt.subplots(1,nT, subplot_kw={'projection':'polar'}, figsize=(5.6*nT, 6.3))

    for j, trial in enumerate(plot_trials):
        ax_top  = axes_pooled[j]
        H_list  = trial_histograms[trial]
        A_list  = trial_angles[trial]

        """if (not H_list) or (sum(h.sum() for h in H_list) == 0):
            ax_top.set_axis_off()
            ax_top.set_title(f"{trial_names.get(trial, f'Trial {trial}')} - Not enough events", fontsize=fontsize, pad=24)
            continue"""

        # --- (A) pooled density from concatenated angles ---
        all_ang = np.concatenate(A_list) if len(A_list) else np.array([], float)
        pooled_counts, _ = np.histogram(all_ang, bins=bin_edges)
        total = pooled_counts.sum()
        pooled_density = (pooled_counts / max(total, 1)) / bin_w
        mu_pooled, R_pooled = circ_from_density(pooled_density, bin_centers, bin_w)

        N = pooled_counts.size
        C = np.mean(np.cos(pooled_counts))           # mean cos
        S = np.mean(np.sin(pooled_counts))           # mean sin
        R = np.hypot(C, S)                   # || \bar r ||, the mean resultant length
        Rn = R*N
        z, p_value = rayleigh_test(Rn, N)
        mode_bin = np.argmax(pooled_counts)
        #md_pooled = bin_centers[bin]

        ax = ax_top
        bars = ax.bar(bin_centers, pooled_density, width=bin_w, linewidth=2, edgecolor="black", alpha=1.0)

        for b in bars:
             b.set_facecolor("white")

        bars[mode_bin].set_facecolor("grey")
        bars[mode_bin].set_edgecolor("grey")

        # highlight max
        m = np.zeros_like(pooled_density); m[np.argmax(pooled_density)] = pooled_density.max()
        #ax.bar(bin_centers, m, width=bin_w, color="red", edgecolor="white", alpha=0.8)
        rmx = max(pooled_density.max(), 1e-9)
        """if p_value < 0.01:
            ax.text(1.30, 0.00, f"Rayleigh p={p_value:.3g}", transform=ax.transAxes, ha='right', va='bottom', fontsize=16)
        else:
            ax.text(1.30, 0.00, f"Rayleigh p={p_value:.2g}", transform=ax.transAxes, ha='right', va='bottom', fontsize=16)"""
        ax.annotate("", xy=(mu_pooled, rmx), xytext=(mu_pooled, 0.0),
                    arrowprops=dict(arrowstyle="-|>", color="red", lw=4))
        ax.set_title(f"{trial_names.get(trial, f'Trial {trial}')} (events={total})",fontsize=fontsize, pad=30)
        ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
        ax.set_xticklabels(["0", "π/2", "π", "3π/2"], fontsize=20)
        ax.set_yticklabels([]); ax.grid(True)

        print(f"Pooled-{trial}-{event_name}: Rayleigh p={p_value:.3g}")

    title_evt = "saccade onsets" if event_name == "saccade_onset" else \
                "blink onsets"   if event_name == "blink_onset"   else event_name.replace("_"," ").title()

    fig_pooled.suptitle(f"Pooled average of {title_evt} in respiration cycle (phase domain)", fontsize=24, fontweight="bold", y=0.98)

    """handles_p = make_polar_row_handles()
    put_row_legend(fig_pooled, axes_pooled[:], handles_p)"""

    filename = f"{output_dir}/histograms/Respiration/{sub_folder}/Summary_pooled.pdf"
    plt.savefig(filename, bbox_inches="tight")
    plt.show()

    # MEAN
    fig_mean, axes_mean = plt.subplots(1,nT, subplot_kw={'projection':'polar'}, figsize=(5.6*nT, 6.3))

    """if nT == 1:
        axes_mean = np.array([axes_mean])"""

    for j, trial in enumerate(plot_trials):
        ax_bot  = axes_mean[j]
        H_list  = trial_histograms[trial]
        A_list  = trial_angles[trial]

        """if (not H_list) or (sum(h.sum() for h in H_list) == 0):
            ax_bot.set_axis_off()
            ax_bot.set_title(f"{trial_names.get(trial, f'Trial {trial}')} - Not enough events", fontsize=fontsize, pad=24)
            continue"""

        # SUBJECT MEAN
        # build per-subject densities from angles (so subjects with diff N are equal-weighted)
        per_subj_density = []
        for angles in A_list:
            if len(angles):
                counts, _ = np.histogram(angles, bins=bin_edges)
                dens = (counts / counts.sum()) / bin_w
                per_subj_density.append(dens)
        per_subj_density = np.vstack(per_subj_density) if per_subj_density else np.zeros((0, num_bins), float)

        all_ang = np.concatenate(A_list) if len(A_list) else np.array([], float)
        all_counts, _ = np.histogram(all_ang, bins=bin_edges)
        total = all_counts.sum()

        ax = ax_bot
        if per_subj_density.size:
            mean_dens = per_subj_density.mean(axis=0)
            lo, hi    = bootstrap_ci_subject_mean(per_subj_density, B=bootstrap_B, alpha=bootstrap_alpha, seed=seed)
            mu_mean, R_mean = circ_from_density(mean_dens, bin_centers, bin_w)

            C = np.mean(np.cos(counts))           # mean cos
            S = np.mean(np.sin(counts))           # mean sin
            R = np.hypot(C, S)                   # || \bar r ||, the mean resultant length
            Rn = R*N
            z,p_value = rayleigh_test(Rn, N)
            mode_bin = np.argmax(mean_dens)

            bars = ax.bar(bin_centers, mean_dens, width=bin_w, linewidth=2, edgecolor="black", alpha=1.0, label="Mean subj. density")

            for b in bars:
                b.set_facecolor("white")

            bars[mode_bin].set_facecolor("grey")
            bars[mode_bin].set_edgecolor("grey")

            rmx = max(mean_dens.max(), 1e-9)

            """if p_value < 0.01:
                ax.text(1.30, 0.00, f"Rayleigh p={p_value:.3g}", transform=ax.transAxes, ha='right', va='bottom', fontsize=16)
            else:
                ax.text(1.30, 0.00, f"Rayleigh p={p_value:.2g}", transform=ax.transAxes, ha='right', va='bottom', fontsize=16)"""
            ax.annotate("", xy=(mu_mean, rmx), xytext=(mu_mean, 0.0),
                        arrowprops=dict(arrowstyle="-|>", color="red", lw=4))
        else:
            ax.bar(bin_centers, np.zeros(num_bins), width=bin_w, edgecolor="black", alpha=1.0)

        ax.set_title(f"{trial_names.get(trial, f'Trial {trial}')} (events={total})", fontsize=fontsize, pad=30)
        ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
        ax.set_xticklabels(["0", "π/2", "π", "3π/2"], fontsize=20)
        ax.set_yticklabels([]); ax.grid(True)

        print(f"Mean-{trial}{event_name}: Rayleigh p={p_value:.3g}")
        print(f"circular mean: {mu_mean}, mode: {mode_bin}, RVL: {R_mean}")

    title_evt = "saccade onsets" if event_name == "saccade_onset" else \
                "blink onsets"   if event_name == "blink_onset"   else event_name.replace("_"," ").title()
    fig_mean.suptitle(f"Grand average of {title_evt} in respiration cycle (phase domain)", fontsize=24, fontweight="bold", y=0.98)

    """handles_p = make_polar_row_handles()
    put_row_legend(fig_mean, axes_mean[:], handles_p)"""

    filename = f"{output_dir}/histograms/Respiration/{sub_folder}/Summary_mean.pdf"
    plt.savefig(filename, bbox_inches="tight")
    plt.show()
    print(f"Figure {filename} saved!")


    return circ_stats_df, events_df
