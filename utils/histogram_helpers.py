import numpy as np
from matplotlib.lines import Line2D
import matplotlib.transforms as mtransforms

"""
Supporting functions for time & phase domain histograms, e.g. defining trial-level and subject-level windows, 
defining phases of events, circular statistics, performing bootstrapping and modifying plots and legends 
by Veera Tarkiainen
"""

# These handles not are not used
def make_polar_row_handles():
    # Only circular mean (arrow)
    return [
        Line2D([0],[0], color='red', lw=2, label='Circular mean')
    ]

def make_polar_mean_row_handles():
    # Circular mean (arrow) + 95% CI (T-like marker)
    return [
        Line2D([0],[0], color='red', lw=2, label='Circular mean'),
        Line2D([0],[0], color='black', lw=1.5, label='95% CI (bootstrap)'),
    ]

def make_timedom_row_handles():
    # T-wave offset as dashed line + Poisson error as T-like marker
    return [
        Line2D([0], [0], color="black", lw=2, ls="--",
               label="T wave offset (end of systole)"),
        Line2D([0],[0], color='black', lw=1.5, label='Poisson error'),
    ]


def make_timedom_mean_row_handles():
    # Same but for CI
    return [
        Line2D([0], [0], color="black", lw=2, ls="--",
               label="T wave offset (end of systole)"),
        Line2D([0],[0], color='black', lw=1.5, label='95% CI (bootstrap)'),
    ]

def put_row_legend(fig, row_axes, handles, *, fontsize=14, pad=0.01, reserve_right=0.82):

    # Make sure there's space for the legend to the right of the grid.
    if fig.subplotpars.right > reserve_right:
        fig.subplots_adjust(right=reserve_right)

    # Union bbox of all axes in the row (in figure coordinates)
    bboxes = [ax.get_position() for ax in row_axes]
    row_bb = mtransforms.Bbox.union(bboxes)

    # Place a single, vertical legend at the row's top-right, just outside the grid
    leg = fig.legend(
        handles=handles,
        loc='upper left',
        bbox_to_anchor=(row_bb.x1 + pad, row_bb.y1),
        bbox_transform=fig.transFigure,
        frameon=False,
        ncol=1,              # vertical stack
        fontsize=fontsize,
        borderaxespad=0.0
    )
    return leg

def new_windows_relative_to_cycle(r_times_s, t_offsets_s, K=6, alpha_clip=(0.25, 0.45)):

    r = np.unique(np.asarray(r_times_s, float))
    t = np.unique(np.asarray(t_offsets_s, float))

    if r.size < 2:
        return [], np.nan, np.nan, np.nan, np.array([]), np.array([]), []

    # --- pick one T per beat R_i..R_{i+1}
    n_beats = len(r) - 1
    T_per = np.full(n_beats, np.nan, float)
    k = 0
    for i in range(n_beats):
        while k < len(t) and t[k] <= r[i]:
            k += 1
        if k < len(t) and t[k] < r[i+1]:
            T_per[i] = t[k]
            k += 1

    # --- median alpha (T occurs alpha*RR after R)
    RR = r[1:] - r[:-1]
    withT = ~np.isnan(T_per)
    if np.any(withT):
        alpha = (T_per[withT] - r[:-1][withT]) / RR[withT]
        alpha_med = float(np.clip(np.nanmedian(alpha), *alpha_clip))
    else:
        alpha_med = 0.35  # conservative fallback

    RR_med_s = float(np.nanmedian(RR))
    T_med_s  = float(alpha_med * RR_med_s)

    RR_med_ms = RR_med_s * 1000.0
    T_med_ms  = T_med_s  * 1000.0

    # Phase durations per beat (same units as indices)
    syst_len_beats = np.maximum(T_per - r[:-1], 0.0)   # shape (n_beats,)
    dias_len_beats = np.maximum(r[1:] - T_per, 0.0)

    # Totals across all beats
    T_syst_total = float(np.nansum(syst_len_beats))
    T_dias_total = float(np.nansum(dias_len_beats))
    T_total      = T_syst_total + T_dias_total
    T_equal      = 0.5 * T_total  # target equal duration per phase

    # Equal-length–adjusted counts (expected counts if both phases had T_equal duration)
    scale_syst = (T_equal / T_syst_total) if T_syst_total > 0 else np.nan
    scale_dias = (T_equal / T_dias_total) if T_dias_total > 0 else np.nan

    # --- K equal-width bins across the RR template [0, RR_med_ms]
    edges = np.linspace(0.0, RR_med_ms, K + 1)
    windows_ms = [(edges[i], edges[i+1]) for i in range(K)]

    # --- phase overlap per bin
    sys_lo, sys_hi = 0.0, T_med_ms
    dia_lo, dia_hi = T_med_ms, RR_med_ms

    s_frac = np.zeros(K, float)
    d_frac = np.zeros(K, float)
    phase_lbls = []

    for i, (lo, hi) in enumerate(windows_ms):
        wlen = hi - lo
        if wlen <= 0:
            phase_lbls.append("mixed")
            continue

        sys_overlap = max(0.0, min(hi, sys_hi) - max(lo, sys_lo))
        dia_overlap = max(0.0, min(hi, dia_hi) - max(lo, dia_lo))

        s_frac[i] = sys_overlap / wlen
        d_frac[i] = dia_overlap / wlen

        if s_frac[i] > 0 and d_frac[i] == 0:
            phase_lbls.append("systole")
        elif d_frac[i] > 0 and s_frac[i] == 0:
            phase_lbls.append("diastole")
        else:
            phase_lbls.append("mixed")

    return windows_ms, RR_med_ms, T_med_ms, alpha_med, s_frac, d_frac, phase_lbls, scale_syst, scale_dias

def new_windows_summary(list_of_subjects_r_t, K=6, agg='median', alpha_clip=(0.25, 0.45)):

    RR_ms_list, alpha_list = [], []

    for d in list_of_subjects_r_t:
        # Get subject-level RR_med and alpha_med (we ignore subject T_med; we’ll derive group T* coherently)
        _, rr_ms, _, a, *_ = new_windows_relative_to_cycle(
            d['r'], d['t'], K=K, alpha_clip=alpha_clip
        )
        if np.isfinite(rr_ms) and np.isfinite(a):
            RR_ms_list.append(rr_ms)
            alpha_list.append(a)

    if not RR_ms_list:
        return [], np.nan, np.nan, np.nan, np.array([]), np.array([]), []

    RR_ms_arr  = np.array(RR_ms_list,  dtype=float)
    alpha_arr  = np.array(alpha_list,  dtype=float)

    if agg == 'mean':
        RR_star = float(np.nanmean(RR_ms_arr))
        alpha_star = float(np.clip(np.nanmean(alpha_arr), *alpha_clip))
    else:  # 'median' (recommended)
        RR_star = float(np.nanmedian(RR_ms_arr))
        alpha_star = float(np.clip(np.nanmedian(alpha_arr), *alpha_clip))

    # Derive T* coherently from alpha* and RR*
    T_star = alpha_star * RR_star

    # K equal-width bins on [0, RR_star]
    edges = np.linspace(0.0, RR_star, K + 1)
    windows_ms = [(edges[i], edges[i+1]) for i in range(K)]

    # phase overlap per group bin
    sys_lo, sys_hi = 0.0, T_star
    dia_lo, dia_hi = T_star, RR_star

    s_frac = np.zeros(K, float)
    d_frac = np.zeros(K, float)
    phase_lbls = []
    for i, (lo, hi) in enumerate(windows_ms):
        wlen = hi - lo
        if wlen <= 0:
            phase_lbls.append("mixed")
            continue
        sys_overlap = max(0.0, min(hi, sys_hi) - max(lo, sys_lo))
        dia_overlap = max(0.0, min(hi, dia_hi) - max(lo, dia_lo))
        s_frac[i] = sys_overlap / wlen
        d_frac[i] = dia_overlap / wlen
        if s_frac[i] > 0 and d_frac[i] == 0:
            phase_lbls.append("systole")
        elif d_frac[i] > 0 and s_frac[i] == 0:
            phase_lbls.append("diastole")
        else:
            phase_lbls.append("mixed")

    return windows_ms, RR_star, T_star, alpha_star, s_frac, d_frac, phase_lbls

def sanitize_edges(edges):
    e = np.asarray(edges, dtype=float)
    e = e[np.isfinite(e)]
    if e.size < 2:
        raise ValueError("sanitize_edges: need at least two finite edges.")
    e.sort()
    for k in range(1, e.size):
        if e[k] <= e[k-1]:
            e[k] = np.nextafter(e[k-1], np.inf)  # minimal bump to enforce strict increase
    return e

# Map events to [0, 2π): 0..π over R→T, π..2π over T→next R (with T fallback)
def phase_theta_rt(event_times_s, r_times_s, t_offset_times_s):
    if r_times_s.size < 2 or event_times_s.size == 0:
        return np.array([], dtype=float)
    e = np.asarray(event_times_s, float)
    r = np.unique(np.asarray(r_times_s, float))
    t = np.unique(np.asarray(t_offset_times_s, float))
    n_beats = len(r) - 1

    # T per beat (first T strictly between R_i and R_{i+1}); fallback via alpha median
    T_per = np.full(n_beats, np.nan, float)
    k = 0
    for i in range(n_beats):
        while k < len(t) and t[k] <= r[i]:
             k += 1
        if k < len(t) and t[k] < r[i+1]:
            T_per[i] = t[k]
            k += 1

    RR = r[1:] - r[:-1]
    withT = ~np.isnan(T_per)
    if np.any(withT):
        alpha = (T_per[withT] - r[:-1][withT]) / RR[withT]
        alpha_med = float(np.clip(np.nanmedian(alpha), 0.25, 0.45))
    else:
        alpha_med = 0.35
    miss = np.isnan(T_per)
    if np.any(miss):
        T_per[miss] = r[:-1][miss] + alpha_med * RR[miss]

    i_prev = np.searchsorted(r, e, side="right") - 1
    valid  = (i_prev >= 0) & (i_prev < n_beats)
    if not np.any(valid):
        return np.array([], dtype=float)

    ee   = e[valid]
    Ri   = r[i_prev[valid]]
    Rip1 = r[i_prev[valid] + 1]
    Ti   = T_per[i_prev[valid]]

    eps = 1e-9
    syst_len = np.maximum(Ti - Ri,   eps)
    dias_len = np.maximum(Rip1 - Ti, eps)

    in_systole = (ee >= Ri) & (ee < Ti)
    theta = np.empty_like(ee)

    theta[in_systole]  = np.pi * (ee[in_systole] - Ri[in_systole]) / syst_len[in_systole]
    theta[~in_systole] = np.pi + np.pi * (ee[~in_systole] - Ti[~in_systole]) / dias_len[~in_systole]

    valid = np.isfinite(theta)
    return np.mod(theta[valid], 2*np.pi)

# Δt relative to previous R
def delta(event_times_s, r_times_s):
    if r_times_s.size == 0 or event_times_s.size == 0:
        return np.array([], dtype=float)
    r = np.sort(r_times_s)
    e = np.asarray(event_times_s)
    i_prev = np.searchsorted(r, e, side='right') - 1
    valid  = (i_prev >= 0) & (i_prev < len(r)-1)
    if not np.any(valid):
        return np.array([], dtype=float)
    e, i_prev = e[valid], i_prev[valid]
    prev, nxt = r[i_prev], r[i_prev+1]
    dt_ms = (e - prev) * 1000.0
        #if fold_pre_next_100ms:
        #    rr_ms = (nxt - prev) * 1000.0
        #    mask  = (rr_ms - dt_ms) < 100
        #    dt_ms[mask] = -(rr_ms[mask] - dt_ms[mask])  # map to [-100,0)
    return dt_ms

def scale_exhale_inhale(exhale_starts, inhale_starts, end_time=None):

    exh = np.unique(np.asarray(exhale_starts, float))
    inh = np.unique(np.asarray(inhale_starts, float))
    exh = exh[np.isfinite(exh)]
    inh = inh[np.isfinite(inh)]
    exh.sort(); inh.sort()

    # Exhale durations: from each exhale start to the next inhale start
    j = np.searchsorted(inh, exh, side='left')        # index of first inh >= each exh
    mask_exh = j < inh.size
    exh_durs = np.maximum(inh[j[mask_exh]] - exh[mask_exh], 0.0)

    # If a trailing exhale has no following inhale and an end time is known
    if end_time is not None and np.any(~mask_exh):
        tail = np.maximum(end_time - exh[~mask_exh], 0.0)
        exh_durs = np.concatenate([exh_durs, tail])

    # Inhale durations: from each inhale start to the next exhale start
    k = np.searchsorted(exh, inh, side='left')        # index of first exh >= each inh
    mask_inh = k < exh.size
    inh_durs = np.maximum(exh[k[mask_inh]] - inh[mask_inh], 0.0)

    if end_time is not None and np.any(~mask_inh):
        tail = np.maximum(end_time - inh[~mask_inh], 0.0)
        inh_durs = np.concatenate([inh_durs, tail])

    # Totals
    T_exh_total = float(np.nansum(exh_durs))
    T_inh_total = float(np.nansum(inh_durs))
    T_total     = T_exh_total + T_inh_total

    # If we have no durations (e.g., missing labels), avoid NaNs
    if T_exh_total <= 0 or T_inh_total <= 0:
        return 1.0, 1.0, T_exh_total, T_inh_total

    # Make the phases “equal length” in expectation
    T_equal   = 0.5 * T_total
    scale_exh = T_equal / T_exh_total
    scale_inh = T_equal / T_inh_total

    return scale_exh, scale_inh

def bootstrap_ci_subject_mean(rows, B=2000, alpha=0.05, seed=0):
    rows = np.asarray(rows, float)
    if rows.size == 0:
        return np.zeros(rows.shape[1] if rows.ndim == 2 else 0), np.zeros(rows.shape[1] if rows.ndim == 2 else 0)
    S, Kk = rows.shape
    rng = np.random.default_rng(seed)
    draws = np.empty((B, Kk), float)
    for b in range(B):
        idx = rng.integers(0, S, size=S)
        draws[b] = rows[idx].mean(axis=0)
    lo = np.quantile(draws, alpha/2, axis=0)
    hi = np.quantile(draws, 1 - alpha/2, axis=0)
    return lo, hi

def circ_from_density(d, centers, bin_w):
    C = np.sum(d * np.cos(centers)) * bin_w
    S = np.sum(d * np.sin(centers)) * bin_w
    mu = (np.arctan2(S, C)) % (2*np.pi)
    Rbar = np.hypot(C, S)
    return mu, Rbar

def rayleigh_test(Rn, N):
    z = (Rn**2) / N
    p = float(np.exp(np.sqrt(1 + 4*N + 4*(N**2 - Rn**2)) - (1 + 2*N)))
    return z,p

