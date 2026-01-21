import numpy as np
import pandas as pd
from scipy.stats import f, friedmanchisquare
import pingouin as pg
import itertools

"""
Functions for statistical testing in the time domain (by two way repeated measures
ANOVA) and in the phase domain (circular ANOVA, Friedman test) for both cardiac and respiratory cycles by Veera Tarkiainen
"""

#%%

# watson-williams (circular ANOVA) to test differences between mean phases across conditions
def kappa_from_rbar(Rbar):
    if Rbar < 0.53:
        return 2*Rbar + Rbar**3 + 5*Rbar**5/6
    elif Rbar < 0.85:
        return -0.4 + 1.39*Rbar + 0.43/(1 - Rbar)
    else:
        return 1/(Rbar**3 - 4*Rbar**2 + 3*Rbar)

def watson_williams_test(angles, groups):
    angles = np.asarray(angles, float)
    groups = np.asarray(groups)

    m = ~np.isnan(angles)
    angles, groups = angles[m], groups[m]

    angles = np.mod(angles, 2 * np.pi)

    labs = np.unique(groups)
    k = len(labs)
    n_js, R_js, C_js, S_js = [], [], [], []
    for L in labs:
        a = angles[groups == L]
        n = len(a)
        C = np.sum(np.cos(a)); S = np.sum(np.sin(a))
        R = np.hypot(C, S)
        n_js.append(n); R_js.append(R); C_js.append(C); S_js.append(S)
    n_js = np.array(n_js); R_js = np.array(R_js)
    N = n_js.sum()
    C = np.sum(C_js); S = np.sum(S_js); R = np.hypot(C, S)
    Rbar = R / N
    kappa = max(kappa_from_rbar(Rbar), 1e-8)
    beta = 1 + 3/(8*kappa)

    A = np.sum(R_js) - R
    B = N - np.sum(R_js)
    df1, df2 = k - 1, N - k
    Fstat = beta * ((df2 * A) / (df1 * B))
    p = f.sf(Fstat, df1, df2)
    return Fstat, p

#within subject permutation test for W-W
def ww_perm_within_subject(df, n_perm=5000, seed=0): #permute Trial labels within each subject

    rng = np.random.default_rng(seed)
    ang = np.mod(df["Circular_mean_rad"].to_numpy(), 2*np.pi)
    grp = df["Trial"].to_numpy()
    sub = df["Subject"].to_numpy()

    F_obs, _ = watson_williams_test(ang, grp)

    unique_sub = np.unique(sub)
    idx_per_sub = {s: np.where(sub == s)[0] for s in unique_sub}

    F_perm = np.empty(n_perm)
    for i in range(n_perm):
        perm_grp = grp.copy()
        for s, idx in idx_per_sub.items():
            perm_grp[idx] = rng.permutation(perm_grp[idx])
        F_perm[i], _ = watson_williams_test(ang, perm_grp)

    p_perm = (np.sum(F_perm >= F_obs) + 1) / (n_perm + 1)
    return F_obs, p_perm

# Pairwise comparisons
def ww_posthoc_pairwise(df, n_perm=5000):
    trials = df["Trial"].unique()
    pairs = list(itertools.combinations(trials, 2))
    results = []

    for t1, t2 in pairs:
        # Filter data for just this pair
        pair_df = df[df["Trial"].isin([t1, t2])].copy()

        # Run the same permutation test on the pair
        f_obs, p_perm = ww_perm_within_subject(pair_df, n_perm=n_perm)
        results.append({
            "Comparison": f"{t1} vs {t2}",
            "F_obs": f_obs,
            "p_uncorrected": p_perm
        })

    # Apply Bonferroni Correction
    posthoc_df = pd.DataFrame(results)
    posthoc_df["p_corrected"] = (posthoc_df["p_uncorrected"] * len(pairs)).clip(upper=1.0)
    return posthoc_df

def perform_watson_williams(df, label=""):
    df = df.dropna(subset=["Circular_mean_rad"]).copy()
    df["Circular_mean_rad"] = np.mod(df["Circular_mean_rad"], 2*np.pi)

    F_obs, p_perm = ww_perm_within_subject(df, n_perm=5000, seed=1)
    print(f"{label} Permutation WW (within-subject): F={F_obs:.3f}, p={p_perm:.4g}")

    if p_perm < 0.05:
        print("Significant result found. Running pairwise post-hoc...")
        posthoc_results = ww_posthoc_pairwise(df)
        print(posthoc_results)
        return F_obs, p_perm, posthoc_results

    return F_obs, p_perm, None

# Wilcoxon signed rank test with Bonferonni correction
def perform_friedman_posthoc(df, value_col="Resultant_vector_length"):
    # Pingouin's pairwise_tests handles the Wilcoxon signed-rank test
    # which is the standard post-hoc follow-up for Friedman.
    posthoc = pg.pairwise_tests(
        data=df,
        dv=value_col,
        within="Trial",
        subject="Subject",
        parametric=False,  # This forces Wilcoxon signed-rank
        padjust="bonf"
    )
    return posthoc

#Friedman test to test differences in vector length (locking strength)
def perform_friedman_test(df,
                          value_col="Resultant_vector_length",
                          subject_col="Subject",
                          trial_col="Trial",
                          label=""):

    d = df.dropna(subset=[value_col, subject_col, trial_col]).copy()
    pivot = d.pivot(index=subject_col, columns=trial_col, values=value_col)
    pivot_cc = pivot.dropna(axis=0, how="any")  # keep only subjects with all trials

    print(pivot.shape)
    print(pivot.var(axis=1))  # per-subject variance across conditions
    print("NaNs per condition:\n", pivot.isna().sum())
    print("Subjects with any NaN:", pivot.isna().any(axis=1).sum())
    print("Complete-case subjects:", pivot.dropna().shape[0])

    cond_arrays = [pivot_cc[c].to_numpy() for c in pivot_cc.columns]
    stat, p = friedmanchisquare(*cond_arrays)

    print(f"[{label}] Friedman test on {value_col}: χ² = {stat:.3f}, p = {p:.4g}")
    print(f"Conditions: {list(pivot_cc.columns)}")

    if p < 0.05:
        print("Significant result found. Running pairwise post-hoc...")
        posthoc_results = perform_friedman_posthoc(d)
        print(posthoc_results)
        return stat, p, posthoc_results

    return stat, p, None

# Two-way repeated measures ANOVA for time domain
def perform_two_way_repeated_measures(df,
                                      subject_col="Subject",
                                      trial_col="Trial",
                                      bin_col="bin_idx",
                                      count_col="count",
                                      label=""):
    d = df[[subject_col, trial_col, bin_col, count_col]].copy()

    subject = d[subject_col].unique()
    trial = np.sort(d[trial_col].unique())
    time = np.sort(d[bin_col].unique())

    fullix = pd.MultiIndex.from_product([subject, trial, time],
                                        names=[subject_col, trial_col, bin_col])
    d = (d.set_index([subject_col, trial_col, bin_col])
           .reindex(fullix, fill_value=0)
           .reset_index())

    # 2) Proportions per subject×trial (Laplace smoothing to avoid 0/1)
    d["total"] = d.groupby([subject_col, trial_col])[count_col].transform("sum")
    d = d[d["total"] > 0].copy()  # drop subject×trial with no events at all
    d["prop"] = (d[count_col] + 0.5) / (d["total"] + 1.0)

    # 3) Arcsin-sqrt transform (ANOVA-friendly)
    d["Score"] = np.arcsin(np.sqrt(d["prop"]))

    # 4) Tidy column names for pingouin
    d = d.rename(columns={trial_col: "Condition", bin_col: "Time", subject_col: "Subject"})
    d["Condition"] = pd.Categorical(d["Condition"])
    d["Time"] = pd.Categorical(d["Time"])

    # 5) Two-way repeated-measures ANOVA
    anova = pg.rm_anova(dv="Score",
                        within=["Time", "Condition"],
                        subject="Subject",
                        data=d,
                        detailed=True,
                        effsize="np2")  # partial eta^2

    print(f"{label}", anova)
    return anova, d
