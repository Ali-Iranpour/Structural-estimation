#!/usr/bin/env python3
"""
Build the SMM target file from the Stata moment files.

    python3 tools/make_smm_targets.py

Writes Input/smm_targets_baseline.toml. Julia never reads .dta: the targets are
frozen into a small, readable, version-controlled file so a run is reproducible
and a change in targets shows up as a diff.

WHY EACH MOMENT IS DEFINED THE WAY IT IS
----------------------------------------
Scale. One model unit is 10,000 US dollars per year. Confirmed three ways:
ASSET_RESCALE = 10 in tables.jl; the model's mean after-tax household income is
5.2264 model units, i.e. $52,264, a plausible US figure; and the pre-existing
SMM targets in docs/SMM.md use the same x10 display. So a dollar moment enters
as  dollars / 10_000.

Time. The model's endowment is 1 and it splits as l_p + h_p + t_p = 1. The data
builds leisure as 112 - own work - own active childcare, where 112 = 168 less a
56-hour sleep allowance. That is the SAME identity, so a time moment enters as
hours per week / 112. Verified on the data: mean(leisure + work + active) is
112.00 exactly, per parent.

Per parent, not per household. The model's wage_func multiplies by 2 ("2 x mean
parental wage represents household earnings"), so one modelled adult stands for
two earners sharing a single time allocation. The data counterpart is therefore
the AVERAGE of mother and father, not the sum: (leis_mom + leis_dad) / 2 / 112.
Using leis_hh would double the target.

Consumption excludes housing. The model has no housing sector, no mortgage and
no durable stock, so imputed housing services have nothing to map onto; folding
them into c_p would compare a durable service flow against non-durable
consumption. cons_exhous_real_w99 it is. Winsorised at p99 because raw cons_real
carries a $12.07m outlier that inflates its SD to $554k -- a mean built on that
is not a moment, it is an accident.

Ages 1-17. The parent block runs t = 1..17 over child ages 1..17. Age 0 exists
in the data and has no model counterpart, so it is dropped rather than silently
averaged in.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date
import subprocess

REPO = Path(__file__).resolve().parents[1]
MICRO = REPO / "Input" / "SMM_Moments_Micro.dta"
OUT = REPO / "Input" / "smm_targets_baseline.toml"
# One by-age file per SAMPLE. Both are plotted, because they differ enough to matter:
# the cohort restriction moves assets by -28% and monetary investment by -17.8%.
BY_AGE_SOURCES = {
    "all":    ("SMM_Moments_ByAge.dta",        "smm_moments_by_age.csv",
               "all one-child families, no parental-age restriction"),
    "cohort": ("SMM_Moments_ByAge_Cohort.dta", "smm_moments_by_age_cohort.csv",
               "cohort_dad2530: father aged 25-30 at the child's birth"),
}

DOLLARS_PER_MODEL_UNIT = 10_000.0   # ASSET_RESCALE = 10, in thousands
HOURS_PER_WEEK = 112.0              # 168 less a 56-hour sleep allowance
AGE_LO, AGE_HI = 1, 17              # the parent block's t = 1..17

# Investment is split at child age 9 into early (1-9) and late (10-17) so that
# sigma_2_1, the AGE SLOPE of the money elasticity, is identified. A single
# average over all ages cannot separate the slope from the level sigma_2_0: many
# (sigma_2_0, sigma_2_1) pairs give the same overall mean, and the optimizer would
# slide along that ridge and return whichever point its seed happened to be near.
#
# Split here and the data says early 0.3532, late 0.4414 -- a 1.25x rise that a
# slope can actually be fitted to. Cross-checked against Input/SMM_Moments_ByAge.dta,
# whose N-weighted pooling over ages 1-17 reproduces the micro file exactly
# (n = 15360, mean = 0.394459, sd = 0.574738).
#
# CAVEAT worth carrying: the underlying profile is U-SHAPED, not monotone -- it
# falls from 0.353 at age 1 to a trough of 0.241 at 12, then nearly triples to
# 0.650 by 17. The model's sigma_2_t = exp(sigma_2_0 + sigma_2_1*(t-1)) is monotone
# by construction, so two group means are the most it can be asked to match. Do not
# read a good fit on these two as the model reproducing the age profile.
AGE_SPLIT = 9

# The Woodcock-Johnson composite is not administered before age 3: x_gach has 0
# observations at age 1 and exactly 1 at age 2. Selecting `Child_Age <= AGE_SPLIT` for the
# early HC moment therefore labelled the target "ages 3-9" while quietly including that
# single age-2 observation -- and, under the equal-age weighting below, it would have
# carried the same weight as age 5's 35 observations. Both sides now start at 3;
# moments.jl has the matching SMM_AGE_HC_LO.
AGE_HC_LO = 3



# =============================================================================
# MOMENT COVARIANCE -- the input standard errors need and the file did not carry
# =============================================================================
# `sd` in this file is the cross-sectional SD of one variable. It is NOT the standard
# error of a moment, and it says nothing about the COVARIANCE between two moments. Using
# per-observation SDs as if they were moment SEs is the specific mistake the review
# warned about, and inference needs the real thing:
#
#   * the moments are means, so their sampling variance falls with the number of
#     independent units -- not with the number of observations, which are repeat
#     observations of the same families;
#   * they are computed on OVERLAPPING samples. mean_c_p and mean_e_p_early are measured
#     on the same households in the same years, so their sampling errors are correlated,
#     and a diagonal weighting matrix built from SDs would get the standard errors wrong
#     in an unknown direction;
#   * each moment is an EQUAL-AGE mean -- a mean over child ages of per-age means -- so
#     an observation's influence depends on how many observations share its age.
#
# The estimator below is the standard clustered sandwich, built from influence functions.
# For moment j with age set A_j and per-age counts n_{j,a}, the influence of observation
# (i, a) is
#
#     psi_{i,j} = (1/|A_j|) * (1/n_{j,a}) * (y_{ij} - ybar_{j,a})
#
# and Omega = sum_c (sum_{i in c} psi_i)(sum_{i in c} psi_i)', clustering on the family.
# CLUSTER_ON is `Fam_id`: the same family appears in many years and, for the moments split
# by child age, in both age groups, so the family is the independent unit.
#
# WHAT THIS IS NOT. It is the covariance of the DATA moments only. It does not include
# simulation error (the model side uses a fixed seed and simN draws), and it is not itself
# a weighting matrix -- see code/smm/standard_errors.jl, which combines it with a saved
# Jacobian and says what each assumption buys.
CLUSTER_ON = "Fam_id"

# The moments the estimator actually targets, in SMM_MOMENTS order. `mean_l_p` and the
# pooled `mean_e_p` are written to the file for reference but are not targeted, so they
# are not part of the covariance the weighting matrix would be built from.
TARGETED = ["mean_c_p", "mean_h_p",
            "mean_t_p_early", "mean_t_p_late",
            "mean_e_p_early", "mean_e_p_late",
            "mean_i_c_early", "mean_i_c_late",
            "mean_hc_early", "mean_hc_late"]


def moment_influence(series, ages, clusters):
    """Per-cluster influence of an equal-age mean. Returns a Series indexed by cluster."""
    s = series.dropna()
    a = ages.loc[s.index]
    c = clusters.loc[s.index]
    n_age = a.map(a.value_counts())            # observations sharing this observation's age
    ybar_age = a.map(s.groupby(a).mean())      # that age's own mean
    psi = (s - ybar_age) / (n_age * a.nunique())
    return psi.groupby(c).sum()


def moment_covariance(moments, r):
    """Cluster-robust covariance of the targeted moment vector."""
    clusters = r[CLUSTER_ON]
    infl = {}
    for mo in moments:
        infl[mo["name"]] = moment_influence(mo["series"], r["Child_Age"], clusters)
    names = [mo["name"] for mo in moments]
    # One row per cluster, one column per moment; a cluster that contributes nothing to a
    # moment contributes a zero, not a dropped row -- that is what carries the overlap.
    all_clusters = sorted(set().union(*(set(v.index) for v in infl.values())))
    P = np.zeros((len(all_clusters), len(names)))
    idx = {c: i for i, c in enumerate(all_clusters)}
    for j, nm in enumerate(names):
        for c, v in infl[nm].items():
            P[idx[c], j] = v
    Omega = P.T @ P
    n_cl = {nm: int((np.abs(P[:, j]) > 0).sum()) for j, nm in enumerate(names)}
    return names, Omega, n_cl, len(all_clusters)


def git_sha():
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            text=True).strip()
    except Exception:
        return "unknown"


def main():
    m = pd.read_stata(MICRO)
    r = m[(m.Child_Age >= AGE_LO) & (m.Child_Age <= AGE_HI)].copy()

    # ------------------------------------------------------------------
    # t_p USES par_time_tot, BY INSTRUCTION (2026-08-28). READ THIS BEFORE
    # INTERPRETING phi_2_0 OR ANY LEISURE NUMBER.
    # ------------------------------------------------------------------
    # par_time_tot is the broader time concept -- active PLUS nearby/supervisory
    # presence -- chosen deliberately over per-parent active time. Two properties
    # of it are worth having in front of you:
    #
    # (1) It is a CHILD-side union, not a per-parent allocation. par_time_act
    #     (27.38) sits between max(mom, dad) = 22.07 and the sum = 36.69, which is
    #     the signature of "time the child spent with AT LEAST ONE parent", not an
    #     average over parents. The model's t_p is the PARENT's own time out of a
    #     112-hour week. These are different objects.
    #
    # (2) It does not fit an exhaustive time budget, because nearby time overlaps
    #     with leisure and work. Measured, per parent:
    #         leis_mom + wh_mom + Mom_Total_Act = 112.00   <- exact
    #         leis_mom + wh_mom + par_time_act  = 117.28
    #         leis_mom + wh_mom + par_time_tot  = 133.25   <- +21 hrs
    #
    # CONSEQUENCE, which does not go away by leaving l_p untargeted: the model
    # enforces l_p + h_p + t_p = 1 identically, so targeting h_p = 0.3070 and
    # t_p = 0.3874 FORCES model leisure to 0.3056, i.e. 34.2 hrs/wk -- against the
    # 59.2 hrs/wk this same dataset measures. That 25-hour gap has to land
    # somewhere, and where it lands is phi_2_0, the taste for leisure. Do not read
    # the estimated phi_2_0 as a preference parameter without this caveat.
    #
    # To revert to the budget-consistent measure, set:
    #     r["t_share"] = ((r.Mom_Total_Act + r.Dad_Total_Act) / 2.0) / HOURS_PER_WEEK
    # and the identity closes exactly (112.00 for both parents).
    r["leis_share"] = ((r.leis_mom + r.leis_dad) / 2.0) / HOURS_PER_WEEK
    r["h_share"] = ((r.wh_mom + r.wh_dad) / 2.0) / HOURS_PER_WEEK
    r["t_share"] = r.par_time_tot / HOURS_PER_WEEK

    moments = [
        dict(name="mean_c_p",
             series=r.cons_exhous_real_w99 / DOLLARS_PER_MODEL_UNIT,
             source="cons_exhous_real_w99",
             units="model units (10k USD/yr, real 2015)",
             model="mean of sim_c over t = 1..17"),
        # l_p is NO LONGER TARGETED -- kept for reference. Targeting h_p and t_p
        # instead is strictly more information: l_p = 1 - h_p - t_p, so matching
        # leisure alone pins the SUM of work and child time but not the split, and
        # the split is where the model was wrong (work 14% below data, child time
        # 27% above, the two errors cancelling inside the leisure moment).
        dict(name="mean_l_p",
             series=r.leis_share,
             source="(leis_mom + leis_dad)/2 / 112",
             units="share of the 112h non-sleep week, per parent",
             model="mean of 1 - sim_h - sim_t over t = 1..17"),
        # h_p is flat in child age (0.3062 early vs 0.3080 late), so one pooled
        # mean is right -- and it carries 15,665 observations against t_p's 1,065,
        # because work hours are measured for everyone and time diaries only for
        # the CDS subsample.
        dict(name="mean_h_p",
             series=r.h_share,
             source="(wh_mom + wh_dad)/2 / 112",
             units="share of the 112h non-sleep week, per parent",
             model="mean of sim_h over t = 1..17"),
        # t_p IS split, because sigma_1_1 (the age slope of the HC elasticity to
        # parent TIME) needs a second moment exactly as sigma_2_1 did. On
        # par_time_tot the profile runs 52.3 -> 36.2 hrs/wk, late/early 0.692x --
        # monotone, so exp(sigma_1_0 + sigma_1_1*(t-1)) can reproduce its shape.
        # (Per-parent active time falls faster, 25.1 -> 12.9, 0.512x.)
        dict(name="mean_t_p_early",
             series=r[r.Child_Age <= AGE_SPLIT].t_share,
             source=f"par_time_tot / 112, child ages {AGE_LO}-{AGE_SPLIT}",
             units="share of the 112h non-sleep week (active+nearby, child-side union)",
             model=f"mean of sim_t over t = {AGE_LO}..{AGE_SPLIT}"),
        dict(name="mean_t_p_late",
             series=r[r.Child_Age > AGE_SPLIT].t_share,
             source=f"par_time_tot / 112, child ages {AGE_SPLIT+1}-{AGE_HI}",
             units="share of the 112h non-sleep week (active+nearby, child-side union)",
             model=f"mean of sim_t over t = {AGE_SPLIT+1}..{AGE_HI}"),
        # Kept for reference and for switching back to the 3-moment design; the
        # estimation targets the two age groups below instead. See SMM_MOMENTS in
        # code/smm/moments.jl for which set is live.
        dict(name="mean_e_p",
             series=r.m_method2_final_w99 / DOLLARS_PER_MODEL_UNIT,
             source="m_method2_final_w99",
             units="model units (10k USD/yr, real 2015)",
             model="mean of sim_e over t = 1..17"),
        dict(name="mean_e_p_early",
             series=r[r.Child_Age <= AGE_SPLIT].m_method2_final_w99 / DOLLARS_PER_MODEL_UNIT,
             source=f"m_method2_final_w99, child ages {AGE_LO}-{AGE_SPLIT}",
             units="model units (10k USD/yr, real 2015)",
             model=f"mean of sim_e over t = {AGE_LO}..{AGE_SPLIT}"),
        dict(name="mean_e_p_late",
             series=r[r.Child_Age > AGE_SPLIT].m_method2_final_w99 / DOLLARS_PER_MODEL_UNIT,
             source=f"m_method2_final_w99, child ages {AGE_SPLIT+1}-{AGE_HI}",
             units="model units (10k USD/yr, real 2015)",
             model=f"mean of sim_e over t = {AGE_SPLIT+1}..{AGE_HI}"),
    ]

    # i_c: the child's OWN study time. It only exists in the model from t = T_CHILD_VOICE
    # = 6, so the early group starts at 6, not 1 -- averaging in ages 1-5, where the model
    # has no child decision at all, would target a number the model cannot produce.
    r["i_share"] = r.study_hrs / HOURS_PER_WEEK
    # HC in LOGS. The model now carries HC in the data's own W-score units, so this is a
    # like-for-like comparison -- and it is what separates the VALUATION parameters
    # (phi_3, lambda_2) from the TECHNOLOGY parameters (R_0, sigma_1, sigma_2, sigma_4),
    # which are otherwise collinear: both raise investment, only technology raises HC.
    # Logs, because that is the form the production function uses and the SD is far more
    # stable in logs (0.083 at age 3 to 0.031 at 17).
    moments += [
        dict(name="mean_i_c_early",
             series=r[(r.Child_Age >= 6) & (r.Child_Age <= AGE_SPLIT)].i_share,
             source=f"study_hrs / 112, child ages 6-{AGE_SPLIT}",
             units="share of the 112h non-sleep week",
             model=f"mean of sim_i over t = 6..{AGE_SPLIT}"),
        dict(name="mean_i_c_late",
             series=r[r.Child_Age > AGE_SPLIT].i_share,
             source=f"study_hrs / 112, child ages {AGE_SPLIT+1}-{AGE_HI}",
             units="share of the 112h non-sleep week",
             model=f"mean of sim_i over t = {AGE_SPLIT+1}..{AGE_HI}"),
        dict(name="mean_hc_early",
             series=r[(r.Child_Age >= AGE_HC_LO) & (r.Child_Age <= AGE_SPLIT)].x_gach,
             source=f"x_gach (log PCA composite), child ages {AGE_HC_LO}-{AGE_SPLIT}",
             units="log W-score; the model's HC is in the SAME units after the rescaling",
             model=f"mean of log(sim_hc) over t = {AGE_HC_LO}..{AGE_SPLIT}"),
        dict(name="mean_hc_late",
             series=r[r.Child_Age > AGE_SPLIT].x_gach,
             source=f"x_gach (log PCA composite), child ages {AGE_SPLIT+1}-{AGE_HI}",
             units="log W-score; the model's HC is in the SAME units after the rescaling",
             model=f"mean of log(sim_hc) over t = {AGE_SPLIT+1}..{AGE_HI}"),
    ]

    lines = [
        "# SMM targets, baseline parent block. GENERATED by tools/make_smm_targets.py.",
        "# Do not edit by hand -- rerun the script.",
        "#",
        "# One model unit = 10,000 USD/yr. Time is a share of the 112h non-sleep week,",
        "# per parent (the model's single adult stands for two earners: wage_func x2).",
        "# Child ages 1-17 only, matching the parent block's t = 1..17.",
        "",
        f'generated  = "{date.today().isoformat()}"',
        f'git_commit = "{git_sha()}"',
        f'source     = "Input/SMM_Moments_Micro.dta"',
        f'age_range  = [{AGE_LO}, {AGE_HI}]',
        f'age_split  = {AGE_SPLIT}   # early = {AGE_LO}..{AGE_SPLIT}, late = {AGE_SPLIT+1}..{AGE_HI}',
        f'dollars_per_model_unit = {DOLLARS_PER_MODEL_UNIT}',
        f'hours_per_week = {HOURS_PER_WEEK}',
        "",
    ]

    # EQUAL WEIGHT PER CHILD AGE, not per observation.
    #
    # The model's moment is a mean over (family, age) cells with every simulated family
    # present at every age, so each age carries exactly 1/17 of the weight. Pooling the
    # micro data instead weights each age by how many observations it happens to have,
    # and the counts are far from uniform -- investment ranges 556 to 1754 observations
    # per age, parental time 36 to 112. The two sides were therefore computing different
    # statistics and the optimizer was asked to absorb the difference in the parameters.
    #
    # MEASURED cost of the mismatch (pooled -> equal-age):
    #     parental time, early   0.4672 -> 0.4544   (-2.7%)
    #     parental time, late    0.3232 -> 0.3333   (+3.1%)
    #     investment,    early   0.3532 -> 0.3429   (-2.9%)
    #     investment,    late    0.4414 -> 0.3911   (-11.4%)
    #
    # Equal-age is the side that moved, because it is the side the model fixes: the
    # simulation has no age composition to match. `mean_pooled` is still written so the
    # change stays auditable.
    print(f"{'moment':12s} {'N':>7s} {'mean':>10s} {'pooled':>10s} {'sd':>10s}   source")
    print("-" * 84)
    for mo in moments:
        s = mo["series"].dropna()
        ages = r.loc[s.index, "Child_Age"]
        mean_equal  = s.groupby(ages).mean().mean()
        mean_pooled = s.mean()
        lines += [
            f"[{mo['name']}]",
            f'source = "{mo["source"]}"',
            f'units  = "{mo["units"]}"',
            f'model  = "{mo["model"]}"',
            f"n      = {len(s)}",
            f"n_ages = {ages.nunique()}",
            f"mean   = {mean_equal:.6f}",
            f"mean_pooled = {mean_pooled:.6f}   # observation-weighted; NOT what is targeted",
            # SD is recorded but NOT targeted in the baseline run: the model's only
            # cross-sectional heterogeneity is a 5-node wage shock plus initial asset,
            # HC and college draws, and it cannot reach the data's dispersion (leisure
            # SD is 7.4x too small). Kept here so a later run can weight it in
            # deliberately rather than rediscover the number.
            f"sd     = {s.std():.6f}",
            f"median = {s.median():.6f}",
            "",
        ]
        print(f"{mo['name']:12s} {len(s):7d} {mean_equal:10.4f} {mean_pooled:10.4f} "
              f"{s.std():10.4f}   {mo['source']}")

    # ---- clustered moment covariance ------------------------------------------
    targeted = [mo for mo in moments if mo["name"] in TARGETED]
    names, Omega, n_cl, n_clusters = moment_covariance(targeted, r)
    se = np.sqrt(np.diag(Omega))
    D = np.diag(1.0 / np.where(se > 0, se, 1.0))
    Corr = D @ Omega @ D

    lines += [
        "# ---------------------------------------------------------------------------",
        "# Cluster-robust covariance of the TARGETED moment vector.",
        "#",
        f"# Clustered on {CLUSTER_ON} ({n_clusters} families). `se` is the standard error of",
        "# each moment -- NOT the cross-sectional `sd` above, which is a different quantity",
        "# and is 20-100x larger. `cov` is row-major over `names`; `corr` is the same matrix",
        "# scaled to unit diagonal, which is the readable one.",
        "#",
        "# Used by code/smm/standard_errors.jl. Read its header before quoting anything",
        "# built on this: it is the covariance of the DATA moments, and it does not include",
        "# simulation error.",
        "[moment_cov]",
        "cluster_on = \"" + CLUSTER_ON + "\"",
        f"n_clusters = {n_clusters}",
        "names      = [" + ", ".join(f'"{n}"' for n in names) + "]",
        "n_clusters_by_moment = [" + ", ".join(str(n_cl[n]) for n in names) + "]",
        "se         = [" + ", ".join(f"{v:.10g}" for v in se) + "]",
        "cov        = [",
    ]
    for i in range(len(names)):
        lines.append("  [" + ", ".join(f"{Omega[i, j]:.10g}" for j in range(len(names))) + "],")
    lines += ["]", "corr       = ["]
    for i in range(len(names)):
        lines.append("  [" + ", ".join(f"{Corr[i, j]:.6f}" for j in range(len(names))) + "],")
    lines += ["]", ""]

    print()
    print(f"{'moment':16s} {'se':>12s} {'sd':>12s}   se/sd   clusters")
    print("-" * 66)
    for j, nm in enumerate(names):
        sd = next(mo["series"].dropna().std() for mo in targeted if mo["name"] == nm)
        print(f"{nm:16s} {se[j]:12.6f} {sd:12.6f} {se[j]/sd:7.4f} {n_cl[nm]:10d}")
    off = Corr[np.triu_indices(len(names), 1)]
    print(f"\nmoment correlations: min {off.min():+.3f}  max {off.max():+.3f}  "
          f"|corr|>0.3 in {int((np.abs(off) > 0.3).sum())} of {len(off)} pairs")

    OUT.write_text("\n".join(lines))
    print(f"\nwrote {OUT.relative_to(REPO)}")
    write_by_age()


def write_by_age():
    """
    Per-child-age means, for overlaying the data on the baseline figure.

    A SECOND kind of file, and CSV rather than TOML, because it is tabular and read by the
    notebook rather than by the estimation. Julia has no Stata reader in this project's
    Project.toml, so the same rule as the targets file applies: Python touches the .dta,
    Julia reads a small tracked text file. `DelimitedFiles.readdlm` is stdlib, so the
    notebook needs no new dependency.

    TWO files, one per sample, and the difference between them is not cosmetic:

        moment                    all families    cohort (dad 25-30)    diff
        assets_real                  191,200          137,672          -28.0%
        m_method2_final_w99            3,945            3,243          -17.8%
        cons_exhous_real_w99          31,577           29,162           -7.7%
        par_time_tot                   43.39            45.00           +3.7%
        leis_mom_wk                    84.55            84.64           +0.1%

    The MODEL assumes parents are aged 26 at the child's birth, which is what the cohort
    file restricts to -- so the cohort sample is the one the model actually describes,
    while the SMM targets are currently built from the unrestricted micro file. Money
    moments move a lot under the restriction and time moments barely at all. Plotting
    both makes that visible instead of leaving it as an assumption.

    Units match the model's, so the notebook can plot these directly:
      c_p, e_p, a_p   model units (10k USD/yr)     t_p, h_p, i_c   share of the 112h week
      x_gach, x_lw    LOG human capital -- comparable to log(model hc) up to an ADDITIVE
                      constant log(M), the scale factor that was never applied.
    """
    # THIS STAGE IS OPTIONAL AND MUST NOT BLOCK THE TARGETS.
    #
    # The by-age CSVs are a plotting convenience for the notebook; the targets file above
    # is what the estimation reads. They were previously written in the same pass with no
    # guard, so when Input/ carried a .dta that lacked a column the script crashed AFTER
    # writing the targets -- leaving a non-zero exit on a run that had in fact succeeded at
    # its main job. Measured 2026-09-06: SMM_Moments_ByAge.dta has no `mu_assets_real`
    # column and SMM_Moments_ByAge_Cohort.dta is not in Input/ at all, so BOTH by-age
    # outputs are stale relative to the .dta files actually present. The committed CSVs
    # were generated from a newer extract that is not in the repository.
    #
    # Skipping loudly is the honest behaviour: the CSVs on disk are left untouched and
    # named as stale, rather than half-rewritten or silently accepted.
    for key, (src, dst, note) in BY_AGE_SOURCES.items():
        path = REPO / "Input" / src
        if not path.exists():
            print(f"  SKIP {dst}: {src} is not in Input/. "
                  f"The committed CSV is left as it is and is STALE.")
            continue
        d = pd.read_stata(path)
        missing = [c for c in ("mu_cons_exhous_real_w99", "mu_m_method2_final_w99",
                               "mu_assets_real", "mu_leis_mom_wk", "mu_leis_dad_wk",
                               "mu_par_time_tot", "mu_study_hrs", "mu_x_gach", "mu_x_lw")
                   if c not in d.columns]
        if missing:
            print(f"  SKIP {dst}: {src} is missing {', '.join(missing)}. "
                  f"The committed CSV is left as it is and is STALE.")
            continue
        d = d[(d.Child_Age >= AGE_LO) & (d.Child_Age <= AGE_HI)].sort_values("Child_Age")
        out = pd.DataFrame({
            "child_age": d.Child_Age.astype(int),
            "c_p": d.mu_cons_exhous_real_w99 / DOLLARS_PER_MODEL_UNIT,
            "e_p": d.mu_m_method2_final_w99 / DOLLARS_PER_MODEL_UNIT,
            # Net worth EXCLUDING home equity. The model has no housing sector, no
            # mortgage and no durable stock, so home equity has nothing to map onto --
            # the same reason consumption uses cons_exhous_real.
            "a_p": d.mu_assets_real / DOLLARS_PER_MODEL_UNIT,
            # work is not stored directly by age; leis_*_wk IS 112 - own work, so invert it
            "h_p": (((HOURS_PER_WEEK - d.mu_leis_mom_wk) +
                     (HOURS_PER_WEEK - d.mu_leis_dad_wk)) / 2.0) / HOURS_PER_WEEK,
            "t_p": d.mu_par_time_tot / HOURS_PER_WEEK,
            "i_c": d.mu_study_hrs / HOURS_PER_WEEK,
            "x_gach": d.mu_x_gach,
            "x_lw":   d.mu_x_lw,
        })
        (REPO / "Input" / dst).write_text(
            f"# Per-child-age data means for the baseline figure. Sample: {note}.\n"
            f"# GENERATED by tools/make_smm_targets.py from {src} -- do not edit by hand.\n"
            "# c_p, e_p, a_p: model units (10k USD/yr). a_p EXCLUDES home equity.\n"
            "# h_p, t_p, i_c: share of the 112h week.\n"
            "# x_gach / x_lw: LOG human capital (PCA composite / Letter-Word), comparable\n"
            "#   to log(model hc) up to an ADDITIVE constant log(M).\n"
            + out.to_csv(index=False, float_format="%.6f"))
        print(f"wrote Input/{dst}  ({len(out)} ages, {key})")

if __name__ == "__main__":
    main()
