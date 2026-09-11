#!/usr/bin/env python3
"""
Build the SMM target file from the Stata moment files.

    python3 tools/make_smm_targets.py

Writes output/smm_runs/<timestamp>_targets/targets.toml. Julia never reads .dta: the targets are
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
from datetime import date, datetime
import subprocess

REPO = Path(__file__).resolve().parents[1]
MICRO = REPO / "Input" / "SMM_Moments_Micro.dta"
OUT = REPO / "output" / "smm_runs" / (datetime.now().strftime("%Y-%m-%d_%H%M%S_%f") + "_targets") / "targets.toml"
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
AGE_HC_LATE_LO = 12
CHILD_TIME_SPEC = "own_study_fixed_school_v1"


# =============================================================================
# THE TAS BLOCK -- seven child-level moments for the four kappa parameters
# =============================================================================
# A SECOND SAMPLE, not more rows of the first. `Input/SMM_TAS_Micro.dta` is one row per
# TAS-linked child (4,248 of them, 1,481 family clusters); `SMM_Moments_Micro.dta` is one
# row per child-YEAR of the PSID one-child panel (17,791 rows, 1,825 clusters). They
# overlap in 492 clusters and in nothing else -- different unit of observation, different
# frame, different weighting rule. So they are built separately here and joined only at
# the covariance, on the cluster key they share.
#
# UNWEIGHTED, per Input/CODEBOOK.md. CHILD_WT is a CDS CHILDHOOD weight, non-missing for
# 424 of 2,603 records, and the codebook's own check reports that the full-frame-vs-
# weighted-subset difference is mostly selection rather than weighting. Applying it would
# reweight to a population neither file describes.
TAS_MICRO = REPO / "Input" / "SMM_TAS_Micro.dta"

# The cluster key. `famclust` in the TAS file and `Fam_id` in the parent file are the SAME
# object -- the PSID 1968 family interview number, ER30001 -- which is what makes a joint
# covariance possible at all. Siblings share it, and so does a TAS child with the
# child-years of the same lineage in the parent block.
TAS_CLUSTER_ON = "famclust"

# COMPLETION, NOT ENTRY (decision 2026-09-10, and the codebook's own primary outcome).
# The model's college path is binary and has no dropout: enrol, study t_college = 4 years,
# then earn the graduate wage E = 1. Nobody enrols without finishing, so the path IS a
# completed four-year degree. Entry (0.6162) counts respondents who never receive that
# premium and would be matched against a mechanism that always pays it. `y_entry` is
# carried through to the target file as an untargeted diagnostic.
TAS_OUTCOME = "y_complete"
TAS_FOLLOWUP = "hf_complete"      # OUTCOME-SPECIFIC follow-up mask -- see below

# FOLLOW-UP IS PER OUTCOME, NOT PER CHILD. `hf_entry` and `hf_complete` are different
# masks: 2,603 children have a usable entry observation at age 23-25 and 2,771 a usable
# completion one, and neither is a subset of the other. Using a single `has_followup` for
# both would silently change the denominator of whichever moment it did not belong to.

# ACHIEVEMENT: age 17, g_ACH, completion frame.
# Age 17 over age 18 because the existing HC targets are the g_ACH composite and the
# age-17 panel is far larger (455 children with a last-CDS wave at 17 against 174 at 18);
# the age-18 group comes almost entirely from CDS-2002/2007 since CDS-2014 and CDS-2019
# stop at 17. LW, age 18 and the entry variants stay out of the targeted set and are
# written as sensitivity rows.
#
# TERTILES ARE CUT WITHIN AGE GROUP AND ON THE COMPLETION FRAME, so T1/T2/T3 are not
# comparable in level between the age-17 and age-18 panels -- only the gradient within a
# panel is. `tert_ga` in the micro file already carries that cut; it is used as given
# rather than recut here, so the target and the published moment cannot drift apart.
TAS_ACH_AGE = 17
TAS_ACH_TERT = "tert_ga"

# TERMINAL WEALTH: strict definition, net worth EXCLUDING home equity, WINSORISED AT p99.
#
# WHY WINSORISED (decision 2026-09-10). The raw moment is a mean of $429,803 on a median
# of $49,243, an SD of $1,902,538 and a maximum of $41.3m; the top 1% alone moves the mean
# by ~$98k. That is the same pathology this file already handles on the parent side --
# `cons_real` carries a $12.07m outlier and is targeted winsorised at p99 because "a mean
# built on that is not a moment, it is an accident". The model cannot produce the tail
# either: its asset grid tops out at a_max = 100 model units = $1m.
#
# NEGATIVE NET WORTH IS RETAINED, not dropped -- 13.0% of the qualifying sample is
# negative, and dropping it would bias the target upward on top of everything else. Note
# that the model CANNOT reproduce it: retained assets are floored at delta_P. That is a
# recorded limitation of the moment, not something to fix by censoring the data.
#
# STRICT over loose/broad because it is the definition that comes closest to the model's
# concept -- separated, not in a parental home, and no tuition/housing/bills support -- so
# the parent has stopped paying for college by the time wealth is measured.
TAS_WEALTH_DEF = "strict"
TAS_WEALTH_VAR = "pwx_strict"     # excl. home equity; pwi_* is the incl.-home variant
TAS_WEALTH_WINSOR_P = 99.0

# TIMING GAP -- OPEN, and not corrected here. The model's object is the parent's retained
# assets at the transfer, i.e. at child age 18. The data is the LATEST parental wealth
# observation at or after the first qualifying wave: median gap 4 years, up to 16, and a
# median child age at measurement of about 29. Parents keep accumulating over that decade,
# so the target is measured later in the parental life cycle than the model's counterpart.
# The model has no post-separation parent to age forward, so this cannot be closed without
# a new mechanism. It is carried as a limitation on kappa_terminal.

TAS_MOMENTS = ["k0_complete",
               "kth_ga17_t1_c", "kth_ga17_t2_c", "kth_ga17_t3_c",
               "kpe_g0_c", "kpe_g1_c",
               "kterm_x_strict_w99"]



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
PARENT_TARGETED = ["mean_c_p", "mean_h_p",
                   "mean_t_p_early", "mean_t_p_late",
                   "mean_e_p_early", "mean_e_p_late",
                   "mean_i_c_early", "mean_i_c_late",
                   "mean_hc_early", "mean_hc_late"]

# SEVENTEEN moments against FOURTEEN parameters -- over-identified, so the weighting
# matrix now changes the answer in a way it could not when the system was square. The
# order here IS the order of `SMM_MOMENTS` in code/smm/moments.jl and of every row and
# column of the covariance below; moments.jl checks it and refuses to run if they differ.
TARGETED = PARENT_TARGETED + TAS_MOMENTS


def moment_influence(series, ages, clusters):
    """Per-cluster influence of an equal-age mean. Returns a Series indexed by cluster."""
    s = series.dropna()
    a = ages.loc[s.index]
    c = clusters.loc[s.index]
    n_age = a.map(a.value_counts())            # observations sharing this observation's age
    ybar_age = a.map(s.groupby(a).mean())      # that age's own mean
    psi = (s - ybar_age) / (n_age * a.nunique())
    return psi.groupby(c).sum()


def ratio_influence(num, den, clusters):
    """
    Estimate and per-cluster influence of a RATIO OF MEANS, the form every TAS moment
    takes: mean(numerator)/mean(denominator) over the WHOLE frame, with both zero off the
    subgroup. That is exactly how `23_smm_tas_moments.do` estimates them
    (`ratio ..., cluster(famclust)`), and reproducing the form is what makes the
    reconstructed covariance the same object as the one the do-file exports.

        r        = mean(num) / mean(den)
        psi_i    = (num_i - r*den_i) / mean(den) / N
        Var(r)   = sum_c (sum_{i in c} psi_i)^2 ,  clustered on the family

    N is the WHOLE-FRAME row count, not the subgroup's -- the subgroup indicator lives
    inside `den`. Getting that wrong rescales every standard error.

    VERIFIED: this reproduces all seven published estimates AND all seven published
    standard errors in `Input/SMM_TAS_Moments.csv` to six decimal places. That is why the
    missing `SMM_TAS_VCov.dta` does not have to be requested -- see the codebook's
    "Standard errors and the covariance matrix" section for what is being reproduced.
    """
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    n = len(den)
    dbar = den.mean()
    if dbar <= 0:
        raise ValueError("ratio moment has an empty denominator")
    r = num.mean() / dbar
    psi = pd.Series((num - r * den) / dbar / n, index=clusters.index)
    return r, psi.groupby(clusters).sum()


def joint_covariance(influences, names, dof_correct=True):
    """
    Cluster-robust covariance of a moment vector assembled from SEVERAL BLOCKS.

    `influences` maps a moment name to a per-cluster influence Series. The blocks are
    measured on different samples with different units of observation, and the whole point
    of stacking them here is that they are NOT independent: 492 of the TAS block's 1,481
    family clusters also appear among the parent block's 1,825, so a child whose
    completion outcome enters `k0_complete` can be from the same 1968 lineage as the
    child-years behind `mean_c_p`.

    A cluster that contributes nothing to a moment contributes a ZERO to that column, not
    a dropped row. That is what carries the overlap: the cross-block covariance comes
    entirely from the rows where both columns are non-zero, and dropping rows would set
    those terms to zero by construction -- which is the "silently assume independent
    blocks" mistake.

    `dof_correct` applies the G/(G-1) finite-cluster correction used by Stata's
    `cluster()`. It is applied PER BLOCK in the published SEs (each moment's own cluster
    count), so it is applied here on the joint cluster count and the per-moment counts are
    reported separately; the difference is under 0.1% at these G.
    """
    all_clusters = sorted(set().union(*(set(v.index) for v in influences.values())))
    idx = {c: i for i, c in enumerate(all_clusters)}
    P = np.zeros((len(all_clusters), len(names)))
    for j, nm in enumerate(names):
        v = influences[nm]
        P[[idx[c] for c in v.index], j] = v.values
    G = len(all_clusters)
    Omega = P.T @ P
    if dof_correct and G > 1:
        Omega = Omega * (G / (G - 1.0))
    n_cl = {nm: int((np.abs(P[:, j]) > 0).sum()) for j, nm in enumerate(names)}
    return Omega, n_cl, G


def moment_covariance(moments, r):
    """Per-cluster influence of each parent-block moment, keyed on the cluster id."""
    clusters = r[CLUSTER_ON]
    return {mo["name"]: moment_influence(mo["series"], r["Child_Age"], clusters)
            for mo in moments}


def winsorise(x, p):
    """Upper-tail winsorisation at percentile `p`. The LOWER tail is left alone."""
    x = np.asarray(x, dtype=float)
    return np.minimum(x, np.percentile(x, p))


def build_tas_moments(t):
    """
    The seven TAS targets, plus the untargeted diagnostics, as ratio-of-means moments.

    Every one is defined on the FULL TAS frame with the subgroup indicator inside the
    denominator, so `n_obs` below is the subgroup size (the denominator's support) while
    the influence function runs over all 4,248 rows. Both are reported.
    """
    n_all = len(t)
    out = []

    def add(name, mask, value, *, source, units, model, block, targeted=True, note=""):
        mask = np.asarray(mask, dtype=float)
        value = np.asarray(value, dtype=float)
        # A NaN outcome off the subgroup must not poison the product; the mask is what
        # decides membership, so the value is only ever read where the mask is 1.
        num = np.where(mask > 0, np.nan_to_num(value, nan=0.0), 0.0)
        out.append(dict(name=name, num=num, den=mask, source=source, units=units,
                        model=model, block=block, targeted=targeted, note=note))

    y = t[TAS_OUTCOME]
    hf = (t[TAS_FOLLOWUP] == 1)

    # ---- kappa_0: the overall completion rate --------------------------------
    add("k0_complete", hf, y,
        source=f"{TAS_OUTCOME} | {TAS_FOLLOWUP}",
        units="share of linked children completing a four-year degree by age 25",
        model="share of resimulated children with path_choice == :college",
        block="kappa_0")
    # Untargeted: the entry series, kept visible so the entry/completion gap that decided
    # this specification stays in the file rather than only in the codebook.
    add("k0_entry", (t.hf_entry == 1), t.y_entry,
        source="y_entry | hf_entry",
        units="share entering a four-year track by age 25",
        model="(not targeted; the model has no entry-without-completion state)",
        block="kappa_0", targeted=False,
        note="degree SOUGHT, runs ~0.62 against a national ~0.45; see CODEBOOK")

    # ---- kappa_theta: completion by age-17 achievement tertile ----------------
    for k in (1, 2, 3):
        add(f"kth_ga17_t{k}_c",
            hf & (t.ach_age == TAS_ACH_AGE) & (t[TAS_ACH_TERT] == k), y,
            source=f"{TAS_OUTCOME} | {TAS_FOLLOWUP}, ach_age=={TAS_ACH_AGE}, {TAS_ACH_TERT}=={k}",
            units="completion share within the achievement tertile",
            model=f"college share within model-internal tertile {k} of sim_hc at child age 17",
            block="kappa_theta")

    # ---- kappa_theta SENSITIVITY VARIANTS, untargeted -------------------------
    # `docs/SMM_14PARAM_TAS.md` promised these and the generator did not build them, so
    # the "sensitivity checks" the specification calls for did not exist. They are cheap:
    # the same ratio moment on a different achievement measure or age panel.
    #
    # NOT COMPARABLE IN LEVEL ACROSS PANELS. Tertiles are cut WITHIN age group, so the
    # age-17 T1 and the age-18 T1 are different ability cuts. Only the gradient within a
    # panel means anything, which is why these are diagnostics and not alternative targets.
    for (tag, tvar, age) in (("lw17", "tert_lw", 17), ("lw18", "tert_lw", 18),
                             ("ga18", "tert_ga", 18)):
        for k in (1, 2, 3):
            add(f"kth_{tag}_t{k}_c", hf & (t.ach_age == age) & (t[tvar] == k), y,
                source=f"{TAS_OUTCOME} | {TAS_FOLLOWUP}, ach_age=={age}, {tvar}=={k}",
                units="completion share within the achievement tertile",
                model=f"(not targeted; sensitivity variant of kth_ga17_t{k}_c)",
                block="kappa_theta", targeted=False,
                note="tertiles are cut WITHIN age group -- levels are not comparable "
                     "to the age-17 g_ACH panel, only the gradient is")

    # ---- kappa_ParEd: completion by parental education -----------------------
    # OPEN MISMATCH, by instruction 2026-09-10: pared_col is EITHER-parent 16+, the model's
    # BothCollege is both. Targeted as supplied and recorded as P7c in docs/ERRORS.md.
    for g in (0, 1):
        add(f"kpe_g{g}_c", hf & (t.pared_col == g), y,
            source=f"{TAS_OUTCOME} | {TAS_FOLLOWUP}, pared_col=={g}",
            units="completion share within the parental-education group",
            model=f"college share among simulated parents with BothCollege == {g}",
            block="kappa_ParEd",
            note="EITHER-parent 16+ in data vs BothCollege in model -- see ERRORS.md P7c")
    # The unknown group is its own moment and is NOT folded into g0. The codebook is
    # explicit that doing so estimates "no OBSERVED college parent", a different quantity.
    # `pared_unknown`, NOT `pared_col.isna()`. The two differ by 13 children on the
    # completion frame -- those with NEITHER parent observed, which `pared_unknown`
    # (exactly one parent observed, and below 16) excludes. Using the published variable
    # reproduces the published `kpe_gu_c` exactly; using the missing-mask does not.
    add("kpe_gu_c", hf & (t.pared_unknown == 1), y,
        source=f"{TAS_OUTCOME} | {TAS_FOLLOWUP}, pared_unknown==1",
        units="completion share where at least one parent's education is unobserved",
        model="(not targeted; the model has no unknown-education state)",
        block="kappa_ParEd", targeted=False)

    # ---- kappa_terminal: parental net worth after independence ---------------
    wmask = (t[f"ever_{TAS_WEALTH_DEF}"] == 1) & t[TAS_WEALTH_VAR].notna()
    raw = t[TAS_WEALTH_VAR].where(wmask)
    cut = np.percentile(raw.dropna(), TAS_WEALTH_WINSOR_P)
    wins = raw.clip(upper=cut)
    add("kterm_x_strict_w99", wmask, wins / DOLLARS_PER_MODEL_UNIT,
        source=f"{TAS_WEALTH_VAR} winsorised at p{TAS_WEALTH_WINSOR_P:g} "
               f"(cut = {cut:,.0f} USD) | ever_{TAS_WEALTH_DEF}",
        units="model units (10k USD, real 2015)",
        model="mean of (parent sim_a at T+1) - transfer, i.e. assets retained AFTER the transfer",
        block="kappa_terminal",
        note=f"{100*(raw < 0).mean():.1f}% of the qualifying sample is negative and is RETAINED; "
             "the model floors retained assets at delta_P and cannot reproduce it")
    # Untargeted: the raw mean, so the effect of the winsorisation is auditable, and the
    # incl.-home variant.
    add("kterm_x_strict_raw", wmask, raw / DOLLARS_PER_MODEL_UNIT,
        source=f"{TAS_WEALTH_VAR} | ever_{TAS_WEALTH_DEF} (NOT winsorised)",
        units="model units (10k USD, real 2015)",
        model="(not targeted; the p99 winsorised variant is)",
        block="kappa_terminal", targeted=False)
    add("kterm_i_strict_raw", wmask & t.pwi_strict.notna(), t.pwi_strict,
        source="pwi_strict | ever_strict (net worth INCLUDING home equity)",
        units="model units (10k USD, real 2015)",
        model="(not targeted; the model has no housing sector)",
        block="kappa_terminal", targeted=False)
    # Rebuild the incl.-home series in model units without disturbing the mask logic.
    out[-1]["num"] = np.where(np.asarray(out[-1]["den"]) > 0,
                              np.nan_to_num(t.pwi_strict.values, nan=0.0) / DOLLARS_PER_MODEL_UNIT,
                              0.0)
    return out


def psychic_centre(t):
    """
    `m_psychic`, the frozen centring constant for the psychic cost of college.

    The cost is `kappa_0 + kappa_theta*(log(theta) - m_psychic)`. Recentring is
    BEHAVIOURALLY NEUTRAL -- exactly the device the wage equation already uses, where
    `m_theta = 7.3486` is "centring only; offset exactly by lnw0" -- but it is what makes
    the two parameters separately identified. Raw, `log(theta)` has a mean of about 6.26
    and a standard deviation of about 0.035, so the columns [1, log theta] have a
    condition number near 180 and `kappa_0` has to move ~180 units to undo one unit of
    `kappa_theta`. Centred, the two are orthogonal: `kappa_0` is the psychic cost at mean
    ability and `kappa_theta` is the pure gradient.

    Frozen HERE, alongside the targets, rather than computed from the simulation: a
    constant that moved with the parameter vector would not be a reparameterisation, it
    would be a new nonlinearity, and the estimate would depend on the simulation draw.

    Measured on the SAME frame the tertiles are cut on -- age-17 last-CDS wave, completion
    follow-up -- so `kappa_0` is the psychic cost at the mean ability of the children whose
    completion rates identify it.
    """
    f = t[(t[TAS_FOLLOWUP] == 1) & (t.ach_age == TAS_ACH_AGE) & t.g_ACH.notna()]
    return float(np.log(f.g_ACH).mean()), len(f)


def git_sha():
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            text=True).strip()
    except Exception:
        return "unknown"


def main():
    m = pd.read_stata(MICRO)
    t = pd.read_stata(TAS_MICRO)          # the TAS block's own frame -- see TAS_MICRO
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
    # Own study is chosen. School is fixed and separately deducted from leisure.
    # Use the actual median-school variable, not c_time_hrs - study_hrs: the
    # total input also contains imputed observations with different coverage.
    r["i_share"] = r.study_hrs / HOURS_PER_WEEK
    school = r.groupby("Child_Age").school_hrs.mean().reindex(range(AGE_LO, AGE_HI+1))
    if school.isna().any() or not np.isfinite(school).all():
        raise ValueError("Missing fixed school schedule: school_hrs is required at every age")
    if (school.loc[1:5] != 0).any() or ((school < 0) | (school >= HOURS_PER_WEEK)).any():
        raise ValueError("Invalid fixed school schedule")
    school_share = school / HOURS_PER_WEEK
    # HC in LOGS. The model now carries HC in the data's own W-score units, so this is a
    # like-for-like comparison -- and it is what separates the VALUATION parameters
    # (phi_3, lambda_2) from the TECHNOLOGY parameters (R_0, sigma_1, sigma_2, sigma_4),
    # which are otherwise collinear: both raise investment, only technology raises HC.
    # Logs, because that is the form the production function uses and the SD is far more
    # stable in logs (0.083 at age 3 to 0.031 at 17).
    moments += [
        dict(name="mean_i_c_early",
             series=r[(r.Child_Age >= 6) & (r.Child_Age <= AGE_SPLIT)].i_share,
             source=f"study_hrs / 112 (own study only), child ages 6-{AGE_SPLIT}",
             units="share of the 112h non-sleep week",
             model=f"mean of sim_i over t = 6..{AGE_SPLIT}"),
        dict(name="mean_i_c_late",
             series=r[r.Child_Age > AGE_SPLIT].i_share,
             source=f"study_hrs / 112 (own study only), child ages {AGE_SPLIT+1}-{AGE_HI}",
             units="share of the 112h non-sleep week",
             model=f"mean of sim_i over t = {AGE_SPLIT+1}..{AGE_HI}"),
        dict(name="mean_hc_early",
             series=r[(r.Child_Age >= AGE_HC_LO) & (r.Child_Age <= AGE_SPLIT)].x_gach,
             source=f"x_gach (log PCA composite), child ages {AGE_HC_LO}-{AGE_SPLIT}",
             units="log W-score; the model's HC is in the SAME units after the rescaling",
             model=f"mean of log(sim_hc) over t = {AGE_HC_LO}..{AGE_SPLIT}"),
        dict(name="mean_hc_late",
             series=r[r.Child_Age >= AGE_HC_LATE_LO].x_gach,
             source=f"x_gach (log PCA composite), child ages {AGE_HC_LATE_LO}-{AGE_HI}",
             units="log W-score; the model's HC is in the SAME units after the rescaling",
             model=f"mean of log(sim_hc) over t = {AGE_HC_LATE_LO}..{AGE_HI}"),
    ]

    # The TAS block is computed HERE, before `lines` is assembled, because its scalar
    # metadata are TOP-LEVEL TOML keys. Any bare `key = value` emitted after the first
    # `[table]` header belongs to that table, not to the document -- so writing m_psychic
    # further down silently nested it inside the last moment's table and load_targets
    # could not find it.
    tas = build_tas_moments(t)
    m_psychic, n_psychic = psychic_centre(t)
    tas_est, tas_infl = {}, {}
    for mo in tas:
        est, infl = ratio_influence(mo["num"], mo["den"], t[TAS_CLUSTER_ON])
        tas_est[mo["name"]] = est
        tas_infl[mo["name"]] = infl
        mo["estimate"] = est
        mo["n_obs"] = int((np.asarray(mo["den"]) > 0).sum())
        mo["n_clusters"] = int(t.loc[np.asarray(mo["den"]) > 0, TAS_CLUSTER_ON].nunique())
    _wmask = (t[f"ever_{TAS_WEALTH_DEF}"] == 1) & t[TAS_WEALTH_VAR].notna()
    cut = float(np.percentile(t[TAS_WEALTH_VAR].where(_wmask).dropna(), TAS_WEALTH_WINSOR_P))

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
        f'child_time_spec = "{CHILD_TIME_SPEC}"',
        f'age_hc_early = [{AGE_HC_LO}, {AGE_SPLIT}]',
        f'age_hc_late = [{AGE_HC_LATE_LO}, {AGE_HI}]',
        'school_time = [' + ', '.join(f'{v:.17g}' for v in school_share) + ']',
        'school_time_source = "school_hrs: median within Year/Age, averaged across nonmissing rows at each age; divided by 112"',
        'school_time_role = "fixed time deducted from child leisure; HC production uses own study only"',

        f'age_split  = {AGE_SPLIT}   # early = {AGE_LO}..{AGE_SPLIT}, late = {AGE_SPLIT+1}..{AGE_HI}',
        f'dollars_per_model_unit = {DOLLARS_PER_MODEL_UNIT}',
        f'hours_per_week = {HOURS_PER_WEEK}',
        "",
        "# ---- TAS block metadata (TOP-LEVEL keys: they must precede every [table]) ----",
        f'tas_source      = "Input/SMM_TAS_Micro.dta"',
        f'tas_cluster_on  = "{TAS_CLUSTER_ON}"',
        f'tas_n_children  = {len(t)}',
        f'tas_n_clusters  = {int(t[TAS_CLUSTER_ON].nunique())}',
        f'tas_outcome     = "{TAS_OUTCOME}"',
        f'tas_followup    = "{TAS_FOLLOWUP}"',
        f'tas_ach_age     = {TAS_ACH_AGE}',
        f'tas_ach_tertile = "{TAS_ACH_TERT}"',
        f'tas_wealth_var  = "{TAS_WEALTH_VAR}"',
        f'tas_wealth_winsor_p = {TAS_WEALTH_WINSOR_P}',
        "# The winsorisation cut, IN MODEL UNITS, so the model side can apply the SAME",
        "# functional. The data moment is E[min(W, cut)], not E[W]; comparing it against a",
        "# plain simulated mean would be comparing two different estimators. In practice the",
        "# model never reaches the cut -- its asset grid tops out at a_max = 100 -- so the",
        "# minimum binds on no simulated household, which moments.jl checks and reports.",
        f'tas_wealth_winsor_cut = {cut / DOLLARS_PER_MODEL_UNIT:.17g}',
        "",
        "# Centring constant for the psychic cost of college:",
        "#     kappa_0 + kappa_theta*(log(theta) - m_psychic)",
        "# Behaviourally neutral -- the same device as m_theta in the wage equation -- but it",
        "# orthogonalises kappa_0 and kappa_theta, which are otherwise collinear at a",
        "# condition number near 180. FROZEN here: a centring that moved with the parameter",
        "# vector would be a new nonlinearity, not a reparameterisation.",
        f'm_psychic       = {m_psychic:.17g}',
        f'm_psychic_n     = {n_psychic}',
        f'm_psychic_source = "mean log g_ACH, ach_age=={TAS_ACH_AGE}, completion follow-up frame"',
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

    # =========================================================================
    # THE TAS BLOCK
    # =========================================================================
    lines += [
        "# -------------------------------------------------------------------------",
        "# TAS BLOCK -- child-level outcomes for the four kappa parameters.",
        "#",
        "# A DIFFERENT SAMPLE FRAME from everything above: one row per TAS-linked child",
        f"# ({len(t)} children, {t[TAS_CLUSTER_ON].nunique()} family clusters), against the parent",
        "# block's one row per child-YEAR. Unweighted, per the codebook. Every moment is a",
        "# ratio of means over the whole frame with the subgroup indicator in the",
        "# denominator, which is how the source do-file estimates them.",
        "#",
        "# Completion, not entry: the model's college path has no dropout, so enrolling IS",
        "# completing. Entry is written below as an untargeted diagnostic.",
        "",
    ]

    print()
    print(f"{'TAS moment':22s} {'estimate':>13s} {'N':>7s} {'clusters':>9s}   source")
    print("-" * 100)
    for mo in tas:
        tag = "" if mo["targeted"] else "   [untargeted]"
        lines += [
            f"[{mo['name']}]",
            f'block  = "{mo["block"]}"',
            f'source = "{mo["source"]}"',
            f'units  = "{mo["units"]}"',
            f'model  = "{mo["model"]}"',
            f'targeted = {"true" if mo["targeted"] else "false"}',
            f"n      = {mo['n_obs']}",
            f"n_clusters = {mo['n_clusters']}",
            f"mean   = {mo['estimate']:.17g}",
            # `sd` is written for interface compatibility with the parent-block entries,
            # which moments.jl reads generically. For a ratio moment it is the SD of the
            # subgroup's own values, NOT the moment's standard error -- the SE is in
            # [moment_cov] below and is the only thing inference may use.
            f"sd     = {np.std(np.asarray(mo['num'])[np.asarray(mo['den']) > 0], ddof=1):.10g}",
        ]
        if mo["note"]:
            lines.append(f'note   = "{mo["note"]}"')
        lines.append("")
        print(f"{mo['name']:22s} {mo['estimate']:13.6f} {mo['n_obs']:7d} "
              f"{mo['n_clusters']:9d}   {mo['source'][:44]}{tag}")

    # ---- joint clustered moment covariance, BOTH blocks ----------------------
    targeted = [mo for mo in moments if mo["name"] in TARGETED]
    influences = moment_covariance(targeted, r)
    influences.update({k: v for k, v in tas_infl.items() if k in TARGETED})
    names = [n for n in TARGETED]
    missing = [n for n in names if n not in influences]
    if missing:
        raise ValueError(f"no influence function for targeted moment(s): {missing}")
    Omega, n_cl, n_clusters = joint_covariance(influences, names)
    se = np.sqrt(np.diag(Omega))

    # The overlap is the reason this is one matrix and not two. Report it, so that a
    # future change to either frame shows up as a change in the number of shared clusters
    # rather than silently as a different weighting matrix.
    shared = set(r[CLUSTER_ON].unique()) & set(t[TAS_CLUSTER_ON].unique())
    print(f"\ncluster overlap: {len(shared)} families appear in BOTH blocks "
          f"({len(set(r[CLUSTER_ON].unique()))} parent, {len(set(t[TAS_CLUSTER_ON].unique()))} TAS, "
          f"{n_clusters} distinct in the joint system)")
    ip = [names.index(n) for n in PARENT_TARGETED]
    it_ = [names.index(n) for n in TAS_MOMENTS]
    cross = Omega[np.ix_(ip, it_)]
    # corr(i,j) = cov(i,j) / (se_i * se_j). This had a stray sqrt -- np.sqrt(np.outer(...))
    # -- which divides by sqrt(se_i*se_j) and, since these SEs are well below 1, made every
    # reported cross-block correlation about an order of magnitude too SMALL. The EXPORTED
    # `corr` matrix below was always correct; only this printed diagnostic, and the
    # documentation quoting it, were wrong.
    dsd = np.outer(se[ip], se[it_])
    xcorr = cross / np.where(dsd > 0, dsd, 1.0)
    print(f"cross-block correlations: min {xcorr.min():+.4f}  max {xcorr.max():+.4f}  "
          f"|corr|>0.05 in {int((np.abs(xcorr) > 0.05).sum())} of {xcorr.size} pairs")
    # WITHIN-block correlation is the number that actually bears on whether a diagonal
    # weighting matrix loses efficiency, and it is much larger than the cross-block one.
    # Reporting only the cross-block figure invited the conclusion that the moments are
    # nearly uncorrelated, which they are not.
    Corr_full = Omega / np.where(np.outer(se, se) > 0, np.outer(se, se), 1.0)
    within = Corr_full[np.triu_indices(len(names), 1)]
    blocks = np.zeros_like(Corr_full, dtype=bool)
    blocks[np.ix_(ip, it_)] = True
    blocks[np.ix_(it_, ip)] = True
    wmask = (~blocks)[np.triu_indices(len(names), 1)]
    print(f"WITHIN-block correlations: min {within[wmask].min():+.4f}  "
          f"max {within[wmask].max():+.4f}  -- these are what a diagonal weight ignores")
    D = np.diag(1.0 / np.where(se > 0, se, 1.0))
    Corr = D @ Omega @ D

    lines += [
        "# ---------------------------------------------------------------------------",
        "# Cluster-robust covariance of the TARGETED moment vector.",
        "#",
        f"# Clustered on {CLUSTER_ON} / {TAS_CLUSTER_ON} -- the SAME key, the PSID 1968 family",
        f"# interview number ({n_clusters} distinct families across both blocks, {len(shared)} of them",
        "# in both). `se` is the standard error of",
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
        "se         = [" + ", ".join(f"{v:.17g}" for v in se) + "]",
        "cov        = [",
    ]
    for i in range(len(names)):
        lines.append("  [" + ", ".join(f"{Omega[i, j]:.17g}" for j in range(len(names))) + "],")
    lines += ["]", "corr       = ["]
    for i in range(len(names)):
        lines.append("  [" + ", ".join(f"{Corr[i, j]:.6f}" for j in range(len(names))) + "],")
    lines += ["]", ""]

    print()
    print(f"{'moment':22s} {'se':>12s} {'sd':>12s}   se/sd   clusters")
    print("-" * 72)
    for j, nm in enumerate(names):
        if nm in PARENT_TARGETED:
            sd = next(mo["series"].dropna().std() for mo in targeted if mo["name"] == nm)
        else:
            mo = next(mo for mo in tas if mo["name"] == nm)
            sd = float(np.std(np.asarray(mo["num"])[np.asarray(mo["den"]) > 0], ddof=1))
        print(f"{nm:22s} {se[j]:12.6f} {sd:12.6f} {se[j]/sd:7.4f} {n_cl[nm]:10d}")
    off = Corr[np.triu_indices(len(names), 1)]
    print(f"\nmoment correlations: min {off.min():+.3f}  max {off.max():+.3f}  "
          f"|corr|>0.3 in {int((np.abs(off) > 0.3).sum())} of {len(off)} pairs")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("x") as target_file:  # immutable snapshot; never rewrite an older run
        target_file.write("\n".join(lines))
    print(f"\nwrote {OUT.relative_to(REPO)}")
    write_by_age()
    write_assets_by_age()


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
      x_gach, x_lw    mean LOG human capital in the same W-score units as the model.
    """
    # THIS STAGE IS OPTIONAL AND MUST NOT BLOCK THE TARGETS.
    #
    # The by-age CSVs are a plotting convenience for the notebook; the targets file above
    # is what the estimation reads. They were previously written in the same pass with no
    # guard, so when Input/ carried a .dta that lacked a column the script crashed AFTER
    # writing the targets -- leaving a non-zero exit on a run that had in fact succeeded at
    # its main job.
    #
    # The guard stays; the diagnosis that motivated it was WRONG and is corrected here so
    # nobody acts on it again. Re-checked 2026-09-06 against the .dta files in this
    # repository: SMM_Moments_ByAge.dta DOES have `mu_assets_real` (both by-age files
    # carry the full mu_/sd_/md_/wmu_ asset set), SMM_Moments_ByAge_Cohort.dta IS in
    # Input/, and regenerating both CSVs reproduces the committed ones byte for byte --
    # they are current, not stale. The crash came from an environment that did not have
    # the .dta files, because `Input/*.dta` was gitignored until this commit; a fresh
    # clone had no by-age inputs at all.
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
                               "mu_par_time_tot", "mu_c_time_hrs", "mu_study_hrs", "mu_school_hrs", "mu_x_gach", "mu_x_lw")
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
            "school_c": d.mu_school_hrs / HOURS_PER_WEEK,
            "i_total": d.mu_c_time_hrs / HOURS_PER_WEEK,
            # Same leisure identity: school, own study and parental time are
            # deducted. These are by-age means with variable-specific coverage.
            # Active+nearby parental time retains its pre-existing overlap caveat.
            "l_c": (HOURS_PER_WEEK - d.mu_study_hrs - d.mu_school_hrs - d.mu_par_time_tot) / HOURS_PER_WEEK,
            "x_gach": d.mu_x_gach,
            "x_lw":   d.mu_x_lw,
        })
        (REPO / "Input" / dst).write_text(
            f"# Per-child-age data means for the baseline figure. Sample: {note}.\n"
            f"# GENERATED by tools/make_smm_targets.py from {src} -- do not edit by hand.\n"
            "# c_p, e_p, a_p: model units (10k USD/yr). a_p EXCLUDES home equity.\n"
            "# h_p, t_p, i_c, school_c, i_total, l_c: shares of the 112h week.\n"
            "# i_c = own study; school_c = mean of the median-school variable by age.\n"
            "# i_total = legacy school-plus-study input (includes imputations).\n"
            "# l_c = 1 - t_p - i_c - school_c; untargeted, variable-specific coverage.\n"
            "# x_gach / x_lw: mean LOG human capital in W-score units; no shift required.\n"
            + out.to_csv(index=False, float_format="%.6f"))
        print(f"wrote Input/{dst}  ({len(out)} ages, {key})")



# =============================================================================
# Assets by child age
# =============================================================================
ASSETS_SRC = "SMM_Assets_ByChildAge.dta"
ASSETS_DST = "smm_assets_by_child_age.csv"


def write_assets_by_age():
    """
    Parental net worth by child age, in MODEL UNITS, for comparison against sim_a.

    TWO ASSET CONCEPTS, and the difference is not small -- at child age 17 the
    home-inclusive mean is roughly 1.5x the exclusive one. `a_p` is the one the model
    can be compared against: `assets_real` EXCLUDES home equity, for the same reason
    consumption uses cons_exhous_real. The model has no housing sector, no mortgage and
    no durable stock, so home equity has nothing to map onto. `a_p_home` is carried
    alongside so the choice stays visible rather than silently made.

    MEDIANS MATTER HERE MORE THAN USUAL. The mean is wildly skewed -- at child age 2 the
    SD is 949k against a mean of 117k -- so `md_` is the column to read for a typical
    household, and the p10-p90 spread says how little the mean represents anyone. The
    model's simulated distribution should be compared against the quantiles, not just
    the mean.

    Ages 0-30 are in the source; all of them are written. The parent block only runs
    t = 1..17, and the age-18 handoff column is where sim_a becomes the child's initial
    assets, so 18+ is the child block's comparison, not the parent's.
    """
    path = REPO / "Input" / ASSETS_SRC
    if not path.exists():
        print(f"  SKIP {ASSETS_DST}: {ASSETS_SRC} is not in Input/.")
        return
    d = pd.read_stata(path).sort_values("Child_Age")
    stats = ["n", "mu", "sd", "p10", "p25", "md", "p75", "p90"]
    missing = [f"{s}_assets_{k}" for k in ("real", "home") for s in stats
               if f"{s}_assets_{k}" not in d.columns]
    if missing:
        print(f"  SKIP {ASSETS_DST}: {ASSETS_SRC} is missing {', '.join(missing)}.")
        return

    out = pd.DataFrame({"child_age": d.Child_Age.astype(int),
                        "n": d.n_assets_real.astype(int)})
    # n is a count and stays a count; everything else is dollars -> model units.
    for k, suffix in (("real", ""), ("home", "_home")):
        for s in stats[1:]:
            out[f"a_p{suffix}_{s}"] = d[f"{s}_assets_{k}"] / DOLLARS_PER_MODEL_UNIT

    header = [
        "# Parental net worth by child age, in MODEL UNITS (10k USD).",
        f"# GENERATED by tools/make_smm_targets.py from {ASSETS_SRC} -- do not edit by hand.",
        "#",
        "# a_p_*       EXCLUDES home equity -- the concept the model can be compared to,",
        "#             for the same reason consumption uses cons_exhous_real.",
        "# a_p_home_*  INCLUDES it, carried so the choice stays visible.",
        "#",
        "# READ THE MEDIAN, NOT THE MEAN. The distribution is severely right-skewed",
        "# (at child age 2 the SD is 95 model units against a mean of 11.7), so a_p_md",
        "# describes a typical household and a_p_mu does not. Compare the model's",
        "# simulated spread against p10-p90, not against the mean alone.",
        "#",
        "# Ages 0-30 are all written; the parent block only covers t = 1..17 and age 18",
        "# is the handoff where sim_a becomes the child's initial assets.",
    ]
    (REPO / "Input" / ASSETS_DST).write_text(
        "\n".join(header) + "\n" + out.to_csv(index=False, float_format="%.6f"))
    print(f"wrote Input/{ASSETS_DST}  ({len(out)} child ages)")
    a17 = out.loc[out.child_age == 17]
    if not a17.empty:
        r = a17.iloc[0]
        print(f"  at child age 17: mean {r.a_p_mu:.2f} / median {r.a_p_md:.2f} model units "
              f"(${r.a_p_mu*DOLLARS_PER_MODEL_UNIT:,.0f} / ${r.a_p_md*DOLLARS_PER_MODEL_UNIT:,.0f})")

if __name__ == "__main__":
    main()
