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
OUT_BY_AGE = REPO / "Input" / "smm_moments_by_age.csv"

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

    print(f"{'moment':12s} {'N':>7s} {'mean':>10s} {'sd':>10s}   source")
    print("-" * 72)
    for mo in moments:
        s = mo["series"].dropna()
        lines += [
            f"[{mo['name']}]",
            f'source = "{mo["source"]}"',
            f'units  = "{mo["units"]}"',
            f'model  = "{mo["model"]}"',
            f"n      = {len(s)}",
            f"mean   = {s.mean():.6f}",
            # SD is recorded but NOT targeted in the baseline run: the model's only
            # cross-sectional heterogeneity is a 5-node wage shock plus initial asset,
            # HC and college draws, and it cannot reach the data's dispersion (leisure
            # SD is 7.4x too small). Kept here so a later run can weight it in
            # deliberately rather than rediscover the number.
            f"sd     = {s.std():.6f}",
            f"median = {s.median():.6f}",
            "",
        ]
        print(f"{mo['name']:12s} {len(s):7d} {s.mean():10.4f} {s.std():10.4f}   {mo['source']}")

    OUT.write_text("\n".join(lines))
    print(f"\nwrote {OUT.relative_to(REPO)}")
    write_by_age()


def write_by_age():
    """
    Per-child-age means, for overlaying the data on the baseline figure.

    A SECOND file, and a CSV rather than TOML, because it is tabular and read by the
    notebook rather than by the estimation. Julia has no Stata reader in this project's
    Project.toml, so the same rule as the targets file applies: Python touches the .dta,
    Julia reads a small tracked text file. `DelimitedFiles.readdlm` is stdlib, so the
    notebook needs no new dependency.

    Units match the model's, so the notebook can plot these directly:
      c_p, e_p   model units (10k USD/yr)      t_p, h_p, i_c   share of the 112h week
      hc         RAW PCA W-score -- NOT comparable in level to the model's human capital,
                 which is in model units. Plot the shape only, indexed to the model.
    """
    d = pd.read_stata(REPO / "Input" / "SMM_Moments_ByAge.dta")
    d = d[(d.Child_Age >= AGE_LO) & (d.Child_Age <= AGE_HI)].sort_values("Child_Age")
    out = pd.DataFrame({
        "child_age": d.Child_Age.astype(int),
        "c_p": d.mu_cons_exhous_real_w99 / DOLLARS_PER_MODEL_UNIT,
        "e_p": d.mu_m_method2_final_w99 / DOLLARS_PER_MODEL_UNIT,
        # work is not stored directly by age; leis_*_wk IS 112 - own work, so invert it
        "h_p": (((HOURS_PER_WEEK - d.mu_leis_mom_wk) +
                 (HOURS_PER_WEEK - d.mu_leis_dad_wk)) / 2.0) / HOURS_PER_WEEK,
        "t_p": d.mu_par_time_tot / HOURS_PER_WEEK,
        "i_c": d.mu_study_hrs / HOURS_PER_WEEK,
        # LOG human capital, not the raw W-score. The model's HC and the data's are the
        # same object up to a scale factor M that was never applied (see ERRORS.md), and in
        # LOGS a scale factor is a pure ADDITIVE shift -- so log growth is directly
        # comparable even though the levels are not. x_gach is the log PCA composite,
        # x_lw the log Letter-Word score; both are "the x of the production function".
        "x_gach": d.mu_x_gach,
        "x_lw":   d.mu_x_lw,
    })
    OUT_BY_AGE.write_text(
        "# Per-child-age data means for the baseline figure. GENERATED by\n"
        "# tools/make_smm_targets.py -- do not edit by hand.\n"
        "# c_p, e_p: model units (10k USD/yr). h_p, t_p, i_c: share of the 112h week.\n"
        "# x_gach / x_lw: LOG human capital (PCA composite / Letter-Word). Comparable to\n"
        "#   log(model hc) up to an ADDITIVE constant log(M), the unapplied HC rescaling.\n"
        + out.to_csv(index=False, float_format="%.6f"))
    print(f"wrote {OUT_BY_AGE.relative_to(REPO)}  ({len(out)} ages)")


if __name__ == "__main__":
    main()
