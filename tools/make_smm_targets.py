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

    # Per-parent leisure share. leis_mom/leis_dad are already 112 - work - active.
    r["leis_share"] = ((r.leis_mom + r.leis_dad) / 2.0) / HOURS_PER_WEEK

    moments = [
        dict(name="mean_c_p",
             series=r.cons_exhous_real_w99 / DOLLARS_PER_MODEL_UNIT,
             source="cons_exhous_real_w99",
             units="model units (10k USD/yr, real 2015)",
             model="mean of sim_c over t = 1..17"),
        dict(name="mean_l_p",
             series=r.leis_share,
             source="(leis_mom + leis_dad)/2 / 112",
             units="share of the 112h non-sleep week, per parent",
             model="mean of 1 - sim_h - sim_t over t = 1..17"),
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


if __name__ == "__main__":
    main()
