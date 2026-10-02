#!/usr/bin/env python3
"""
Write the SMM target file from the Stata moment vector.

    python3 tools/make_smm_targets.py

Writes output/smm_runs/<timestamp>_targets/targets.toml. Julia never reads .dta or
the Stata CSVs: the targets are frozen into a small, readable file so a run is
reproducible and a change in targets shows up as a diff.

THIS SCRIPT NO LONGER COMPUTES MOMENTS (2026-10-01)
---------------------------------------------------
Every moment, its standard error, the joint covariance and every calibrated constant
are computed in Stata by `data_cleaning/Child_Time_Study/Code/28_smm_moments.do`
(memo 19 in that repository). That file replaces the old split in which the parent
block and the TAS block were computed here, the skill block in Stata step 27, and the
covariance was block-diagonal with zero cross-block terms. It runs ONE family-cluster
bootstrap over every frame, so the covariance written below has real cross-block terms.

This script only:
  1. reads Input/SMM_Moments.csv, Input/SMM_VCov.csv and Input/SMM_Constants.csv,
     which are copied by hand from Child_Time_Study/Output/Data/SMM/ as before, and
     records their sha256;
  2. validates them: unique names, finite estimates, positive SEs, a symmetric
     positive semi-definite covariance whose order equals the moment order and whose
     diagonal reproduces the SEs, and a parameter and model counterpart for every
     targeted row;
  3. reads the fixed school schedule from Input/SMM_Moments_Micro.dta, exactly as
     before;
  4. writes targets.toml in the established layout.

CONVENTIONS THAT STILL HOLD
---------------------------
Scale. One model unit is 10,000 US dollars per year (ASSET_RESCALE = 10 in tables.jl).

Time. A time moment is hours per week / 112, per parent (the model's single adult
stands for two earners: wage_func x2).

Consumption INCLUDES housing and EXCLUDES the money investment (user 2026-10-01).
Sahber's PSID Consumption already contains tuition + other school + childcare, which
is e_p, so the old target (consumption excluding housing) counted e_p twice in the
budget c_p + e_p. Housing (rent, or 6% of the house value for owners) is now inside
c_p; the model has no housing sector, so it is read as part of non-investment spending.

WHAT CHANGED FOR JULIA
----------------------
The moment names, the skill measure (raw Letter-Word, DFVW binomial, memo 18), the
college outcome (BA at the first TAS wave at 25/26), the wealth target (median net worth
incl. home at first-child ages 21-22, matched to a_term) and the child weight mu_t (a
logistic in age from the autonomy index) all changed. code/smm/moments.jl checks the
moment names and refuses this file until its simulated counterparts are updated; that is
intended. Older run folders are untouched.
"""

import hashlib
import subprocess
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
MICRO = REPO / "Input" / "SMM_Moments_Micro.dta"
MOMENTS_CSV = REPO / "Input" / "SMM_Moments.csv"
VCOV_CSV = REPO / "Input" / "SMM_VCov.csv"
CONST_CSV = REPO / "Input" / "SMM_Constants.csv"
OUT = REPO / "output" / "smm_runs" / (datetime.now().strftime("%Y-%m-%d_%H%M%S_%f") + "_targets") / "targets.toml"
# One by-age file per SAMPLE, both plotted (see write_by_age).
BY_AGE_SOURCES = {
    "all":    ("SMM_Moments_ByAge.dta",        "smm_moments_by_age.csv",
               "all one-child families, no parental-age restriction"),
    "cohort": ("SMM_Moments_ByAge_Cohort.dta", "smm_moments_by_age_cohort.csv",
               "cohort_dad2530: father aged 25-30 at the child's birth"),
}

DOLLARS_PER_MODEL_UNIT = 10_000.0   # ASSET_RESCALE = 10, in thousands
HOURS_PER_WEEK = 112.0              # 168 less a 56-hour sleep allowance
AGE_LO, AGE_HI = 1, 17              # the parent block's t = 1..17
CHILD_TIME_SPEC = "own_study_fixed_school_v1"

MOMENT_COLUMNS = ["order", "moment", "block", "targeted", "parameter", "estimate", "se",
                  "n_obs", "n_clusters", "units", "model", "measure"]
TARGET_BLOCKS = {"P", "S", "T", "W"}
# The child weight is a calibrated per-age vector: ages 6..17 are the bargaining periods,
# mu_half is the age-18 half period. Parent weight = 1 - mu_t (1 before age 6).
MU_AGES = list(range(6, 18))


def git_sha():
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            text=True).strip()
    except Exception:
        return "unknown"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def toml_str(s):
    s = "" if pd.isna(s) else str(s)
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def read_stata_vector():
    """Read and validate the three Stata files. Raises on any inconsistency."""
    for p in (MOMENTS_CSV, VCOV_CSV, CONST_CSV):
        if not p.exists():
            raise FileNotFoundError(
                f"{p.relative_to(REPO)} is missing: copy it from "
                "Child_Time_Study/Output/Data/SMM/ (written by 28_smm_moments.do)")
    m = pd.read_csv(MOMENTS_CSV)
    v = pd.read_csv(VCOV_CSV)
    k = pd.read_csv(CONST_CSV)

    missing = [c for c in MOMENT_COLUMNS if c not in m.columns]
    if missing:
        raise ValueError(f"SMM_Moments.csv lacks columns {missing}")
    m = m.sort_values("order").reset_index(drop=True)
    n = len(m)
    if list(m["order"]) != list(range(1, n + 1)):
        raise ValueError("SMM_Moments.csv: `order` is not 1..N")
    if m["moment"].duplicated().any():
        raise ValueError(f"duplicate moment names: {m.loc[m.moment.duplicated(), 'moment'].tolist()}")
    if not np.isfinite(m["estimate"]).all():
        raise ValueError("non-finite estimate(s): " + ", ".join(m.loc[~np.isfinite(m.estimate), "moment"]))
    if not (m["se"] > 0).all():
        raise ValueError("non-positive SE(s): " + ", ".join(m.loc[~(m.se > 0), "moment"]))
    t = m[m.targeted == 1]
    bad = t[~t.block.isin(TARGET_BLOCKS)]
    if len(bad):
        raise ValueError(f"targeted rows outside blocks {TARGET_BLOCKS}: {bad.moment.tolist()}")
    if (m.loc[m.block == "X", "targeted"] != 0).any():
        raise ValueError("a diagnostic (block X) row is flagged targeted")
    for col in ("parameter", "model"):
        empty = t[t[col].isna() | (t[col].astype(str).str.strip().isin(["", "-"]))]
        if len(empty):
            raise ValueError(f"targeted rows without a {col}: {empty.moment.tolist()}")

    if "order" in v.columns:
        if list(v["order"]) != list(range(1, n + 1)):
            raise ValueError("SMM_VCov.csv row order is not 1..N")
        v = v.drop(columns="order")
    V = v.to_numpy(dtype=float)
    if V.shape != (n, n):
        raise ValueError(f"SMM_VCov is {V.shape}, expected {(n, n)}")
    if not np.allclose(V, V.T, rtol=1e-10, atol=1e-14):
        raise ValueError("SMM_VCov is not symmetric")
    eig = np.linalg.eigvalsh(V)
    if eig.min() < -1e-10 * max(1.0, eig.max()):
        raise ValueError(f"SMM_VCov is not positive semi-definite (min eigenvalue {eig.min():.3g})")
    if not np.allclose(np.sqrt(np.diag(V)), m["se"].to_numpy(), rtol=1e-6):
        raise ValueError("sqrt(diag(SMM_VCov)) does not reproduce the SE column")

    for c in ("name", "value"):
        if c not in k.columns:
            raise ValueError(f"SMM_Constants.csv lacks column {c}")
    need = ["m_psychic", "mu_half"] + [f"mu_age{a}" for a in MU_AGES]
    absent = [c for c in need if c not in set(k.name)]
    if absent:
        raise ValueError(f"SMM_Constants.csv lacks {absent}")
    return m, V, k


def school_schedule():
    """The fixed school schedule by child age, from the micro file (unchanged rule)."""
    micro = pd.read_stata(MICRO)
    r = micro[(micro.Child_Age >= AGE_LO) & (micro.Child_Age <= AGE_HI)]
    school = r.groupby("Child_Age").school_hrs.mean().reindex(range(AGE_LO, AGE_HI + 1))
    if school.isna().any() or not np.isfinite(school).all():
        raise ValueError("Missing fixed school schedule: school_hrs is required at every age")
    if (school.loc[1:5] != 0).any() or ((school < 0) | (school >= HOURS_PER_WEEK)).any():
        raise ValueError("Invalid fixed school schedule")
    return school / HOURS_PER_WEEK


def main():
    m, V, k = read_stata_vector()
    school_share = school_schedule()
    const = dict(zip(k.name, k.value))
    const_se = dict(zip(k.name, k.get("se", pd.Series([np.nan] * len(k)))))
    const_desc = dict(zip(k.name, k.get("desc", pd.Series([""] * len(k)))))

    tgt = m.index[m.targeted == 1].to_numpy()
    names = m.loc[tgt, "moment"].tolist()
    Omega = V[np.ix_(tgt, tgt)]
    se = np.sqrt(np.diag(Omega))
    Corr = Omega / np.outer(se, se)

    lines = [
        "# SMM targets. GENERATED by tools/make_smm_targets.py from the Stata moment vector",
        "# of Child_Time_Study/Code/28_smm_moments.do. Do not edit by hand -- rerun both.",
        "#",
        "# One model unit = 10,000 USD/yr. Time is a share of the 112h week, per parent.",
        "",
        f'generated  = "{date.today().isoformat()}"',
        f'git_commit = "{git_sha()}"',
        'source     = "Child_Time_Study 28_smm_moments.do via Input/SMM_Moments.csv"',
        f'sha256_moments   = "{sha256(MOMENTS_CSV)}"',
        f'sha256_vcov      = "{sha256(VCOV_CSV)}"',
        f'sha256_constants = "{sha256(CONST_CSV)}"',
        f'age_range  = [{AGE_LO}, {AGE_HI}]',
        f'child_time_spec = "{CHILD_TIME_SPEC}"',
        'school_time = [' + ', '.join(f'{v:.17g}' for v in school_share) + ']',
        'school_time_source = "school_hrs: median within Year/Age, averaged across nonmissing rows at each age; divided by 112"',
        'school_time_role = "fixed time deducted from child leisure; HC production uses own study only"',
        f'dollars_per_model_unit = {DOLLARS_PER_MODEL_UNIT}',
        f'hours_per_week = {HOURS_PER_WEEK}',
        f'n_targeted   = {len(names)}',
        f'n_diagnostic = {int((m.targeted == 0).sum())}',
        "",
        "# ---- calibrated constants (TOP-LEVEL keys: they must precede every [table]) ----",
        "# Psychic cost of college: kappa_0 + kappa_theta*(log theta - m_psychic), theta in",
        "# DFVW ln k units; m_psychic = latent mean ln k at 17 on the college frame.",
        f'm_psychic        = {const["m_psychic"]:.17g}',
        f'm_psychic_se     = {const_se.get("m_psychic", float("nan")):.17g}',
        f'm_psychic_source = {toml_str(const_desc.get("m_psychic", ""))}',
        f'sd_lnk17         = {const.get("sd_lnk17", float("nan")):.17g}',
        f'mean_lw17        = {const.get("mean_lw17", float("nan")):.17g}',
        f'L0               = {const.get("L0", float("nan")):.17g}   # DFVW location normalisation, L1 = 1',
        "# Initial skill distribution at age 1: ln k_1 = m_0 + m_BC*BothCollege + s_0*z",
        f'init_m0  = {const["init_m0_baseline"]:.17g}',
        f'init_mBC = {const["init_mBC_baseline"]:.17g}',
        f'init_s0  = {const["init_s0_baseline"]:.17g}',
        f'init_m0_alt  = {const["init_m0_alt"]:.17g}',
        f'init_mBC_alt = {const["init_mBC_alt"]:.17g}',
        f'init_s0_alt  = {const["init_s0_alt"]:.17g}',
        "# Child bargaining weight mu_t (the CHILD's weight; the parent's is 1 - mu_t for",
        "# t >= 6 and 1 before 6). Logistic in age fitted to the Overall autonomy index;",
        "# mu_half is the weight at the age-18 half period (index at 18).",
        f'mu_ages  = [{", ".join(str(a) for a in MU_AGES)}]',
        'mu_by_age = [' + ', '.join(f'{const[f"mu_age{a}"]:.17g}' for a in MU_AGES) + ']',
        f'mu_half  = {const["mu_half"]:.17g}',
        f'cw_L  = {const.get("cw_L", float("nan")):.17g}',
        f'cw_U  = {const.get("cw_U", float("nan")):.17g}',
        f'cw_k  = {const.get("cw_k", float("nan")):.17g}',
        f'cw_t0 = {const.get("cw_t0", float("nan")):.17g}',
        'mu_mapping = "parent weight = 1 - mu_t for t >= 6, 1 before 6; mu_half at the half period"',
        "",
    ]
    print(f"{'moment':28s} {'block':5s} {'tgt':>3s} {'estimate':>13s} {'se':>11s} {'N':>6s}")
    print("-" * 72)
    for _, row in m.iterrows():
        lines += [
            f"[{row.moment}]",
            f"block     = {toml_str(row.block)}",
            f"parameter = {toml_str(row.parameter)}",
            f"measure   = {toml_str(row.measure)}",
            f"units     = {toml_str(row.units)}",
            f"model     = {toml_str(row.model)}",
            f"targeted  = {'true' if row.targeted == 1 else 'false'}",
            f"n         = {int(row.n_obs)}",
            f"n_clusters = {int(row.n_clusters)}",
            f"mean      = {row.estimate:.17g}",
            f"se        = {row.se:.17g}",
            "",
        ]
        print(f"{row.moment:28s} {row.block:5s} {int(row.targeted):3d} {row.estimate:13.6f} "
              f"{row.se:11.6f} {int(row.n_obs):6d}")

    lines += [
        "# ---------------------------------------------------------------------------",
        "# Family-cluster bootstrap covariance of the TARGETED moments (Stata 28, one",
        "# resampling of 1968 families across every frame, so cross-block terms are real).",
        "# `cov` is row-major over `names`; `corr` is the same matrix with unit diagonal.",
        "[moment_cov]",
        'method     = "family-cluster bootstrap over all frames (28_smm_moments.do)"',
        "names      = [" + ", ".join(f'"{n}"' for n in names) + "]",
        "n_clusters_by_moment = [" + ", ".join(str(int(x)) for x in m.loc[tgt, "n_clusters"]) + "]",
        "se         = [" + ", ".join(f"{v:.17g}" for v in se) + "]",
        "cov        = [",
    ]
    for i in range(len(names)):
        lines.append("  [" + ", ".join(f"{Omega[i, j]:.17g}" for j in range(len(names))) + "],")
    lines += ["]", "corr       = ["]
    for i in range(len(names)):
        lines.append("  [" + ", ".join(f"{Corr[i, j]:.6f}" for j in range(len(names))) + "],")
    lines += ["]", ""]

    off = Corr[np.triu_indices(len(names), 1)]
    print(f"\n{len(names)} targeted moments, {int((m.targeted == 0).sum())} diagnostics; "
          f"moment correlations: min {off.min():+.3f}  max {off.max():+.3f}")

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
                               "mu_leis_mom_wk", "mu_leis_dad_wk",
                               "mu_par_time_tot", "mu_c_time_hrs", "mu_study_hrs", "mu_school_hrs", "mu_x_gach", "mu_x_lw")
                   if c not in d.columns]
        if missing:
            print(f"  SKIP {dst}: {src} is missing {', '.join(missing)}. "
                  f"The committed CSV is left as it is and is STALE.")
            continue
        d = d[(d.Child_Age >= AGE_LO) & (d.Child_Age <= AGE_HI)].sort_values("Child_Age")
        # ASSETS COME FROM THE BINNED FILE since 2026-09-11. The Stata rerun dropped the
        # twelve `*_assets_*` columns from the by-age files and moved parental net worth
        # to SMM_Assets_ByChildAge.dta in TWO-YEAR bins (`age_bin` = lower edge). The
        # by-age CSV keeps its `a_p` column -- the notebook reads it -- filled with the
        # mean of the bin that contains each age, so ages 2k and 2k+1 share a value.
        # Same sample for both by-age files (the assets file is not cohort-split).
        a_p = assets_by_age_from_bins(d.Child_Age.astype(int).values)
        out = pd.DataFrame({
            "child_age": d.Child_Age.astype(int),
            "c_p": d.mu_cons_exhous_real_w99 / DOLLARS_PER_MODEL_UNIT,
            "e_p": d.mu_m_method2_final_w99 / DOLLARS_PER_MODEL_UNIT,
            # Net worth EXCLUDING home equity. The model has no housing sector, no
            # mortgage and no durable stock, so home equity has nothing to map onto --
            # the same reason consumption uses cons_exhous_real.
            "a_p": a_p,
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
            "# c_p, e_p, a_p: model units (10k USD/yr). a_p EXCLUDES home equity and is the\n"
            "# mean of the TWO-YEAR age bin containing each age (SMM_Assets_ByChildAge.dta).\n"
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


def _read_assets_bins():
    """The binned assets file, or None. `age_bin` is the LOWER EDGE of a two-year bin."""
    path = REPO / "Input" / ASSETS_SRC
    if not path.exists():
        return None
    d = pd.read_stata(path)
    if "age_bin" not in d.columns:
        # pre-2026-09-11 layout: one row per single year of child age
        d = d.rename(columns={"Child_Age": "age_bin"})
        d["bin_width"] = 1
    else:
        d["bin_width"] = 2
    return d.sort_values("age_bin").reset_index(drop=True)


def assets_by_age_from_bins(ages):
    """Mean net worth (excl. home, model units) for each single age, from its bin."""
    d = _read_assets_bins()
    if d is None or "mu_assets_real" not in d.columns:
        return np.full(len(ages), np.nan)
    lo = d.age_bin.astype(int).values; w = d.bin_width.values
    out = np.full(len(ages), np.nan)
    for i, a in enumerate(ages):
        j = np.where((lo <= a) & (a < lo + w))[0]
        if len(j):
            out[i] = d.mu_assets_real.values[j[0]] / DOLLARS_PER_MODEL_UNIT
    return out


def write_assets_by_age():
    """
    Parental net worth by child age, in MODEL UNITS, for comparison against sim_a.

    TWO ASSET CONCEPTS, and the difference is not small -- at child age 17 the
    home-inclusive mean is roughly 1.5x the exclusive one. `a_p` is the one the model
    can be compared against: `assets_real` EXCLUDES home equity, for the same reason
    consumption uses cons_exhous_real. The model has no housing sector, no mortgage and
    no durable stock, so home equity has nothing to map onto. `a_p_home` is carried
    alongside so the choice stays visible rather than silently made.

    MEDIANS MATTER HERE MORE THAN USUAL. The mean is wildly skewed -- at child ages 2-3 the
    SD is 71 model units against a mean of 9.2 -- so `md_` is the column to read for a
    typical household, and the p10-p90 spread says how little the mean represents anyone.

    TWO-YEAR BINS since the 2026-09-11 Stata rerun: `age_bin` is the lower edge, so the
    row for bin 16 covers child ages 16 and 17 and bin 18 covers 18-19. `age_lo` /
    `age_hi` make that explicit; `child_age` is kept equal to `age_lo` for readers of the
    old single-year layout. The parent block only runs t = 1..17; the age-18 handoff is
    where sim_a becomes the child's initial assets.
    """
    d = _read_assets_bins()
    if d is None:
        print(f"  SKIP {ASSETS_DST}: {ASSETS_SRC} is not in Input/.")
        return
    stats = ["n", "mu", "sd", "p10", "p25", "md", "p75", "p90"]
    missing = [f"{s}_assets_{k}" for k in ("real", "home") for s in stats
               if f"{s}_assets_{k}" not in d.columns]
    if missing:
        print(f"  SKIP {ASSETS_DST}: {ASSETS_SRC} is missing {', '.join(missing)}.")
        return

    lo = d.age_bin.astype(int)
    out = pd.DataFrame({"child_age": lo,
                        "age_lo": lo,
                        "age_hi": lo + d.bin_width.astype(int) - 1,
                        "n": d.n_assets_real.astype(int)})
    # n is a count and stays a count; everything else is dollars -> model units.
    for k, suffix in (("real", ""), ("home", "_home")):
        for s in stats[1:]:
            out[f"a_p{suffix}_{s}"] = d[f"{s}_assets_{k}"] / DOLLARS_PER_MODEL_UNIT

    header = [
        "# Parental net worth by child age, in MODEL UNITS (10k USD).",
        f"# GENERATED by tools/make_smm_targets.py from {ASSETS_SRC} -- do not edit by hand.",
        "#",
        "# ONE ROW PER TWO-YEAR BIN of child age: [age_lo, age_hi]. child_age == age_lo.",
        "# a_p_*       EXCLUDES home equity -- the concept the model can be compared to,",
        "#             for the same reason consumption uses cons_exhous_real.",
        "# a_p_home_*  INCLUDES it, carried so the choice stays visible.",
        "#",
        "# READ THE MEDIAN, NOT THE MEAN. The distribution is severely right-skewed, so",
        "# a_p_md describes a typical household and a_p_mu does not. Compare the model's",
        "# simulated spread against p10-p90, not against the mean alone.",
        "#",
        "# The parent block only covers t = 1..17 and age 18 is the handoff where sim_a",
        "# becomes the child's initial assets.",
    ]
    (REPO / "Input" / ASSETS_DST).write_text(
        "\n".join(header) + "\n" + out.to_csv(index=False, float_format="%.6f"))
    print(f"wrote Input/{ASSETS_DST}  ({len(out)} age bins)")
    a17 = out.loc[(out.age_lo <= 17) & (17 <= out.age_hi)]
    if not a17.empty:
        r = a17.iloc[0]
        print(f"  bin containing child age 17 ({int(r.age_lo)}-{int(r.age_hi)}): mean {r.a_p_mu:.2f} / "
              f"median {r.a_p_md:.2f} model units "
              f"(${r.a_p_mu*DOLLARS_PER_MODEL_UNIT:,.0f} / ${r.a_p_md*DOLLARS_PER_MODEL_UNIT:,.0f})")

if __name__ == "__main__":
    main()
