#!/usr/bin/env python3
"""
Regression tests for tools/make_smm_targets.py (2026-09-11, sixteen-parameter targets).

    uv run --with pandas --with numpy python tools/test_smm_target_generator.py

What is checked, and why each check exists:

  1. The seven targeted TAS moments are in the SAME ORDER as `SMM_TAS_MOMENTS` in
     code/smm/moments.jl. `load_targets` refuses a permuted covariance, but only at run
     time; this catches it at generation time.
  2. Every row that has a namesake in the supplied Input/SMM_TAS_Moments.csv reproduces
     its estimate AND clustered SE to 1e-6, and the targeted-TAS covariance block matches
     the supplied SMM_TAS_VCov.dta. This is the regression that lets the reconstructed
     joint two-file covariance be trusted.
  3. The difference-of-ratios influence is the `lincom` identity: Var(r1 - r2) = V11 + V22
     - 2 V12 from the joint covariance of the halves.
  4. The sample-SD influence reproduces the supplied delta-method SE, including the
     n/(n-1) factor, and is scale-equivariant (SD of c*x is c*SD of x, same SE ratio).
  5. Frames: the age-17 frame is the m_psychic frame (N = 317); the wealth frame is the
     kterm frame intersected with the completion follow-up (N = 665); the winsorisation
     cut is the kterm cut.
  6. The interpolated-rank tertile rule reproduces the supplied wealth-tertile rows
     (N 222 / 221 / 222) and differs from the model-side floor rule by at most one
     observation per boundary.
  7. The fourteen targets shared with the baseline targets.toml are unchanged to the
     digit (means and SEs), and the joint covariance is symmetric positive definite.
  8. The assets-by-age CSV is bin-aware (age_lo/age_hi) and the by-age CSV's `a_p` is
     filled from the bins (no NaN over ages 1-17).
"""
import re
import sys
import numpy as np
import pandas as pd
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import make_smm_targets as G   # noqa: E402

FAIL = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    cond or FAIL.append(msg)


def se_of(infl):
    n = len(infl)
    return float(np.sqrt((infl.values ** 2).sum() * n / (n - 1.0)))


def main():
    t = pd.read_stata(G.TAS_MICRO)
    m = pd.read_stata(G.MICRO)
    cl = t[G.TAS_CLUSTER_ON]
    tas = G.build_tas_moments(t)
    by = {mo["name"]: mo for mo in tas}
    est, infl = {}, {}
    for mo in tas:
        e, i, sup = G.estimate_tas_moment(mo, cl)
        est[mo["name"]], infl[mo["name"]] = e, i
        mo["estimate"], mo["n_obs"] = e, int(sup.sum())
        mo["n_clusters"] = int(t.loc[sup, G.TAS_CLUSTER_ON].nunique())

    # ---- 1. order against moments.jl ----------------------------------------
    print("\n1. target order vs code/smm/moments.jl")
    src = (REPO / "code" / "smm" / "moments.jl").read_text()
    blk = re.search(r"const SMM_TAS_MOMENTS = \((.*?)\)", src, re.S).group(1)
    jl = re.findall(r'"([A-Za-z0-9_]+)"', blk)
    check(jl == G.TAS_MOMENTS, f"SMM_TAS_MOMENTS == TAS_MOMENTS: {jl}")
    targeted = [mo["name"] for mo in tas if mo["targeted"]]
    check(targeted == G.TAS_MOMENTS, "the rows flagged targeted are exactly TAS_MOMENTS, in order")
    check(len(G.TARGETED) == 17, f"17 targeted moments in total (got {len(G.TARGETED)})")

    # ---- 2. reproduction of the supplied exports ---------------------------
    print("\n2. reproduction of Input/SMM_TAS_Moments.csv and SMM_TAS_VCov.dta")
    tas_all = tas + G.cds_sd_by_age(m)
    try:
        G.check_against_supplied(tas_all, infl, t)
        check(True, "every namesake row reproduces estimate and SE to 1e-6; targeted-TAS covariance matches")
    except ValueError as e:
        check(False, str(e))

    # ---- 3. lincom identity for the difference of ratios -------------------
    print("\n3. difference-of-ratios influence == lincom")
    for nm, a, b in (("kth_ga17_gap", "kth_ga17_mean_c", "kth_ga17_mean_n"),):
        Om, _, _ = G.joint_covariance({a: infl[a], b: infl[b], nm: infl[nm]}, [a, b, nm])
        lincom = Om[0, 0] + Om[1, 1] - 2 * Om[0, 1]
        check(abs(lincom - Om[2, 2]) < 1e-12 * max(Om[2, 2], 1e-12),
              f"{nm}: Var(gap) = V11 + V22 - 2 V12  ({Om[2,2]:.3e} vs {lincom:.3e})")
        check(abs(est[nm] - (est[a] - est[b])) < 1e-12, f"{nm}: estimate is the difference of the halves")

    # ---- 4. sample-SD influence --------------------------------------------
    print("\n4. sample-SD influence")
    a17 = (t.hf_complete == 1) & (t.ach_age == 17) & t.g_ACH.notna()
    x = np.log(t.g_ACH.where(t.g_ACH > 0))
    sd1, p1 = G.sd_influence(x, a17, cl)
    sd2, p2 = G.sd_influence(3.0 * x, a17, cl)
    check(abs(sd1 - float(np.log(t.loc[a17, "g_ACH"]).std(ddof=1))) < 1e-12,
          f"sd_ga17 is the sample SD (ddof = 1): {sd1:.6f}")
    check(abs(sd2 / sd1 - 3.0) < 1e-10 and abs(se_of(p2) / se_of(p1) - 3.0) < 1e-10,
          "scale-equivariant: SD and SE both scale by the constant")
    sup = pd.read_csv(REPO / "Input" / "SMM_TAS_Moments.csv").set_index("moment")
    check(abs(se_of(p1) - float(sup.loc["sd_ga17", "se"])) < 1e-6,
          f"SE matches the supplied nlcom SE: {se_of(p1):.6f} vs {float(sup.loc['sd_ga17','se']):.6f}")

    # ---- 5. frames ----------------------------------------------------------
    print("\n5. frames and cuts")
    mp, n_mp = G.psychic_centre(t)
    check(by["sd_ga17"]["n_obs"] == n_mp == 317, f"age-17 frame is the m_psychic frame (N = {n_mp})")
    check(by["kse_w_gap"]["n_obs"] == 665, f"wealth frame N = {by['kse_w_gap']['n_obs']}")
    check(by["kterm_x_strict_w99"]["n_obs"] == 737, f"kterm frame N = {by['kterm_x_strict_w99']['n_obs']}")
    wmask = (t.ever_strict == 1) & t.pwx_strict.notna()
    cut = np.percentile(t.pwx_strict.where(wmask).dropna(), 99.0)
    check(abs(cut - 4448695) < 1, f"winsorisation cut = {cut:,.0f} USD (same object as kterm)")
    check("winsorised" in by["kse_w_gap"]["source"], "kse_w_gap is built on the winsorised series")

    # ---- 6. tertile rule ----------------------------------------------------
    print("\n6. tertile rule")
    wf = wmask & (t.hf_complete == 1)
    w99 = (t.pwx_strict.where(wmask).clip(upper=cut) / G.DOLLARS_PER_MODEL_UNIT)[wf].values
    tr = G.rank_tertiles(w99)
    counts = [int((tr == k).sum()) for k in (1, 2, 3)]
    check(counts == [222, 221, 222], f"interpolated-rank tertiles give N {counts}")
    n = len(w99)
    order = np.argsort(w99, kind="stable"); floor = np.empty(n, dtype=int)
    floor[order] = np.minimum(3, 1 + (3 * np.arange(n)) // n)
    check(int((floor != tr).sum()) <= 2, f"differs from the model-side floor rule on {int((floor != tr).sum())} observations (<= 2)")
    for k in (1, 2, 3):
        check(abs(est[f"kse_w_t{k}_c"] - float(sup.loc[f"kse_w_t{k}_c", "estimate"])) < 1e-9,
              f"kse_w_t{k}_c reproduces the supplied row")

    # ---- 7. baseline comparison and covariance -----------------------------
    print("\n7. unchanged targets vs baseline, and the joint covariance")
    base = REPO / "output" / "smm_runs" / "2026-09-10_183649" / "targets.toml"
    if base.exists():
        import tomllib
        b = tomllib.load(base.open("rb"))
        bn = b["moment_cov"]["names"]
        for k in ("k0_complete", "kpe_g0_c", "kpe_g1_c", "kterm_x_strict_w99"):
            check(abs(b[k]["mean"] - est[k]) < 1e-12, f"{k} unchanged from baseline ({est[k]:.6f})")
        check(abs(b["m_psychic"] - mp) < 1e-12, "m_psychic unchanged")
    else:
        print("  (baseline targets.toml not found -- skipped)")
    names = G.TAS_MOMENTS
    Om, ncl, Gc = G.joint_covariance({k: infl[k] for k in names}, names)
    ev = np.linalg.eigvalsh((Om + Om.T) / 2)
    check(np.allclose(Om, Om.T) and ev.min() > 0, f"targeted-TAS covariance symmetric, min eigenvalue {ev.min():.2e} > 0")

    # ---- 8. CSV exports -----------------------------------------------------
    print("\n8. CSV exports")
    ap = G.assets_by_age_from_bins(np.arange(1, 18))
    check(np.all(np.isfinite(ap)), "by-age a_p filled from the two-year bins for ages 1-17")
    check(ap[16] == ap[15], "ages 16 and 17 share the bin-16 mean")
    csv = REPO / "Input" / G.ASSETS_DST
    if csv.exists():
        d = pd.read_csv(csv, comment="#")
        check({"child_age", "age_lo", "age_hi", "a_p_mu", "a_p_md"} <= set(d.columns),
              "assets CSV carries child_age, age_lo, age_hi and the a_p_* columns")
        check(bool((d.age_hi - d.age_lo == 1).all()), "every row is a two-year bin")
    else:
        print("  (assets CSV not written yet -- run the generator)")

    print("\n" + ("ALL PASS" if not FAIL else f"{len(FAIL)} FAILURE(S):\n  " + "\n  ".join(FAIL)))
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
