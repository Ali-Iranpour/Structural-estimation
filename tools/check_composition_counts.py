#!/usr/bin/env python3
"""check_composition_counts.py -- CROSS-CHECK of the composition counts (Ali, 2026-10-02, item 3d).

The decision is that the counts come from Stata (Code/28_smm_moments.do, rows comp_* in SMM_Constants.csv).
This script recomputes the same twelve frames (docs/SMM_COMPOSITION.md) in Python from the micro file 28 saves,
Output/Data/SMM/SMM_Moments_Stack.dta (the stacked frames AFTER smm_gen, so e_w, e_y and ratio are the capped
values the moments use), applying each moment's own listwise sample (28's _mom1: its `if` plus non-missing
values of its variables). It checks that every pooled S moment's N in SMM_Moments.csv is reproduced, and,
when SMM_Constants.csv has comp_* rows, that they equal these counts cell by cell.
NOT for use in an estimation without Ali's OK.

    uv run --with pandas --with pyreadstat python tools/check_composition_counts.py [CHILD_TIME_STUDY_DIR]
"""
import sys, re
from pathlib import Path
import pandas as pd

CTS = Path(sys.argv[1] if len(sys.argv) > 1 else "/srv/project/speech/apps/Child_Time_Study")
SMM = CTS / "Output/Data/SMM"
st = pd.read_stata(SMM / "SMM_Moments_Stack.dta", convert_categoricals=False)
mom = pd.read_csv(SMM / "SMM_Moments.csv").set_index("moment")

# age at the end of a pair: the child's age at the CDS wave 5 years after the base (28 builds frame 2 from the
# same child-wave file; the pair frame in the stack does not keep age_end)
cw = pd.read_stata(CTS / "Output/Data/Child_Time_Stylized_facts_v2_alternatives.dta",
                   columns=["Fam_id", "Per_id", "Year", "Child_Age"], convert_categoricals=False)
cw = cw.rename(columns={"Year": "Year_end", "Child_Age": "age_end"})
f1 = st[st.frm == 1]
f2 = st[st.frm == 2].copy()
f2["Year_end"] = f2.Year + 5
f2 = f2.merge(cw, on=["Fam_id", "Per_id", "Year_end"], how="left", validate="1:1")
assert f2.age_end.notna().all() and (f2.age_end <= 17).all(), "pair end ages"
f3 = st[st.frm == 3]
nn = lambda d, *v: d.dropna(subset=list(v))

FRAMES = {   # frame -> (rows, pair?, odd mask or None)
    "O_LW":   (nn(f1, "LW"), False, None),
    "O_taup": (nn(f1, "taup", "LW"), False, None),
    "O_tauc": (nn(f1[f1.age >= 6], "tauc", "LW"), False, None),
    "O_ep":   (nn(f1, "e_w", "LW"), False, lambda d: d.Year.isin([1997, 2007])),
    "P_LW":   (nn(f2, "LW", "LWe"), True, None),
    "P_taup": (nn(f2, "taup", "dLW"), True, None),
    "P_tauc": (nn(f2, "tauc", "dLW"), True, None),
    "P_ep":   (nn(f2, "e_w", "dLW"), True, lambda d: d.Year == 1997),
    "D_taup": (nn(f1, "taup"), False, None),
    "D_tauc": (nn(f1, "tauc"), False, None),
    "E_ep":   (nn(f3, "e_y"), False, None),
    "E_epY":  (nn(f3, "ratio"), False, None),
}
cells = {}
for fr, (d, pair, odd) in FRAMES.items():
    keys = ["age", "age_end"] if pair else ["age"]
    g = d.groupby(keys).size()
    o = d[odd(d)].groupby(keys).size() if odd else None
    cells[fr] = {k if isinstance(k, tuple) else (k,): (int(n), int(o.get(k, 0)) if o is not None else None) for k, n in g.items()}

CHECKS = [(f"S1_mean_LW_age{a}", "O_LW", a, a) for a in range(3, 18)] + [
    ("S3_sd_LW_3_17", "O_LW", 3, 17), ("S3_sd_LW_3_7", "O_LW", 3, 7), ("S3_sd_LW_8_11", "O_LW", 8, 11), ("S3_sd_LW_12_17", "O_LW", 12, 17),
    ("S4_mean_dLW_base3_7", "P_LW", 3, 7), ("S4_mean_dLW_base8_12", "P_LW", 8, 12),
    ("S5_corr_LW_LWt5_base3_12", "P_LW", 3, 12), ("S5_corr_LW_LWt5_base3_7", "P_LW", 3, 7), ("S5_corr_LW_LWt5_base8_12", "P_LW", 8, 12),
    ("S6_corr_taup_LW_3_17", "O_taup", 3, 17), ("S6_corr_tauc_LW_6_17", "O_tauc", 6, 17), ("S6_corr_ep_LW_3_17", "O_ep", 3, 17)]
for x, fr, bins in (("taup", "P_taup", ((3, 12), (3, 7), (8, 12))), ("tauc", "P_tauc", ((6, 12), (6, 7), (8, 12))), ("ep", "P_ep", ((3, 12), (3, 7), (8, 12)))):
    CHECKS += [(f"S7_corr_{x}_dLW_base{lo}_{hi}", fr, lo, hi) for lo, hi in bins]
for x, fr, bins in (("taup", "D_taup", ((3, 5), (6, 8), (9, 12), (13, 17))), ("ep", "E_ep", ((3, 5), (6, 8), (9, 12), (13, 17))), ("tauc", "D_tauc", ((6, 8), (9, 12), (13, 17)))):
    for lo, hi in bins:
        CHECKS += [(f"S8_mean_{x}_{lo}_{hi}", fr, lo, hi), (f"S8_sd_{x}_{lo}_{hi}", fr, lo, hi)]
CHECKS.append(("S9_mean_ep_over_Y_3_17", "E_epY", 3, 17))

bad = 0
for name, fr, lo, hi in CHECKS:
    got = sum(n for k, (n, _) in cells[fr].items() if lo <= k[0] <= hi)
    want = int(mom.loc[name, "n_obs"])
    if got != want:
        bad += 1; print(f"MISMATCH {name}: frame {fr} bins {lo}-{hi} sum {got}, moment N {want}")
print(f"{len(CHECKS)} pooled-moment N checks (S1 per age + 44 pooled rows): {len(CHECKS) - bad} reproduced, {bad} mismatched")
for fr, c in cells.items():
    tot = sum(n for n, _ in c.values()); od = sum(o for _, o in c.values() if o is not None)
    print(f"  {fr:7s} {len(c):3d} cells, {tot:5d} rows" + (f", {od} at odd waves" if FRAMES[fr][2] else ""))

const = pd.read_csv(SMM / "SMM_Constants.csv")
comp = const[const.name.str.startswith("comp_")]
if len(comp):
    stata = {}
    for nm, v in zip(comp.name, comp.value):
        m = re.fullmatch(r"comp_([A-Za-z]+_[A-Za-z]+)_(\d+)(?:_(\d+))?(_odd)?", nm)
        k = (int(m.group(2)),) + ((int(m.group(3)),) if m.group(3) else ())
        stata.setdefault((m.group(1), k), [0, 0])[1 if m.group(4) else 0] = int(round(v))
    py = {(fr, k): [n, o or 0] for fr, c in cells.items() for k, (n, o) in c.items()}
    diff = sorted(set(stata) ^ set(py)) + [k for k in py if k in stata and py[k] != stata[k]]
    print(f"Stata comp_* rows: {len(comp)}; cells differing from this script: {len(diff)}", diff[:10])
else:
    print("SMM_Constants.csv has no comp_* rows yet (28 not rerun)")
sys.exit(1 if bad else 0)
