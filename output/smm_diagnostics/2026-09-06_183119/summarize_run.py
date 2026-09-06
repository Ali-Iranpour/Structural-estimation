"""Summarize saved run evidence and the inspect_run.jl diagnostics; original results remain untouched."""
from pathlib import Path
import csv, math, tomllib, hashlib, re
OUT=Path(__file__).resolve().parent
REPO=OUT.parents[2]
RUN=REPO/'output/smm_runs/2026-09-06_183119'
def read_toml(path): return tomllib.loads(path.read_text())
def csv_rows(path): return list(csv.DictReader(path.open()))
est=read_toml(RUN/'estimates.toml'); ck=read_toml(RUN/'checkpoint.toml')
fit=csv_rows(OUT/'fit_moments.csv'); probes=csv_rows(OUT/'probes.csv'); restarts=csv_rows(RUN/'restarts.csv')
d=read_toml(OUT/'fit_diagnostics.toml')
assert len(fit)==10 and len(probes)==25 and len(restarts)==5
assert all(not r['error'] and int(r['invalid'])==0 for r in probes)
assert math.isclose(sum(float(r['Q_contribution']) for r in fit),est['Q_final'],abs_tol=1e-12)
assert d['Q_reproduced']==est['Q_final']
assert hashlib.sha256((REPO/'Input/smm_targets_baseline.toml').read_bytes()).hexdigest()[:16]==ck['targets_sha']
lines=['# Inspection of estimation 2026-09-06_183119','',
'**Assessment: useful converged initial fit; substantial age-profile misfit remains. Investigate `sigma_4_1` before paying for a much larger search.**','',
'This run used **2,000 Sobol points, five restarts, 400 evaluations per local search, and a 400-evaluation polish cap**. Both parent grids are 30; child grids are 30×30×5, simulation size 2,000, seed 1234. The saved winner was reproduced exactly: **Q = 0.2500261422642604**. The input target hash also matches.','',
'## Nine estimates and bounds','',
'Position and nearest-bound distance are percentages of the box in actual search coordinates (logs for the first four parameters). “Near” uses the runner’s 2% convention; it is not proof of a binding constraint.','',
'| Parameter | Estimate | Lower | Upper | Position from lower (%) | Nearest-bound distance (%) | Status |',
'|---|---:|---:|---:|---:|---:|---|']
for n,z,lo,hi,link in zip(ck['param_names'],ck['search_vector']['z'],ck['param_lo'],ck['param_hi'],ck['param_link']):
    a,b=(math.log(lo),math.log(hi)) if link=='log' else (lo,hi)
    pos=(z-a)/(b-a); dist=100*min(pos,1-pos)
    status=('Near lower' if pos<.5 else 'Near upper') if dist<2 else 'Interior'
    lines.append(f'| `{n}` | {ck["parameters"][n]:.8f} | {lo:g} | {hi:g} | {100*pos:.4f} | {dist:.4f} | {status} |')
lines += ['', 'The three near-bound estimates are `sigma_1_0`, `sigma_2_1`, and `sigma_4_0`. Neither `lambda_2` nor `R_0` is near a bound in the search metric.','',
'## Exact moments and residuals','',
'Raw residual = simulated − target. Scaled residual = raw residual / scale; Q is the sum of their squares. Level scales are max(abs(target), 0.05); log-HC scales are 1. Thus the two study-time residuals are divided by 0.05, not their targets. These are **not standard-error-standardized residuals**.','',
'| Moment | Target | Simulated | Raw residual | Scaled residual | Share of Q (%) |',
'|---|---:|---:|---:|---:|---:|']
for r in fit:
    lines.append(f'| `{r["moment"]}` | {float(r["target"]):.6f} | {float(r["simulated"]):.6f} | {float(r["raw_residual"]):+.6f} | {float(r["scaled_residual"]):+.6f} | {100*float(r["Q_share"]):.3f} |')
lines += ['',
'**Study time supplies 72.69% of Q**, with late study alone supplying 50.13%. The model predicts **5.73 → 3.57 hours/week**, while data targets are **4.40 → 5.55**. Early study is 30.2% too high; late study is 35.7% too low. Early parental time is 15.4% too high; monetary investment is 14.7% too low early and 11.5% too high late. HC misses are −0.05447 and +0.04761 log points (approximately −5.30% and +4.88% in levels).','',
'## Termination, restart improvements and cost','',
'The retained winner is the **BOBYQA polish**, returning **FTOL_REACHED**, improving the incumbent, with 149 evaluations. Four restarts met FTOL; restart 1 exhausted its 400-call optimizer budget. There were no unclassified objective exceptions and zero final invalid cells. The corrected winner-specific gate accepted the result. FTOL is a stopping test, not a global-minimum certificate; see [NLopt reference](https://nlopt.readthedocs.io/en/stable/NLopt_Reference/) and [introduction](https://nlopt.readthedocs.io/en/latest/NLopt_Introduction/).','',
'| Stage | Returned Q | Best-Q gain (%) | Termination | Evaluations | Approx. minutes |',
'|---|---:|---:|---|---:|---:|',
'| Sobol + incumbent | 1.377493142 | — | Completed | 2001 | 15.2 |']
prev=read_toml(RUN/'seeds.toml')['f_sobol_best']; nev=[401,395,325,181,107]; mins=[59.6,59.9,49.8,28.1,16.6]
for row,n,m in zip(restarts,nev,mins):
    q=float(row['f_local']); new=min(q,prev); gain=100*(prev-new)/prev
    lines.append(f'| Restart {row["restart"]} | {q:.9f} | {gain:.4f} | {row["ret"]} | {n} | {m:.1f} |'); prev=new
lines.append(f'| Final polish | {est["Q_final"]:.9f} | {100*(prev-est["Q_final"])/prev:.4f} | {est["polish_ret"]} | 149 | 23.5 |')
lines += ['',
'The last logged per-restart counts sum to 1,409 and reconcile exactly: **2,001 + 1,409 + 149 = 3,559 optimization-related evaluations**. Local counts include the seed evaluation outside NLopt, explaining 401 against a 400 cap. Phase minutes are differenced from rounded log times. The main search took 252.7 minutes; the saved duration including final reporting is 253.4 minutes. Start 18:31:19 to final record 22:46:11 gives **254.87 minutes (4h 14m 52s)**, including setup. Additional startup/report solves are outside the 3,559 counter. Local stages plus polish took approximately 237.5 minutes; adding Sobol points will not accelerate them.','',
'Restart 4 improved Q by only 0.0001607 (0.0642%), restart 5 did not improve, and polish gained 0.0002028 (0.0810%). However, restart 5 used a 99.5% incumbent mixture: it provides limited evidence about other basins. One in-box probe setting `sigma_2_1` exactly to −0.05 improves Q by another 0.00003257 (0.0130%). Treat the stopping flag as numerical termination, not exact optimality.','',
'## Invalid evaluations, grids and handoff','',
'- **178 penalized evaluations (reported 5.00%)**: 118 assertion failures, 37 feasibility-screen rejections, 23 recognized solver/error failures. These are not 178 invalid cells in the final simulation. No per-stage/per-point penalty trace is saved, so their concentration and individual messages cannot be reconstructed.',
'- Final and reproduced simulation: **0 invalid cells**, including assets and HC at age 18. No HC state is outside [50,1500]; simulated range is [297.6691,632.9615].',
'- **2/2,000 households (0.10%)** exceed the asset ceiling at every period, including initialization and handoff; no additional households cross it during the family stage. Maximum assets are 254.0333 against a ceiling of 100. Small mass alone does not establish negligible extrapolation error.',
'- Handoff assets: minimum 10.06445, mean 37.08583, maximum 254.03329 model units (one unit = $10,000). Handoff HC: minimum 488.8261, mean 525.8977, maximum 553.9049. BothCollege share is 0.3045 and type is constant across ages. These inspect the handoff states; a new child-lifecycle simulation was not run.',
'- Untargeted implications remain material: terminal mean assets about **$370,858**, implied saving rate **31.27%**, model leisure **29.61 hours/week**. The broad parental-time target overlaps measured leisure; that limits preference interpretation.','',
'## Which extra parameter?','',
'**First: `sigma_4_1`.** The dominant miss is the slope of study time, and the actual full-grid probes improve it in the predicted direction. All 25 diagnostic probes had zero invalid simulated cells. These are controlled evaluations with the remaining estimates fixed, **not joint re-estimations or identification tests**.','',
'| Diagnostic | Q | Interpretation |','|---|---:|---|']
bycase={r['case']:r for r in probes}
for key,note in [('sigma_4_1=0.04','Only this slope changes; nine fitted values held fixed.'),('study_pivot_slope=0.08','Slope 0.08 and intercept −6.345218799; elasticity at age 11 held fixed.'),('study_pivot_slope=0.12','Slope 0.12 and intercept −6.585218799; elasticity at age 11 held fixed.'),('R_1=-0.5','Other estimates fixed; negative TFP slope.'),('R_1=0.5','Other estimates fixed; positive TFP slope.'),('mu_1=-0.05','Other estimates fixed; bargaining-weight change.'),('mu_1=-0.03','Other estimates fixed; bargaining-weight change.')]:
    lines.append(f'| `{key}` | {float(bycase[key]["Q"]):.9f} | {note} |')
lines += ['',
'The slope-0.12/intercept−6.5852 probe reduces Q **56.92%** and gives study hours **4.09 early / 4.49 late**, compared with baseline 5.73 / 3.57. Its early/late HC moments barely move. This candidate directly addresses the largest miss. The intercept is outside the current −6 lower bound, so this is evidence to investigate the joint intercept/slope domain, not a solution inside the old box. Merely lowering `sigma_4_0` with the old slope made Q worse.','',
'**Second: `R_1`, if an age-TFP question remains after study time is addressed.** In this code `R_t = R_0 + R_1(t−1)`, an additive slope, not a log growth rate. Tested changes −1, −0.5, +0.5, +1 all increased Q with other estimates fixed. R_1 affects the HC age profile broadly; HC currently accounts for only 2.09% of Q. Joint re-estimation might change its merit. Enforce positive R_t over all ages for any proposed search domain; a fixed rectangular box must account for changing R_0.','',
'**Third: `mu_1` as an addition to the current set.** Tested changes −0.06, −0.05, −0.03, −0.02 all increased Q with other estimates fixed. The direct study-FOC term is `log(sigma_4_t/(1−mu_t)) = sigma_4_0 − log(−mu_1) + sigma_4_1(t−5) − log(t−5)`. Thus mu_1 directly shifts its intercept and competes with sigma_4_0; it also has endogenous bargaining/value effects, so this is not a claim of exact observational equivalence. Old calibration cosines are not fitted-point identification evidence. With mu_0=1 and t=6..17, a convex parental weight requires −1/12 ≤ mu_1 ≤ 0, preferably avoiding degenerate endpoints. The old identification candidate box [−0.20,−0.005] violates that requirement in part.','',
'## Recommended next sequence','',
'1. Preserve this nine-parameter run as the comparison baseline. Correct the boundary finite-difference bug before using jacobian.jl or standard_errors.jl for inference; three estimates are close enough to bounds for the old 2h denominator to matter.',
'2. Run a limited ten-parameter **sigma_4_1 pilot**, jointly re-optimizing the original nine. Retain mu_1=−0.04 and R_1=0. Include this fitted vector as an explicit seed. Test a domain that permits sigma_4_0 below −6 and sigma_4_1 above 0.05; the successful controlled probes motivate this, but do not select final bounds. A centred study-elasticity intercept at age 11 may make this tradeoff easier to parameterize.',
'3. Check the other near-bound parameters with a constrained local profile. The probes do **not** support automatically widening all three bounds: relaxing sigma_1_0 to −0.15 or sigma_2_1 to −0.075 worsened Q with the other estimates fixed. Joint profiles are needed.',
'4. Use full grid 30, common seed and unchanged targets/weights for comparisons. Confirm grid-range sensitivity for the two asset-tail households, preserving the initial draw and keeping all node counts ≤30.',
'5. After the ten-parameter pilot, expand local restarts in stages (e.g. 10 before 30), with enough per-start budget to avoid repeated MAXEVAL. More than 400 is justified for starts like restart 1; choose final caps from their traces. The initial run already screened 2,000 Sobol points. A 1,000/30 run would reduce global screening while increasing local work. Do not use --resume to change dimension, boxes or restart totals; an explicit saved-fit start route is needed because current run_smm.jl seeds only incumbent().',
'6. Repeat bound-aware fitted-point Jacobians at several steps and check separation before interpreting a tenth parameter. Ten moments versus ten parameters does not ensure identification or exact fit. Follow specification/advisor conventions before results circulate.','',
'## Provenance limitations and remaining tooling issues','',
'`run_record.toml` fails a standard TOML parse at line 41: child_grid is an unquoted Julia NamedTuple. The current writer has already changed this formatting, but the saved file was left untouched. Its runner format matches ca1ce54; it stamps bc02091 at completion, even though bc02091 was committed at 18:42:51 after the 18:31:19 run start. The exact startup runner revision is therefore not established by the completion stamp. The model, moments and TikTak source files did not change between these commits, the target hash matches, and the winner reproduces exactly, so this does not invalidate the numerical fit. Future records should preserve code identity at startup and not replace it with end-of-run HEAD.','',
'The previous review’s A1/A2 defects are fixed for this run: terminal HC is checked and acceptance tracks the winning stage. ErrorException classification was narrowed. Resume was not used here and is not re-certified by this inspection. The known Jacobian-bound and inference defects remain and prevent treating the fitted run as ready for standard errors or parameter-selection inference.','',
'## Reproduction and sources','',
'- [Original estimates](../../smm_runs/2026-09-06_183119/estimates.toml), [restart trace](../../smm_runs/2026-09-06_183119/restarts.csv), [run log](../../smm_runs/2026-09-06_183119/run.log), [checkpoint](../../smm_runs/2026-09-06_183119/checkpoint.toml).',
'- [Diagnostic Julia script](inspect_run.jl), [summary/validation script](summarize_run.py), [exact moments](fit_moments.csv), [grid and handoff diagnostics](fit_diagnostics.toml), [all 25 probes](probes.csv), [probe settings](probe_settings.toml), [execution log](inspection.log).',
'- No estimation-source changes or expanded estimation were performed. The supporting probes took about 250 cumulative parent-solve seconds across four workers, plus startup and the baseline check.','']
(OUT/'inspection_notes.md').write_text('\n'.join(lines))
print('Validated 9 estimates, 10 moments, 5 restarts, 25 probes; wrote inspection_notes.md')
