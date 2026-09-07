"""Rebuild the candidate snapshot and report from frozen run evidence and Julia diagnostics."""
from pathlib import Path
import csv, datetime as dt, hashlib, json, math, re, tomllib
OUT=Path(__file__).resolve().parent
REPO=OUT.parents[2]
RUN=REPO/'output/smm_runs/2026-09-07_114138'
def read(p): return tomllib.loads(p.read_text())
def rows(p):
    with p.open() as f: return list(csv.DictReader(f))
e=read(RUN/'estimates.toml');ck=read(RUN/'checkpoint.toml');record=read(RUN/'run_record.toml')
fit=rows(OUT/'fit_moments.csv');probes=rows(OUT/'probes.csv');restarts=rows(RUN/'restarts.csv');d=read(OUT/'fit_diagnostics.toml')
assert len(fit)==10 and len(probes)==14 and len(restarts)==6
assert d['Q_reproduced']==e['Q_final']
assert math.isclose(sum(float(r['Q_contribution']) for r in fit),e['Q_final'],abs_tol=1e-13)
assert all(not r['error'] and int(r['invalid'])==0 for r in probes)
assert hashlib.sha256((RUN/'targets.toml').read_bytes()).hexdigest()[:16]==ck['targets_sha']
assert e['accepted'] is False and e['params_on_bound']==['sigma_2_1']
log=(RUN/'run.log').read_text();counts=[];times=[]
for j in range(1,7):
    before=log.split(f'  restart {j:3d}/6   DONE')[0]
    counts.append(int(re.findall(r'restart\s+'+str(j)+r'/6\s+eval\s+(\d+)',before)[-1]))
    times.append(float(re.search(r'restart\s+'+str(j)+r'/6\s+DONE.*?([\d.]+) min,',log)[1]))
assert 4001+sum(counts)+e['n_eval_polish']==e['n_eval_total']==6751
params={n:math.exp(z) if link=='log' else z for n,z,link in zip(ck['param_names'],ck['search_vector']['z'],ck['param_link'])}
snapshot=['# Provisional result, not promoted into PARENT_DEFAULTS.',
'source_run = "output/smm_runs/2026-09-07_114138"',
'source_code = "6a3a916"', 'targets_file = "output/smm_runs/2026-09-07_114138/targets.toml"',
'status = "provisional_boundary_review"','accepted = false',
'params_on_bound = ["sigma_2_1"]',f'Q_final = {e["Q_final"]!r}',
'seed = 1234','simN = 2000','grid = 30','', '[parameters]']
snapshot += [f'{k} = {v!r}' for k,v in params.items()]
snapshot += ['', '[fixed]', 'R_1 = 0.0', 'sigma_4_1 = 0.02', 'mu_1 = -0.04', '', '[run_bounds]']
snapshot += [f'{k} = [{a!r}, {b!r}]' for k,a,b in zip(ck['param_names'],ck['param_lo'],ck['param_hi'])]
snapshot += ['', '[source_sha256]']
snapshot += [f'{json.dumps(p.name)} = "{hashlib.sha256(p.read_bytes()).hexdigest()}"' for p in sorted(RUN.iterdir()) if p.is_file() and p.name != "candidate.toml"]
(REPO/'output/smm_runs/2026-09-07_114138/candidate.toml').write_text('\n'.join(snapshot)+'\n')
lines=['# Inspection of school-plus-study run 2026-09-07_114138','',
'**Provisional result: the winning polish met FTOL, but the run failed its conservative acceptance gate because `sigma_2_1` is at the lower search bound. Do not promote this point into model defaults yet.**','',
'The clean run at code `6a3a916` used **4,000 Sobol points, six restarts, 500-evaluation local/polish caps, grid 30, 2,000 simulated households, seed 1234, and 20 worker processes**. These are its actual settings, rather than the earlier proposed 2,000/5/400 pilot. Target SHA-256 prefix `74823b68b9d42e9e` matches the frozen `targets.toml`.','',
f'**Q = {e["Q_final"]} reproduces exactly**, with zero invalid cells. Evaluating the original nine-parameter baseline against these SAME new targets gives Q = {e["Q_incumbent"]}; the fitted reduction is {100*(1-e["Q_final"]/e["Q_incumbent"]):.3f}%. The old Q = 0.2500261422642604 used homework-only targets and different child-time residual scaling; it is not a comparable fit benchmark.','',
'## All nine estimates','',
'Values come from full-precision checkpoint search coordinates. Bound distance is measured in search coordinates (log for the first four). The 2% threshold is a review convention, not a convergence or identification test.','',
'| Parameter | Estimate | Lower | Upper | Nearest edge, % of search box | Status |',
'|---|---:|---:|---:|---:|---|']
for n,z,a,b,link in zip(ck['param_names'],ck['search_vector']['z'],ck['param_lo'],ck['param_hi'],ck['param_link']):
    lo,hi=(math.log(a),math.log(b)) if link=='log' else (a,b);pos=(z-lo)/(hi-lo);dist=100*min(pos,1-pos)
    status='Near lower' if pos<.02 else 'Near upper' if pos>.98 else 'Interior'
    lines.append(f'| `{n}` | {params[n]:.10f} | {a:g} | {b:g} | {dist:.6f} | {status} |')
lines += ['', '`sigma_2_1` is only 5.86e-7 above −0.10. `lambda_2=16.8011` is 2.91% from its upper edge in log space: outside the warning threshold, but worth monitoring in the next fit. The other parameters are comfortably inside their search boxes. `R_1=0`, `sigma_4_1=0.02`, and `mu_1=−0.04` remain fixed.','',
'## Targets, simulated moments and residuals','',
'Raw residual = model − target. Scaled residual divides by max(abs(target),0.05) for level moments and by 1 for log HC. These are not residuals standardized by sampling uncertainty.','',
'| Moment | Target | Simulated | Raw residual | Scaled residual | Share of Q, % |',
'|---|---:|---:|---:|---:|---:|']
for r in fit:
    lines.append(f'| `{r["moment"]}` | {float(r["target"]):.6f} | {float(r["simulated"]):.6f} | {float(r["raw_residual"]):+.6f} | {float(r["scaled_residual"]):+.6f} | {100*float(r["Q_share"]):.2f} |')
bym={r['moment']:r for r in fit};time_share=sum(float(bym[k]['Q_share']) for k in ['mean_i_c_early','mean_i_c_late'])
lines += ['',f'Child-time residuals account for **{100*time_share:.2f}% of Q**. The model predicts **{112*float(bym["mean_i_c_early"]["simulated"]):.2f} → {112*float(bym["mean_i_c_late"]["simulated"]):.2f} hours/week**, against **40.87 → 43.37** in the data. The age slope still goes in the wrong direction, despite the large improvement in levels. Early parental time and consumption are the next visible residuals.','',
'## Winning stage, restarts and runtime','',
'The winner is BOBYQA polish, `FTOL_REACHED`, 161 evaluations. Three local restarts met FTOL and three hit MAXEVAL; there were zero unclassified objective exceptions. `winner_converged=true` and `accepted=false` describe different checks. A constrained optimizer can meet its stopping criterion at a boundary; the runner requires additional boundary review before accepting such results.','',
'| Stage | Returned Q | Best-Q improvement, % | Status | Evaluations | Approx. minutes |',
'|---|---:|---:|---|---:|---:|',
'| Sobol + incumbent | 0.847918394 | — | Finished | 4001 | 30.2 |']
prev=read(RUN/'seeds.toml')['f_sobol_best'];tprev=30.2
for row,n,t in zip(restarts,counts,times):
    q=float(row['f_local']);gain=100*(prev-min(q,prev))/prev
    lines.append(f'| Restart {row["restart"]} | {q:.9f} | {gain:.4f} | {row["ret"]} | {n} | {t-tprev:.1f} |')
    prev=min(prev,q);tprev=t
lines.append(f'| Polish | {e["Q_final"]:.9f} | {100*(prev-e["Q_final"])/prev:.4f} | {e["polish_ret"]} | 161 | {478.4-tprev:.1f} |')
lines += ['', '**4,001 + 2,589 + 161 = 6,751 optimization-related evaluations**; no refinement. Local counts include a seed evaluation outside the 500-call solver cap. The search took 478.4 minutes; the saved duration including reporting is 479.1 minutes. Directory start 11:41:38 to final record 19:42:11 gives **8h 00m 33s (480.55 minutes)** including setup. Initial/report solves are outside the optimizer counter.','',
'Late gains are small: restart 6 improves its incumbent about 0.44%, and polish another 0.053%. However, restart 6 mixes in 99.5% of the current incumbent, so this is weak evidence about other basins. Thirty restarts on the same box would not investigate the active boundary. Half the restarts exhausted their current caps.','',
'## Invalid evaluations, coverage and handoff','',
'286/6,751 evaluations (4.24%) were penalized: 181 `AssertionError`, 75 feasibility-screen rejections (the label `infeasible_sigma_2` covers the combined screen), 29 recognized `ErrorException`, and one assets-below-minimum simulation. No per-draw messages or penalty locations are saved, so this aggregate cannot diagnose each failure. Final/reproduced invalid-cell count is zero.','']
g=d['grid_coverage'];h=d['handoff']
lines += [f'- Assets: {g["a_hh_ever_above"]*2000:.0f}/2,000 households exceed the ceiling 100 at initialization and at handoff; maximum {g["a_max_sim"]:.6f}. Thin mass does not by itself establish negligible flat-extrapolation error.',
f'- HC: all states remain inside [50,1500], with simulated range [{g["hc_min_sim"]:.6f},{g["hc_max_sim"]:.6f}].',
f'- Handoff assets: min {h["a_min"]:.6f}, mean {h["a_mean"]:.6f}, max {h["a_max"]:.6f} in model units ($10,000 each).',
f'- Handoff HC: min {h["hc_min"]:.6f}, mean {h["hc_mean"]:.6f}, max {h["hc_max"]:.6f}. BothCollege share {h["bc_share"]:.4f}; type is constant over family ages. These are inspected handoff states; no new child lifecycle simulation is certified here.',
f'- Untargeted saving rate {100*g["saving_rate"]:.2f}%; terminal mean assets ${10000*h["a_mean"]:,.0f}. Parent/child leisure interpretations retain the overlapping-parental-time caveat.','',
'## Conditional next-step probes','',
'Other fitted parameters are fixed except the listed change. Money pivots hold the money elasticity at age 9 constant while changing its slope and intercept together. These are slices, not local re-optimizations, uncertainty estimates, or identification tests.','',
'| Probe | Q | Change from fitted Q | Invalid cells |',
'|---|---:|---:|---:|']
for r in probes:lines.append(f'| `{r["case"]}` | {float(r["Q"]):.9f} | {float(r["delta_Q"]):+.9f} | {r["invalid"]} |')
lines += ['', 'The asset sensitivity expands BOTH parent and child asset ceilings to 300 with 30 nodes each, holding HC ranges matched and leaving the initial draws unchanged. This changes grid spacing as well as tail coverage, so it is not an isolated estimate of tail extrapolation bias. Inspect its Q and moment changes before deciding on a new common grid; do not compare re-estimated Q across different grids without re-evaluation.','',
'Changing only `sigma_4_1` to 0.03 improves Q by **6.83%**, to 0.010789139. The tested R_1 and mu_1 moves all worsen Q; this supports prioritizing sigma_4_1 among these candidates, conditional on this fitted point. It does not resolve the money-slope boundary or prove identification. Lowering sigma_2_1 alone, and the two tested intercept/slope pivots, all worsen Q; a joint local profile is still needed before attributing the boundary to an overly narrow box.','',
'At asset ceilings 300, no household remains above the asset grid, but Q rises **12.36%** (absolute +0.001431910) to 0.013012541. Among level moments, the largest relative movement is late monetary investment, about **1.46%**. Zero invalid cells persists. This is meaningful numerical sensitivity to assess, not evidence that the larger grid is wrong because its Q is higher.','',
'## Next steps','',
'1. Preserve this run and full-precision candidate separately from the historical baseline. The snapshot is `output/smm_runs/2026-09-07_114138/candidate.toml`; model defaults remain unchanged because acceptance is false.',
'2. Investigate the `sigma_2_1` boundary with a **joint local re-optimization**, initially exploring a provisional lower limit −0.15 instead of −0.10 and retaining upper 0.05. Re-optimize the other eight parameters too. Single-coordinate slices cannot settle a correlated boundary. Monitor `lambda_2` against 20. Do not alter the acceptance gate just to obtain a pass.',
'3. Check the asset-grid sensitivity and use the same selected grid for comparisons. Once this is settled, run a fresh nine-parameter pilot with 1,000 Sobol points, 6–10 restarts, and 1,000 local/polish evaluations as an initial budget. Monitor actual termination; no budget guarantees convergence. Changed bounds require a fresh run, not `--resume` of this one.',
'4. If the child-time age slope remains the dominant miss, `sigma_4_1` is the first structural extension to evaluate jointly with all nine. Compare the controlled probes below their stated fixed-parameter assumptions; do not call them fitted ten-parameter models. R_1 and mu_1 require a distinct economic/identification rationale, not just another degree of freedom. With ten moments, add at most one parameter without adding moments.',
'5. Recompute the Jacobian at the final fit with multiple step sizes and inspect rank/conditioning. Resolve the open inference items in REVIEW_TRIAGE before quoting uncertainty. Only then spend on the planned 1,000-Sobol/30-restart production search. A low Q does not finish all Tier 2 work.','',
'No new estimation, default promotion, or search-bound change was made during this inspection. The boundary warning text was corrected to distinguish proximity from convergence; its conservative acceptance gate is retained.','',
'## Reproduction','',
'```sh',
'julia --threads=1 --project=. output/smm_diagnostics/2026-09-07_114138/inspect_run.jl',
'python3 output/smm_diagnostics/2026-09-07_114138/summarize_run.py',
'```','',
'The frozen run and target hashes are recorded in the candidate snapshot. `probe_settings.toml`, `probes.csv`, `fit_moments.csv`, `fit_diagnostics.toml`, and `inspection.log` retain the evidence. Original six run files are not rewritten.']
(OUT/'inspection_notes.md').write_text('\n'.join(lines)+'\n')
print('Validated 9 parameters, 10 moments, 6 restarts, 14 probe rows; wrote candidate snapshot and report')
