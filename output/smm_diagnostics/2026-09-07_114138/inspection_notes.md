# Inspection of school-plus-study run 2026-09-07_114138

**Provisional result: the winning polish met FTOL, but the run failed its conservative acceptance gate because `sigma_2_1` is at the lower search bound. Do not promote this point into model defaults yet.**

The clean run at code `6a3a916` used **4,000 Sobol points, six restarts, 500-evaluation local/polish caps, grid 30, 2,000 simulated households, seed 1234, and 20 worker processes**. These are its actual settings, rather than the earlier proposed 2,000/5/400 pilot. Target SHA-256 prefix `74823b68b9d42e9e` matches the frozen `targets.toml`.

**Q = 0.011580631181773578 reproduces exactly**, with zero invalid cells. Evaluating the original nine-parameter baseline against these SAME new targets gives Q = 1.6495643445176547; the fitted reduction is 99.298%. The old Q = 0.2500261422642604 used homework-only targets and different child-time residual scaling; it is not a comparable fit benchmark.

## All nine estimates

Values come from full-precision checkpoint search coordinates. Bound distance is measured in search coordinates (log for the first four). The 2% threshold is a review convention, not a convergence or identification test.

| Parameter | Estimate | Lower | Upper | Nearest edge, % of search box | Status |
|---|---:|---:|---:|---:|---|
| `phi_2` | 0.1795556705 | 0.01 | 20 | 37.994175 | Interior |
| `phi_3` | 1.2482684976 | 0.05 | 20 | 46.298778 | Interior |
| `lambda_2` | 16.8010821885 | 0.05 | 20 | 2.908954 | Interior |
| `R_0` | 48.9078093010 | 0.5 | 100 | 13.499250 | Interior |
| `sigma_1_0` | -0.7035276954 | -4 | -0.1 | 15.475069 | Interior |
| `sigma_1_1` | -0.1195904942 | -0.2 | 0.05 | 32.163802 | Interior |
| `sigma_2_0` | -3.7724422199 | -5 | -0.5 | 27.279062 | Interior |
| `sigma_2_1` | -0.0999994140 | -0.1 | 0.05 | 0.000391 | Near lower |
| `sigma_4_0` | -3.1709961592 | -6 | -1 | 43.419923 | Interior |

`sigma_2_1` is only 5.86e-7 above −0.10. `lambda_2=16.8011` is 2.91% from its upper edge in log space: outside the warning threshold, but worth monitoring in the next fit. The other parameters are comfortably inside their search boxes. `R_1=0`, `sigma_4_1=0.02`, and `mu_1=−0.04` remain fixed.

## Targets, simulated moments and residuals

Raw residual = model − target. Scaled residual divides by max(abs(target),0.05) for level moments and by 1 for log HC. These are not residuals standardized by sampling uncertainty.

| Moment | Target | Simulated | Raw residual | Scaled residual | Share of Q, % |
|---|---:|---:|---:|---:|---:|
| `mean_c_p` | 3.115533 | 3.231571 | +0.116038 | +0.037245 | 11.98 |
| `mean_h_p` | 0.307292 | 0.301263 | -0.006029 | -0.019618 | 3.32 |
| `mean_t_p_early` | 0.454449 | 0.474110 | +0.019661 | +0.043264 | 16.16 |
| `mean_t_p_late` | 0.333317 | 0.330748 | -0.002569 | -0.007707 | 0.51 |
| `mean_e_p_early` | 0.342896 | 0.336023 | -0.006873 | -0.020043 | 3.47 |
| `mean_e_p_late` | 0.391094 | 0.399520 | +0.008426 | +0.021546 | 4.01 |
| `mean_i_c_early` | 0.364873 | 0.389248 | +0.024375 | +0.066805 | 38.54 |
| `mean_i_c_late` | 0.387210 | 0.368028 | -0.019182 | -0.049538 | 21.19 |
| `mean_hc_early` | 6.073689 | 6.064589 | -0.009100 | -0.009100 | 0.72 |
| `mean_hc_late` | 6.250776 | 6.254203 | +0.003427 | +0.003427 | 0.10 |

Child-time residuals account for **59.73% of Q**. The model predicts **43.60 → 41.22 hours/week**, against **40.87 → 43.37** in the data. The age slope still goes in the wrong direction, despite the large improvement in levels. Early parental time and consumption are the next visible residuals.

## Winning stage, restarts and runtime

The winner is BOBYQA polish, `FTOL_REACHED`, 161 evaluations. Three local restarts met FTOL and three hit MAXEVAL; there were zero unclassified objective exceptions. `winner_converged=true` and `accepted=false` describe different checks. A constrained optimizer can meet its stopping criterion at a boundary; the runner requires additional boundary review before accepting such results.

| Stage | Returned Q | Best-Q improvement, % | Status | Evaluations | Approx. minutes |
|---|---:|---:|---|---:|---:|
| Sobol + incumbent | 0.847918394 | — | Finished | 4001 | 30.2 |
| Restart 1 | 0.012216231 | 98.5593 | MAXEVAL_REACHED | 501 | 87.6 |
| Restart 2 | 0.013396313 | 0.0000 | MAXEVAL_REACHED | 501 | 83.0 |
| Restart 3 | 0.012423137 | 0.0000 | MAXEVAL_REACHED | 501 | 85.8 |
| Restart 4 | 0.011637865 | 4.7344 | FTOL_REACHED | 377 | 58.5 |
| Restart 5 | 0.011677171 | 0.0000 | FTOL_REACHED | 499 | 76.5 |
| Restart 6 | 0.011586772 | 0.4390 | FTOL_REACHED | 210 | 32.5 |
| Polish | 0.011580631 | 0.0530 | FTOL_REACHED | 161 | 24.3 |

**4,001 + 2,589 + 161 = 6,751 optimization-related evaluations**; no refinement. Local counts include a seed evaluation outside the 500-call solver cap. The search took 478.4 minutes; the saved duration including reporting is 479.1 minutes. Directory start 11:41:38 to final record 19:42:11 gives **8h 00m 33s (480.55 minutes)** including setup. Initial/report solves are outside the optimizer counter.

Late gains are small: restart 6 improves its incumbent about 0.44%, and polish another 0.053%. However, restart 6 mixes in 99.5% of the current incumbent, so this is weak evidence about other basins. Thirty restarts on the same box would not investigate the active boundary. Half the restarts exhausted their current caps.

## Invalid evaluations, coverage and handoff

286/6,751 evaluations (4.24%) were penalized: 181 `AssertionError`, 75 feasibility-screen rejections (the label `infeasible_sigma_2` covers the combined screen), 29 recognized `ErrorException`, and one assets-below-minimum simulation. No per-draw messages or penalty locations are saved, so this aggregate cannot diagnose each failure. Final/reproduced invalid-cell count is zero.

- Assets: 2/2,000 households exceed the ceiling 100 at initialization and at handoff; maximum 257.171230. Thin mass does not by itself establish negligible flat-extrapolation error.
- HC: all states remain inside [50,1500], with simulated range [297.669145,559.958246].
- Handoff assets: min 9.402103, mean 37.255208, max 257.171230 in model units ($10,000 each).
- Handoff HC: min 425.270260, mean 467.542385, max 493.376848. BothCollege share 0.3045; type is constant over family ages. These are inspected handoff states; no new child lifecycle simulation is certified here.
- Untargeted saving rate 30.99%; terminal mean assets $372,552. Parent/child leisure interpretations retain the overlapping-parental-time caveat.

## Conditional next-step probes

Other fitted parameters are fixed except the listed change. Money pivots hold the money elasticity at age 9 constant while changing its slope and intercept together. These are slices, not local re-optimizations, uncertainty estimates, or identification tests.

| Probe | Q | Change from fitted Q | Invalid cells |
|---|---:|---:|---:|
| `fitted` | 0.011580631 | +0.000000000 | 0 |
| `sigma_2_1=-0.11` | 0.022897194 | +0.011316563 | 0 |
| `sigma_2_1=-0.125` | 0.080920424 | +0.069339793 | 0 |
| `sigma_2_1=-0.15` | 0.236419078 | +0.224838447 | 0 |
| `sigma_4_1=0.03` | 0.010789139 | -0.000791492 | 0 |
| `sigma_4_1=0.04` | 0.014000490 | +0.002419859 | 0 |
| `sigma_4_1=0.06` | 0.033166679 | +0.021586047 | 0 |
| `R_1=-0.5` | 0.057326723 | +0.045746092 | 0 |
| `R_1=0.5` | 0.052197578 | +0.040616947 | 0 |
| `mu_1=-0.05` | 0.058655948 | +0.047075317 | 0 |
| `mu_1=-0.03` | 0.071539613 | +0.059958982 | 0 |
| `money_pivot=-0.125` | 0.020364602 | +0.008783971 | 0 |
| `money_pivot=-0.15` | 0.062386192 | +0.050805561 | 0 |
| `asset_max=300` | 0.013012541 | +0.001431910 | 0 |

The asset sensitivity expands BOTH parent and child asset ceilings to 300 with 30 nodes each, holding HC ranges matched and leaving the initial draws unchanged. This changes grid spacing as well as tail coverage, so it is not an isolated estimate of tail extrapolation bias. Inspect its Q and moment changes before deciding on a new common grid; do not compare re-estimated Q across different grids without re-evaluation.

Changing only `sigma_4_1` to 0.03 improves Q by **6.83%**, to 0.010789139. The tested R_1 and mu_1 moves all worsen Q; this supports prioritizing sigma_4_1 among these candidates, conditional on this fitted point. It does not resolve the money-slope boundary or prove identification. Lowering sigma_2_1 alone, and the two tested intercept/slope pivots, all worsen Q; a joint local profile is still needed before attributing the boundary to an overly narrow box.

At asset ceilings 300, no household remains above the asset grid, but Q rises **12.36%** (absolute +0.001431910) to 0.013012541. Among level moments, the largest relative movement is late monetary investment, about **1.46%**. Zero invalid cells persists. This is meaningful numerical sensitivity to assess, not evidence that the larger grid is wrong because its Q is higher.

## Next steps

1. Preserve this run and full-precision candidate separately from the historical baseline. The snapshot is `Input/parent_candidate_school_time.toml`; model defaults remain unchanged because acceptance is false.
2. Investigate the `sigma_2_1` boundary with a **joint local re-optimization**, initially exploring a provisional lower limit −0.15 instead of −0.10 and retaining upper 0.05. Re-optimize the other eight parameters too. Single-coordinate slices cannot settle a correlated boundary. Monitor `lambda_2` against 20. Do not alter the acceptance gate just to obtain a pass.
3. Check the asset-grid sensitivity and use the same selected grid for comparisons. Once this is settled, run a fresh nine-parameter pilot with 1,000 Sobol points, 6–10 restarts, and 1,000 local/polish evaluations as an initial budget. Monitor actual termination; no budget guarantees convergence. Changed bounds require a fresh run, not `--resume` of this one.
4. If the child-time age slope remains the dominant miss, `sigma_4_1` is the first structural extension to evaluate jointly with all nine. Compare the controlled probes below their stated fixed-parameter assumptions; do not call them fitted ten-parameter models. R_1 and mu_1 require a distinct economic/identification rationale, not just another degree of freedom. With ten moments, add at most one parameter without adding moments.
5. Recompute the Jacobian at the final fit with multiple step sizes and inspect rank/conditioning. Resolve the open inference items in REVIEW_TRIAGE before quoting uncertainty. Only then spend on the planned 1,000-Sobol/30-restart production search. A low Q does not finish all Tier 2 work.

No new estimation, default promotion, or search-bound change was made during this inspection. The boundary warning text was corrected to distinguish proximity from convergence; its conservative acceptance gate is retained.

## Reproduction

```sh
julia --threads=1 --project=. output/smm_diagnostics/2026-09-07_114138/inspect_run.jl
python3 output/smm_diagnostics/2026-09-07_114138/summarize_run.py
```

The frozen run and target hashes are recorded in the candidate snapshot. `probe_settings.toml`, `probes.csv`, `fit_moments.csv`, `fit_diagnostics.toml`, and `inspection.log` retain the evidence. Original six run files are not rewritten.
