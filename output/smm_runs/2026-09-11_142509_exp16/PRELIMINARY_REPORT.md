# Sixteen-parameter pilot — preliminary report (written while the run is in progress)

**Status at writing (2026-09-11 ~14:45): RUNNING.** Sobol stage complete (1,001
evaluations, 10.6 min, best Q 451.7 = the seeded warm start; no Sobol draw beat it).
Restart 1 of 5 in progress. This file records everything established *before and during*
the run; the run's own `run.log`, `restarts.csv`, `estimates.toml` and `run_record.toml`
are authoritative for the outcome once it finishes. **The specification change to the HC
technology has not been through the advisor. Nothing here circulates.**

## 1. What was run

```
julia --project=. code/smm/run_smm.jl --sobol 1000 --restarts 5 --local-evals 500 --skip-polish
      --grid 30 --procs 20 --seed 1234
      --targets output/smm_runs/2026-09-11_134542_120129_targets/targets.toml
      --init-from output/smm_runs/2026-09-10_183649/estimates.toml
      --outdir output/smm_runs/2026-09-11_142509_exp16
```

Sixteen parameters — eleven parent (`phi_2, phi_3, lambda_2, R_0, sigma_1_0, sigma_1_1,
sigma_2_0, sigma_2_1, sigma_4_0, sigma_4_1, sigma_eta`) and five child (`kappa_0,
kappa_theta, kappa_ParEd, kappa_terminal, sigma_eps`) — against seventeen moments (ten
parent + `k0_complete, kth_ga17_gap, kpe_g0_c, kpe_g1_c, kterm_x_strict_w99, kse_w_gap,
sd_ga17`). **`R_1 = 0`, fixed and excluded** (asserted in `moments.jl`, recorded in
`run_record.toml` and `estimates.toml`). Spec version `smm16_tas7_gap_v1`. Warm start:
the fourteen 2026-09-10 estimates by name; `sigma_eta = 0.03`, `sigma_eps = 0.5` from
`SMM_START`. Budget: 1,000 Sobol + the seeded incumbent, then up to 5 × 500 local
evaluations (Nelder-Mead), no polish. One evaluation: 27.6 s on the master at grid 30 /
simN 2000 (child cache warm); the local stage is sequential, so the worst case is ~19 h.

## 2. Targets: what changed and what did not

The fourteen targets shared with the baseline are identical to the digit (means and
clustered SEs); `m_psychic`, the wealth cut and the school schedule are identical. Three
targets are new. Every one of the 32 rows in the supplied `SMM_TAS_Moments.csv`
reproduces from the microdata to 1e-6 in estimate and SE, and the targeted-TAS block of
the supplied `SMM_TAS_VCov.dta` to 2×10⁻⁵ relative (`tools/test_smm_target_generator.py`,
28/28 with the order check).

| new target | data | SE | definition |
|---|---|---|---|
| `kth_ga17_gap` | 0.0312 | 0.0031 | mean ln g_ACH, completers − non-completers, age-17 frame (N 317) |
| `kse_w_gap` | 36.84 | 7.49 | mean winsorised parental net worth (10k USD), completers − non, wealth frame (N 665) |
| `sd_ga17` | 0.0329 | 0.0019 | sample SD of ln g_ACH, age-17 frame |

The Stata rerun's other changes — twelve `*_assets_*` columns dropped from the by-age
files, parental net worth moved to `SMM_Assets_ByChildAge.dta` in **two-year bins**
(`age_bin` = lower edge, inferred from the counts; undocumented in the codebook) — are
absorbed in the generator's CSV exports and touch no target.

## 3. The model change, verified

`log HC_{t+1} = log F_t + sigma_eta · z_{t+1}`, `z ~ N(0,1)` i.i.d., zero mean in logs
(conditional mean in levels × exp(σ²/2) = 1.00045 at 0.03), applied to every family-stage
transition including the age-18 handoff, drawn from a dedicated stream (`seed + 4`) so
the four existing streams are untouched. Integrated in the parent's continuation once per
period on the HC grid (`eta_expected_interp`) and by five-node quadrature on the terminal
spline (`eval_child_value_eta`, chain-rule factor on dV/dHC).

* **Exact at zero**: `tools/test_smm_baseline.jl` re-evaluates the 2026-09-10 fit at
  `sigma_eta = 0, sigma_eps = 0.5`: Q = 427.0098967752 vs the run's 427.0098967025
  (2×10⁻¹⁰ relative) and all 17 logged moments to 4 decimals. At the code level the
  zero case returns the untouched interpolant and a single spline call — not a
  degenerate quadrature.
* **Gradients**: central finite differences at `sigma_eta = 0.05` agree with the
  analytic gradients in the terminal, interior-full and parent-only objectives at three
  states (`tools/test_hc_process_shock.jl` §3, 36/36).
* **Effect**: on the test grids, SD log HC₁₇ 0.012 → 0.035 at 0.03; mean log HC₁₇ moves
  by −0.001; decisions and values move; two builds agree bit for bit.
* **`sigma_eps`** is a struct field built in exactly one place (`t_grid`), validated,
  part of `CHILD_ESTIMATED` and the tier-B cache key; the cached work/graduate blocks are
  eps-free (cache parity at `sigma_eps = 0.8` bit-identical, `test_smm_tas.jl` §10).

## 4. Baseline and warm start under the revised moments and weights (grid 30 / simN 2000)

| moment | data | baseline (σ_η = 0) | t | warm start (σ_η = 0.03) | t |
|---|---|---|---|---|---|
| k0_complete | 0.3226 | 0.2150 | −9.4 | 0.1980 | −10.9 |
| kth_ga17_gap | 0.0312 | 0.0086 | −7.2 | 0.0151 | −5.1 |
| kpe_g0_c | 0.2108 | 0.0000 | −16.5 | 0.0000 | −16.5 |
| kpe_g1_c | 0.6013 | 0.7061 | +5.7 | 0.6502 | +2.7 |
| kterm_x_strict_w99 | 33.20 | 23.75 | −2.9 | 23.80 | −2.9 |
| kse_w_gap | 36.84 | 6.62 | −4.0 | 6.48 | −4.1 |
| sd_ga17 | 0.0329 | 0.0081 | −13.3 | 0.0347 | +1.0 |
| **Q** | | **648.4** | | **451.7** | |

Parent block unchanged in both (|t| ≤ 1.3). Diagnostics at the warm start: ability
tertiles 0.123 / 0.210 / 0.261 (data 0.113 / 0.245 / 0.629); wealth tertiles 0.010 /
0.091 / 0.492 (data 0.207 / 0.357 / 0.617). SD of log HC by age, model at σ_η = 0.03:
flat at 0.033–0.043 from age 3 to 17; data: 0.083 at 3 falling to 0.02–0.03 from age 9.
**The model reproduces the adolescent floor and not the early-childhood dispersion** —
the initial draw's SD (0.067 at age 1) decays under σ₃ = 0.41 by age 3. That is a
finding for the advisor, not something this run can fix.

## 5. Identification (`tools/check_jacobian_rank.jl`, grid 16, simN 400, seeds 1234 / 20260910, steps 2 / 5 / 10 % of box)

* **Interior-completion point, 2 % step: full rank 16/16 on both seeds**, condition
  6×10⁴ – 8×10⁴, smallest singular value 0.11 – 0.14. The weak direction is a mix of
  `kappa_theta`, `kappa_ParEd` and `sigma_eps`.
* **Warm start, 2 % step: rank 15/16** — the `sigma_eps` column is numerically zero on
  one seed (norm 0.29 vs 30–7,000 for the others): with g0 = 0 and five Hermite nodes, a
  2 % move in log σ_ε flips no enrolment decision. At 5–10 % the column is finite and
  rank is full over the finite columns.
* **`R_0` produces an invalid (empty-completion-group) evaluation at 5 % and 10 %
  steps** on every point: a 5 % move on its [0.5, 100] log box is ×1.3 in productivity.
  That is the step, not the model; those columns are excluded and reported, not zeroed.
* **`sigma_eta`** is strongly identified: `sd_ga17` moves by ~24,000 data SEs per full
  box, an order of magnitude above anything else; |cos| with `kappa_theta` 0.04 – 0.63.
* **`sigma_eps` is the weakest column at every point and step**, and its identification
  does not come from `kse_w_gap`: its largest sensitivities are `kth_ga17_gap`,
  `kpe_g1_c`, `kpe_g0_c`, `k0_complete`. |cos| with `kappa_ParEd` reaches 0.86 and with
  `kappa_theta` 0.99 (warm start). **The wealth gradient is not separating the taste
  scale from the psychic-cost levels at these points.** The fallback named in the spec —
  the wealth tertiles, already in the export — should be tried if the run leaves
  `sigma_eps` on a wall or with a flat profile. Recorded as unresolved.

## 6. Numerical checks

* Grid caps respected: 30 asset/HC nodes, 5 shock nodes (`Nt`, `Neta`, `Np`); the child
  constructor now refuses `Nt > 5` and the parent `Neta > 5`.
* 3 vs 5 HC-shock nodes at fixed parameters: worst parent-moment difference 0.001 data
  SE at σ_η = 0.03; **0.118 SE at the box top 0.08** (`mean_hc_late`) — marginally above
  the 0.1 screen. Unresolved at the top of the box; irrelevant near the start.
* Support: simulated HC stays within 330–595 (σ_η = 0.03) and 283–713 (0.08) of the
  [50, 1500] grid; no simulated state leaves it; quadrature mass beyond `hc_max` from
  the highest simulated HC is 0. Only the grid's own edge nodes see extrapolation.
  `hc_max = k_max = 1500` on both sides of the handoff, unchanged.
* Value/gradient consistency at boundaries: the η-smoothed PCHIP extrapolates linearly
  with the boundary slope (consistent pair); the terminal spline is clamped in value and
  gradient together at every node.

## 7. Tests run

| test | result |
|---|---|
| `tools/test_smm_target_generator.py` | 28/28 |
| `tools/test_hc_process_shock.jl` | 36/36 |
| `tools/test_smm_tas.jl` (15 sections, two new) | 153/153 |
| `tools/test_smm_baseline.jl` (rewritten: reproduces the 2026-09-10 fit at σ_η = 0) | 27/27 |
| `tools/test_smm_target_paths.jl` | 13/13 |
| `code/smm/selftest.jl` | ALL CHECKS PASS |
| `tools/test_smm_resume.jl` (16-param fixture; old spec, unknown flag, out-of-box and missing `--init-from` refused) | 19/19 resume refusals + 3/3 flag cases (after two fixes to the test's own fixture and command construction) |
| runner smoke (`--quick --serial --report-only --init-from --skip-polish`) | exit 0 |

## 8. Notebook

`code/transfer_CRRA_wage.ipynb` was not executed and was not edited. Every new
constructor argument defaults to the baseline (`sigma_eta = 0`, `sigma_eps = 0.5`,
`Neta = 5`); it reads `a_p` and `child_age` from the by-age CSV, whose schema is
unchanged (`a_p` now the two-year-bin mean). Static inspection only.

## 9. Open items, kept visible

1. The specification change (HC shock) is not advisor-approved.
2. `sigma_eps` is weakly identified and `kse_w_gap` is not what identifies it (§5).
3. Early-childhood HC dispersion (ages 3–8) is underpredicted by a factor ~2 even with
   the shock (§4) — the initial-draw / persistence side of the technology.
4. `kpe_g0_c = 0` at the warm start; whether the run moves it is the question this
   pilot exists to answer.
5. 3-vs-5-node sensitivity at the top of the `sigma_eta` box (0.118 SE).
6. P7c open: `kpe_g0_c` / `kpe_g1_c` remain either-parent targets.
7. Restarts that stop on `MAXEVAL_REACHED` (500) are **budget-limited**, not converged;
   the runner labels them so.

## 10. Progress snapshot

See the end of this file; updated by hand while the run was watched.

```
sobol    complete: 1001 evaluations, best Q 451.7, 10.6 min   (the warm start; no Sobol draw beat it)
restart   1/5    eval    10   this Q        2804   best Q       451.7    12.6 min
restart   1/5    eval    12   this Q       506.7   best Q       451.7    13.0 min
restart   1/5    eval    27   this Q   4.826e+04   best Q       451.7    16.2 min
restart   1/5    eval    28   this Q   1.481e+04   best Q       451.7    16.4 min
restart   1/5    eval    29   this Q       1e+06   best Q       451.7    16.6 min
```
