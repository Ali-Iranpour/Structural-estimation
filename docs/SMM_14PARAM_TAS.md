# The fourteen-parameter SMM with TAS moments

**Status 2026-09-10: implemented, validated, NOT yet estimated.** Everything below is
reproducible from the repository as it stands. No production estimation has been run.

Extends the ten-parameter parent-block SMM with four child parameters — `kappa_0`,
`kappa_theta`, `kappa_ParEd`, `kappa_terminal` — and seven TAS moments. The existing ten
parameters, ten moments, own-study definition and fixed school schedule are unchanged.

---

## 1. What was added

| Parameter | Principally informed by | Box | Start | Link |
|---|---|---|---|---|
| `kappa_0` | `k0_complete` | [-2, 5] | 0.0587 | level |
| `kappa_theta` | `kth_ga17_t{1,2,3}_c` | [-10, 0] | -0.0342 | level |
| `kappa_ParEd` | `kpe_g{0,1}_c` | [-3, 0] | -0.0070 | level |
| `kappa_terminal` | `kterm_x_strict_w99` | [0.5, 40] | 5.0 | log |

These are joint identifying relationships, not exclusive assignments: `kappa_terminal`
moves the college share strongly (a parent that values retained assets transfers less), and
every kappa moves every completion moment.

| Moment | Data | N | Clusters | Model counterpart |
|---|---|---|---|---|
| `k0_complete` | 0.322627 | 2,771 | 1,232 | share of resimulated children choosing college |
| `kth_ga17_t1_c` | 0.113208 | 106 | 98 | college share in model tertile 1 of `sim_hc` at child age 17 |
| `kth_ga17_t2_c` | 0.245283 | 106 | 98 | tertile 2 |
| `kth_ga17_t3_c` | 0.628571 | 105 | 97 | tertile 3 |
| `kpe_g0_c` | 0.210766 | 1,319 | 713 | college share where `BothCollege == 0` |
| `kpe_g1_c` | 0.601314 | 913 | 548 | college share where `BothCollege == 1` |
| `kterm_x_strict_w99` | 33.1977 | 737 | 486 | mean of `sim_a[:, T+1] - transfer` |

Entry (`k0_entry`), the unknown-education group (`kpe_gu_c`), the raw and
including-home wealth means, and the LW / age-18 achievement variants (`kth_lw17_t*_c`,
`kth_lw18_t*_c`, `kth_ga18_t*_c`) are computed and written to the target file as
**untargeted diagnostics**. All nine achievement variants reproduce the published
`SMM_TAS_Moments.csv` values exactly. Tertiles are cut *within* age group, so the age-17 and
age-18 panels are not comparable in level — only the gradient within a panel is.

---

## 2. The five measurement decisions

Taken 2026-09-10. Each foreclosed a different silent error.

**Completion, not entry.** The model's college path is binary and has no dropout: enrol,
study `t_college = 4` years, earn the graduate wage `E = 1`. Nobody enrols without
finishing, so the path *is* a completed degree. Matching it to entry (0.616) would compare a
mechanism that always pays the college premium against a population where 30 percentage
points of it never does. The codebook reaches the same conclusion independently.

**Parental education: use the supplied numbers, change nothing, fix later** (by
instruction). TAS `pared_col = 1` means *either* parent has 16+ years; the model's state is
`BothCollege`. These are different groups, and the reconstructed both-parents rates are
0.7861 / 0.2013 against the targeted 0.6013 / 0.2108 — a gap 50% wider than the one being
targeted. Recorded as **`docs/ERRORS.md` P7c**, with the three ways out costed. Until it is
taken up, the estimated `kappa_ParEd` is *not* the effect of parental education.

**Terminal wealth: winsorised at p99.** The raw moment is a mean of $429,803 on a median of
$49,243, an SD of $1.9m and a maximum of $41.3m — the top 1% alone moves the mean by ~$98k,
and the model's asset grid stops at $1m. This repository already winsorises consumption and
investment at p99 for the same reason. The cut is exported in model units and the model side
applies the *same* functional, `E[min(W, cut)]`, so the two are the same estimator. Negative
net worth (13.0% of the sample) is retained, not dropped.

**Achievement: rank-based tertiles.** The model's HC level is already targeted by
`mean_hc_late`; cutting the simulation at the data's absolute W-score cut points would fold
any level or dispersion miss into `kappa_theta`. Cutting the simulation at its own terciles
asks only what these moments exist to ask. Ties break by stable rank (`MergeSort`), so
simulation noise cannot shuffle tied households between tertiles.

**The psychic cost is recentred.** `kappa_0 + kappa_theta*(log theta - m_psychic)`, with
`m_psychic = 6.263397` frozen in the target file. Behaviourally neutral — the same device
the wage equation already uses — but uncentred, `log(theta)` has mean 6.26 and SD 0.035, so
`[1, log theta]` has a condition number near 180 and `kappa_0` must move ~180 units to undo
one unit of `kappa_theta`. **`kappa_0` is therefore on a new scale**: the legacy 0.2728
corresponds to 0.0587 here. `check_psychic_centring` errors if the two stop agreeing.

---

## 3. How an evaluation works now

The specified order, implemented in `run_pipeline` (`code/smm/moments.jl`):

```
solve child lifecycle and transfer problems
  -> initial child simulation            (demonstration; outputs ERASED)
  -> construct the terminal-value object (from the solved value functions)
  -> solve and simulate parents
  -> initialize children from simulated parent outcomes
  -> resimulate children
  -> calculate all moments
```

The demonstration simulation cannot leak: its `sim_college` and `sim_tr_init` are filled with
NaN before the parent block runs, and `model_moments` errors if the resimulation has not
refilled them. Validation group 6 runs the pipeline with and without it and asserts every
targeted moment is bit-identical.

Handoff, asserted exactly (validation group 5):

```julia
child.sim_a_init  .= parent.sim_a[:,  parent.T + 1]
child.sim_k_init  .= parent.sim_hc[:, parent.T + 1]
child.sim_bc_init .= parent.sim_k[:, 1]
```

Matching `simN` and a common `seed` are both asserted.

### The child solve is rebuilt, not reused — and it is nearly free

The old design solved the child once per process. That was exact only while every estimated
parameter was a parent parameter, and it is now false. **The guard was replaced, not
deleted.** The child solve has four stages and the four parameters do not touch all of them:

| stage | cost (grid 30, warm) | reads any of the four? |
|---|---|---|
| `solve_model_work!` | 6.31 s | **no** |
| `solve_model_college!` stage 1 (graduate life) | ~5.8 s | **no** |
| `solve_model_college!` stage 2 (study years) | ~0.5 s | `kappa_0`, `kappa_theta`, `m_psychic` |
| `optimal_transfer_*!` | 0.50 s | `kappa_terminal` |
| `terminal_value_spline` | ~0.00 s | `kappa_ParEd` |

`build_child_solution` caches the two invariant stages under a complete dependency key
(`child_config` — every non-estimated child setting, no estimated one) and rebuilds the
rest. **Verified bit-identical** to a full re-solve: `max |diff| = 0.000e+00` on
`sol_v_college`, `sol_c_college`, `sol_h_college`, with an identical NaN feasibility
pattern, at **16.9× the speed**. There is no tolerance to tune; the arrays are copied.

An evaluation costs ~1.2 s more than before, not ~12.9 s more — about **+7%**.

The cache stores solution **arrays, never a model object**. A cached model would carry
mutable `sim_*` state and any simulator touching it would contaminate later evaluations;
this makes that structurally impossible.

### One evaluation pipeline, five tools

`jacobian.jl`, `sensitivity.jl`, `profile_param.jl`, `grid_sensitivity.jl` and
`standard_errors.jl` each used to hold a private copy of `build_child_value()` and its own
residual built from `moment_scale`. Four copies of an objective is four chances for one to
drift from the one being optimised, and a Jacobian of the wrong objective is a diagnostic of
nothing. They now share `evaluate_at` / `residuals_from`.

---

## 4. Weighting and the covariance

`Q = sum_j w_j (m_j - mhat_j)^2` with `w_j = 1/se_j^2` — each residual in standard errors of
its own moment. With seventeen moments against fourteen parameters the system is
over-identified, `Q` cannot reach zero, and the weights decide the answer.

**The missing `SMM_TAS_VCov.dta` was reconstructed, not requested.** The codebook specifies
the estimator exactly (a ratio of means, clustered on `famclust`), which is a closed-form
influence function. It reproduces all seven published estimates *and* all seven published
standard errors to six decimal places. `tools/make_smm_targets.py` rebuilds it every run.

**The blocks overlap and the overlap is measured, not assumed.** 488 families appear in both
frames (1,794 parent, 1,481 TAS, 2,629 distinct). Influence functions from both micro files
are stacked on the shared `Fam_id` key. Across all 70 cross-block moment pairs the
correlation runs from **-0.0350 to +0.0453**, none above 0.05 in absolute value — so the two
*blocks* are close to independent in this target set.

**That does not justify the diagonal weight, and an earlier version of this document said it
did.** The correlations that a diagonal weight throws away are the *within*-block ones, and
they are large: the largest in absolute value is **+0.676**. The case for the diagonal
first stage is the ordinary one — it is a clean, well-conditioned first-stage weight, and a
two-step optimal weight estimated on 2,629 clusters can be noisy enough to move the estimate
more than the efficiency it buys. Whether it actually costs much here is an open question
that needs a second-stage comparison, not something the cross-block figure settles.

(The printed cross-block figure was itself wrong until 2026-09-10 — it divided by
`sqrt(se_i*se_j)` instead of `se_i*se_j`, roughly an order of magnitude too small. The
exported `cov` and `corr` matrices were always correct.)

The full joint 17×17 covariance is exported and is what standard errors, sensitivity and the
identification diagnostics use. `standard_errors.jl` inverts the **correlation** form, per
the codebook's warning that the raw form is ill-conditioned because the vector mixes
probabilities with dollars.

---

## 5. Target versus model, at the incumbent

Report-only run, `output/smm_runs/2026-09-10_151548`, grid 30, simN 2000.

| moment | model | data | gap | t | Q share |
|---|---|---|---|---|---|
| `mean_c_p` | 3.2281 | 3.1155 | +3.6% | 3.2 | 0.4% |
| `mean_h_p` | 0.3019 | 0.3073 | -1.8% | -2.5 | 0.3% |
| `mean_t_p_early` | 0.4548 | 0.4544 | +0.1% | 0.1 | 0.0% |
| `mean_t_p_late` | 0.3334 | 0.3333 | +0.0% | 0.0 | 0.0% |
| `mean_e_p_early` | 0.3432 | 0.3429 | +0.1% | 0.0 | 0.0% |
| `mean_e_p_late` | 0.3913 | 0.3911 | +0.1% | 0.0 | 0.0% |
| `mean_i_c_early` | 0.0393 | 0.0393 | -0.1% | -0.0 | 0.0% |
| `mean_i_c_late` | 0.0496 | 0.0496 | +0.0% | 0.0 | 0.0% |
| `mean_hc_early` | 6.0745 | 6.0737 | +0.1% | 0.2 | 0.0% |
| `mean_hc_late` | 6.2543 | 6.2589 | -0.5% | -3.3 | 0.4% |
| **`k0_complete`** | **0.0000** | 0.3226 | -100% | -28.1 | **32.9%** |
| `kth_ga17_t1_c` | 0.0000 | 0.1132 | -100% | -3.7 | 0.6% |
| `kth_ga17_t2_c` | 0.0000 | 0.2453 | -100% | -5.9 | 1.4% |
| `kth_ga17_t3_c` | 0.0000 | 0.6286 | -100% | -13.1 | 7.2% |
| `kpe_g0_c` | 0.0000 | 0.2108 | -100% | -16.5 | 11.3% |
| **`kpe_g1_c`** | **0.0000** | 0.6013 | -100% | -32.6 | **44.3%** |
| `kterm_x_strict_w99` | 15.5740 | 33.1977 | -53.1% | -5.4 | 1.2% |
| **Q** | | | | | **2405.18** |

The parent block fits (as it should — this is its fitted incumbent). **The college share is
exactly zero**, so all six completion moments miss by 100% and carry 98% of `Q`.

This is *not* a feasibility failure: every simulated parent clears the college asset
threshold (`col_min` = 2.28 model units against a minimum terminal asset of 10.03). It is
the value comparison — college is simply never preferred at the incumbent kappas — which is
exactly what estimating them is for.

Wealth accounting: pre-transfer assets 37.60, mean transfer 22.03, retained 15.57. The
target, 33.20, sits *between* the model's pre- and post-transfer values. Substituting
pre-transfer assets would have "improved" the fit while measuring the wrong object.

---

## 6. Identification

**Parameter and moment counts establish nothing.** `tools/check_jacobian_rank.jl` takes the
residual Jacobian by central differences at two points and three step sizes.

| point | step | rank | cond | smallest sv |
|---|---|---|---|---|
| incumbent (share 0) | 2% | **11/14** | Inf | 0 |
| incumbent | 5% | **12/14** | 1.2e20 | 5.9e-17 |
| incumbent | 10% | 14/14 | 8.2e4 | 0.085 |
| interior college | 2% | 14/14 | 1.4e4 | 0.618 |
| interior college | 5% | **14/14** | **1.1e3** | **6.87** |
| interior college | 10% | **14/14** | **922** | **7.78** |

**The model is locally unidentified at the incumbent.** With the college share pinned at
zero, none of the six completion moments responds to a small perturbation and the Jacobian
loses three columns. At an interior-college point it is full rank at every step size, and
rank, conditioning and the least-identified direction are stable between the 5% and 10%
steps — the 2% step is finite-difference noise on a simulated objective, which is what its
20× worse conditioning is showing.

The weakest direction is consistently **`kappa_theta` against `kappa_ParEd`** — the two
psychic-cost gradients trading off. `kappa_ParEd` is the weakest column throughout (44.6–51.9
against 120+ for the next weakest).

---

## 7. Validation

```bash
uv run --with pandas --with numpy python tools/make_smm_targets.py
julia --project=. tools/test_smm_tas.jl <targets.toml>          # 75/75 pass, ~76s
julia --project=. tools/check_jacobian_rank.jl <targets.toml>
julia --project=. code/smm/run_smm.jl --report-only --targets <targets.toml>
julia --project=. tools/test_smm_resume.jl <targets.toml>        # 17 refusal cases + control
julia --project=. code/smm/selftest.jl
julia --project=. code/smm/run_smm.jl --quick --sobol 8 --restarts 1 --procs 6
```

| group | what it rules out |
|---|---|
| 1 target reproduction | the target file disagreeing with the published moments; a covariance whose diagonal is not its own SE vector |
| 2 parameter routing | a child parameter being absorbed by a same-named parent field while the block keeps its default |
| 3 child-solution refresh | a kappa that changes nothing — a converged fit for a parameter that does not act |
| 4 cache-key completeness | the cache serving a stale college solution; a simulated array in the cache |
| 5 handoff arrays | a mis-sized or mis-ordered handoff |
| 6 demonstration isolation | the initial child simulation supplying a final moment |
| 7 determinism | a broken common-random-numbers path; the removed 3-argument entry points silently working |
| 8 centring neutrality | the reparameterisation changing behaviour |
| 9 wealth accounting | pre-transfer assets being substituted for post-transfer |
| 10 cache parity | the cached refresh differing from an INDEPENDENT full solve |
| 11 mutation isolation | simulating a returned child poisoning the cache for the next draw |
| 12 partial points | an omitted kappa falling back to the constructor's default instead of the SMM's |
| 13 `parent_extra` | a non-estimated parent setting failing to reach the constructor |

Existing checks: `test_smm_target_paths` 13/13, `test_smm_boundaries` 184/184 + 5/5,
`test_smm_own_study` 12/12 (its spec assertion was updated from 10/10 to 14/17 with the
parent/child split pinned as well). `test_smm_baseline` was **deliberately scoped to the
parent block on the old proportional scale** — its frozen `Q` is the only frozen reference
to the parent solve that exists, and re-pinning it to the new objective would discard it.

Smoke test: full Sobol → restart → polish → report → `estimates.toml` end to end.

Resume compatibility: a checkpoint must now match on `spec_version`, `source_sha`
(`child_lifecycle.jl` + `parent_family.jl` + `moments.jl` — the model source is part of the
objective now that child parameters are estimated), `moment_names` and `m_psychic`. A
missing field is itself grounds to refuse: "cannot verify" is not "compatible".

---

## 7b. Fixes from external review (2026-09-10)

An independent review found six defects. All are fixed and all now have regression
coverage; the two graded P1 would have produced confidently wrong diagnostics.

| # | Defect | Effect | Now covered by |
|---|---|---|---|
| P1 | `grid_sensitivity.jl` accepted `a_max` but never forwarded it — `run_pipeline` had no channel for non-estimated parent settings | every rung of the asset-ceiling sweep solved at the SAME ceiling, so the tool concluded "the asset grid does not move the moments" vacuously | test group 13; `solve_at` now reads `p.a_max` back and errors on a mismatch |
| — | `grid_sensitivity.jl` aborted on `Main.GS_AMAX = $am` — an interpolation outside a quote, in the `--reoptimize` branch. Julia lowers a top-level `if` whether or not the branch is taken, so it fired on every run. **Pre-existing, unchanged in `HEAD`**, so the tool had never completed | the sweep's output was written but the process always exited with an error | dead code removed; the ceiling now travels as an argument |
| P1 | the same file's summary computed `Δ / 1.0 / sqrt(w)`, and `sqrt(w) = 1/se`, so it MULTIPLIED by the standard error | a 0.001 move on a 0.01 SE reported 0.00001 residual units instead of 0.1 — always under the 0.01 "grid does not matter" threshold | fixed to `Δ * sqrt(w)`; the two bugs pointed the same way, which is why neither showed |
| P1 | resume checks read `haskey(ck, f) && <mismatch> && refuse(...)`, which ACCEPTS a checkpoint that simply lacks the field; `child_grid`, `sim_n` and `seed` were never compared at all | a checkpoint from a different numerical problem could hand its `Q_best` to a run that could not reproduce it, so every later restart was measured against an unbeatable number | `tools/test_smm_resume.jl`: 8 missing-field cases, 8 changed-value cases, plus a CONTROL that the untouched fixture is accepted |
| P2 | `build_child_solution` used `merge(cfg, ckw)` without completing omitted kappas | a partial point — the documented interface every diagnostic uses — fell back to CONSTRUCTOR defaults: `kappa_0 = 0.2728` (the legacy UNCENTRED value) and `kappa_terminal = 10.0`, against SMM defaults of 0.0587 and 5.0. The runner passes all fourteen and never hit it | test group 12 |
| P2 | the printed cross-block correlation divided by `sqrt(se_i*se_j)` | reported -0.0011/+0.0044 against a true -0.0350/+0.0453, and the documentation repeated it. The exported `cov` and `corr` were always correct | the generator now prints within-block correlations too, which are the ones a diagonal weight discards (max **+0.676**) |
| P2 | `selftest.jl` still asserted ten parameters and ten moments | those two checks failed under the new specification, so the self-test's closing verdict was permanently "do not run the estimation" | updated, plus routing and moment-order assertions |
| P2 | the LW and age-18 achievement diagnostics were documented but never built | the promised sensitivity checks did not exist | nine variants now exported, all reproducing `SMM_TAS_Moments.csv` exactly |

Three further improvements from the same review:

- **Cache keys no longer include `simN` or `seed`.** Neither can change a solution — the
  solvers read grids, parameters and the shock discretisation, never the draws — so
  including them produced cache misses, not stale hits. Every tool with a different `simN`
  was re-solving the same 12.1 s.
- **A second cache tier holds the FULL child solution**, keyed on the solve configuration
  and the four kappas. A finite difference in a *parent* parameter changes no kappa, so the
  study years and transfer stage were also being redone for nothing across the ten parent
  columns of a Jacobian. Bounded at two entries (~90 MB each, 20 processes).
- **Standard errors are rank-aware.** `inv(G'WG)` is exactly singular where the Jacobian is
  rank deficient, which is not hypothetical here — rank 11/14 at the incumbent. It is now a
  tolerance-explicit pseudo-inverse that reports `G'WG rank` and warns that the intervals
  are conditional on the unidentified directions.
- **`check_jacobian_rank.jl` sweeps simulation sizes and seeds** (default 400/1000 × two
  seeds). Full numerical rank on one draw of 400 households is not evidence of robust
  identification.

**What the fixed grid sweep now says.** With both bugs repaired, `a_max` 80 against 200 at
grid 14 / simN 400 moves the targeted moments by **2.04 residual units** (Q 2414.6 →
2437.1, terminal assets 37.92 → 35.98) and the tool reports *"the asset grid DOES move the
moments"*. The broken version reported 0.00000 for every moment and the opposite verdict.
The parent's asset ceiling is therefore a live specification choice, not a settled one, and
it should be swept properly (`--a-max 100,150,200,300,400` at the production grid) before
the estimation — Q is not comparable across ceilings.

**Not adopted:** the reviewer suggested a handoff-only simulation path for estimation —
skipping the child's 51-year lifecycle, which the seventeen moments do not need. It would be
a real saving, but the specification for this work explicitly requires the complete
simulation sequence inside each evaluation. That is a specification change and belongs with
the advisor, not in a review fix.

**Skipped by instruction:** the three P3 reporting items — the joint-vs-block finite-cluster
correction wording, the negative-share denominator in an annotation, and comments that
overstate what centring achieves (it improves conditioning; it does not create
identification).

## 8. Remaining limitations

Ordered by how much they could move a result.

1. **`kappa_ParEd` targets the wrong group.** EITHER-parent college in the data against
   `BothCollege` in the model. Open by instruction; `docs/ERRORS.md` P7c.
2. **`kappa_0`'s box is ~71% dead.** The college margin is nearly a step: completion is
   0.000 at `kappa_0 = 0.059` and 1.000 by -1.0. Everything above ~0 is a flat zero where
   the derivative is identically zero — and the 8-point smoke test walked `kappa_0` *up* to
   1.87, improving `Q` on other moments while all six completion moments stayed pinned.
   **Recommend narrowing to [-3, 1] before the production run.**
3. **`kappa_ParEd` saturates.** `g1` reaches 1.000 at about -0.30 and `Q` is flat below it,
   so the optimizer will slide to whatever wall it is given. **Recommend [-1, 0].**
4. **`Q` is 98% TAS.** `kpe_g1_c` (44%) and `k0_complete` (33%) dominate, so the first
   estimation is essentially a fit to the college margin and the parent block is nearly
   unconstrained during the search. `report_fit` prints the concentration on every run. If
   the parent moments drift, the answer is a deliberate scale decision, not a wider box.
5. **The terminal-wealth timing gap is uncorrected.** The data is measured at a median child
   age of ~29, a median 4 years (up to 16) after qualifying independence; the model's object
   is assets at the transfer, child age 18. The model has no post-separation parent to age
   forward, so this cannot be closed without a new mechanism.
6. **The model cannot reproduce negative retained assets.** 13.0% of the qualifying sample is
   negative; `delta_P` floors the model at zero. Recorded, not censored.
7. **'By age 25' is an approximation** — a child last observed at 23 or 24 contributes a zero
   not actually observed through 25.
8. **The last CDS assessment is not demonstrably pre-entry for every child**, and tertiles
   are cut within age group, so only the within-panel gradient is meaningful.
9. **The covariance conditions on the fitted PCA weights and the sample cut points**,
   treating them as fixed. Propagating that needs a family-cluster bootstrap; not done.
10. **Support flags cover parents *or other relatives*** — parent-only support cannot be
    separated in these data.

11. **The parent's asset ceiling is unsettled.** Now that the sweep works, `a_max` 80 vs
    200 moves the moments by 2.04 residual units. Sweep it at the production grid and fix
    a ceiling before estimating; Q is not comparable across ceilings.

**Do not run a production estimation before deciding items 2, 3 and 11.** The first two are one-line box changes,
both measured; the third is one sweep. All three otherwise waste or misdirect the run.
