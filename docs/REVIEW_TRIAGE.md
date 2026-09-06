# Estimation review — current status and remaining work

## Launch-readiness audit — 6 September 2026

**No: the comments through Tier 2 are not all fully resolved.** The previous assessment
below overstates completion. Several fixes reproduce, but there are still executable bugs
in validity checking, acceptance, resume, derivatives, inference and the notebook. A
completed pilot or a delivered script is not equivalent to a validated fitted result.

This audit covers `HEAD = 1b0211a` **plus the existing working-tree edits**, notably the
new `R_0` search box `[0.5, 100]`. It updates this document only; it does not implement the
newly identified fixes or launch the requested estimation. Existing code edits are retained.
The previous pass's measurements remain evidence for their stated calibration and settings,
not certification of every point in the current search box.

**Recommended order:** close A1–A5 before the unattended production run; obtain a fitted
baseline; close A6–A10 before using the relevant inference, sensitivity or notebook outputs.
The Tier 2 analyses that require a fitted baseline necessarily follow that baseline run.
Tier 3 performance work is not a prerequisite. The historical tier numbering differs from
the previous implementation summary, so the audit checks both sets of items.

### What was verified in this audit

| Item | Current evidence and verdict |
|---|---|
| HC residual scaling and age coverage | `moment_scale` gives log-HC residuals scale 1; both sides use ages 3–9 for early HC. Equal-age target means regenerated from the local microdata match all ten saved means exactly. |
| Clustered data-moment covariance | Regenerated all entries of `[moment_cov].cov`; maximum absolute difference from the saved target file is **0**, with **1,633 families**. This verifies reproducibility, not every assumption needed for inference. |
| Parent calibration and grid diagnostics | Fresh solve with parent `Na=Nhc=30`, `Nk=2`, child `Na=Nk=30`, `Nt=5`, `simN=2000`, seed 1234: **0 reported domain violations**. `mean_c_p=3.702105983`, mean parental time `0.358540878`, `mean_e_p=0.334351662`. Assets: 2 households above the ceiling initially, 7 at handoff, maximum **259.0383** against 100. HC range **297.6691–1103.7585**, within `[50,1500]`. |
| Expected continuation and combined value/gradient | `continuation_selftest()` passes at 2,000 random points: combined-call difference **0**, expected-continuation difference **1.332e-15**. Performance timings below were not re-benchmarked in this audit. |
| Parental-education continuation | Family coefficient and offset-before-max are present; parent type selects the corresponding surface. At the full child grid, finite `V(bc=1)-V(bc=0)` differences take **19 distinct values** after rounding to 10 decimals. The earlier six-value measurement used a smaller grid. |
| TikTak optimizer ownership and threading guard | Separate local `Opt` objects and the `parallel=false`/`batch>1` rejection remain implemented. |
| Full-grid result storage and legacy script | Runner stores `Q_search` and `Q_final` separately and checkpoints the final point. Legacy `code/smm.jl` is archived. Resume of that final checkpoint and the notebook repairs remain incomplete; see A3 and A10. |

Checks used Julia 1.11.9, one Julia/BLAS thread per diagnostic process, and the existing
Python environment. The registered Julia MCP tool was unavailable in this session, so
bounded CLI diagnostics were used. Synthetic checks below execute the actual source
functions/acceptance expression with controlled inputs; they are not estimation results.
The full sensitivity exercise, production search and complete notebook were not executed.

### Before the production rerun

**A1 — OPEN, validity: terminal human capital escapes the acceptance check.**
[`moments.jl`, `simulation_violations`](../code/smm/moments.jl#L381) checks assets over
`1:T+1`, but checks HC only over `1:17`. On the freshly solved calibration, injecting
`sim_hc[1,18] = -1.0` or `NaN` gives **zero violations**. The same negative value at
column 1 is caught. Consumption NaN/negative values, invalid hours and negative handoff
assets were also caught as expected. The omitted HC column is the state passed to the
child. **Required:** validate finite, strictly positive HC over all `T+1` state columns;
keep flow checks over `1:T`. Add handoff-specific regression cases and require final
objective/report agreement. Off-grid coverage diagnostics do not substitute for validity.

**A2 — OPEN, acceptance: convergence is not tied to the returned winner.**
[`run_smm.jl`, `ACCEPTED`](../code/smm/run_smm.jl#L783) only requires *some* local restart
to converge, no TikTak exceptions, and no reported final violations. A worse restart can
converge while the winning restart and improving polish both stop at `MAXEVAL_REACHED`.
Refinement failure/return status does not enter the gate, and a refinement exception is
not added to `result.n_exception`. Executing the current acceptance expression on that
controlled case, including `REFINE.status=:failed`, returns **true**.
**Required:** preserve which stage/restart produced the winner and its termination status;
verify convergence at the retained final point on the report grid, finite non-penalty Q,
full validity, and all-stage exception status. A converged final polish should also be
recognized even when no earlier restart converged. Test both false acceptance and false
rejection cases. More Sobol points or restarts alone do not repair this gate.

**A3 — OPEN, resume: continuation is only conditionally exact, and history is lost.**
[`run_smm.jl`, `checkpoint!` / `load_resume`](../code/smm/run_smm.jl#L492) saves `stage`
and `objective_grid` but does not read either when resuming. A final/refined checkpoint is
always fed back into the search-grid TikTak/polish path. A synthetic saved grid-30
objective was accepted into a grid-20 resume; the resulting `f` remained **−10** while
re-evaluating its saved point on the test objective gave **0.25**. This is a controlled
illustration of comparing objectives from different grids, not an observed model Q.

Additionally, [`tiktak.jl`](../code/src/tiktak.jl#L369) resets the trace, evaluation count
and exception count on resume. Resuming after the last restart produces an empty restart
trace, hence zero converged restarts, even if the original run converged. Earlier errors
can disappear from the verdict and cost totals. The loader checks seed dimension, restart
count and search grid, but not parameter names/order, boxes/links, targets, simulation
settings, child grid, solver settings or code identity. The changed `R_0` box has the same
parameter count and would pass those checks.

**Required:** implement stage-aware resume, retain distinct search/final checkpoints and
cumulative history, and validate a complete run configuration and input/code fingerprints.
Save restart-0 state as soon as Sobol survivors are saved: currently an interruption before
the first restart checkpoint still loses a usable resume. Test interrupted versus uninterrupted
runs, final-stage resume, changed-target/box rejection, and preservation of exceptions and
convergence counts. **Do not resume an old run into the new `R_0` box.**

**A4 — OPEN, error handling: generic coding errors can be scored as infeasible draws.**
[`moments.jl`, `smm_objective`](../code/smm/moments.jl) and
[`sensitivity.jl`, `sens_objective`](../code/smm/sensitivity.jl) catch every
`ErrorException`, `AssertionError`, `DomainError` and `InexactError` of those types.
Consequently the earlier claim that unexpected bugs are always rethrown is too strong.
Replacing the solve with `error("injected unexpected programming bug")` makes the actual
SMM objective return **1e6** and increment only `:ErrorException`; TikTak cannot count an
exception that never reaches it. **Required:** distinguish expected model/solver failures
with explicit failure types or narrowly classified conditions; preserve representative
messages and stages; rethrow unexpected errors. Regression checks must cover both expected
infeasibility and injected programming errors, including wrapped NLopt exceptions.

**A5 — OPEN, run evidence: the saved result is not enough to reconstruct or assess the run.**
[`run_smm.jl`](../code/smm/run_smm.jl) writes aggregate return-code counts, but does not
persist the per-restart trace, per-restart evaluation counts, full solver budgets/tolerances,
parameter boxes/links, target snapshot/hash or working-tree patch. A commit marked `-dirty`
is not a recoverable copy of the code. `n_eval_total` includes optimizer refinement calls
but omits its initial full-grid evaluation; startup/report solves are also outside the
reported optimization count, while the penalty counters span a broader set of calls.

**Required before spending the long-run budget:** save an immutable configuration and
input/source snapshot or hashes with a recoverable code revision; persist each restart's
start/result, Q, return code, cost and winner provenance. Define optimization-only versus
all-solve costs explicitly, use matching penalty denominators, and preserve cumulative
values on resume. For the planned run, record **1,000 Sobol points, 30 restarts and the
actual local/polish budgets**, rather than relying on defaults remembered later.

### Complete before using the corresponding Tier 2 outputs

**A6 — OPEN, Jacobian: clipping a perturbation changes the denominator.**
[`jacobian.jl`, `jacobian_at`](../code/smm/jacobian.jl#L197) clamps proposed parameters to
the box but still divides by `2h`, and writes `difference="central"` unconditionally.
For the actual function with the linear residual `r(x)=x`, box `[0,1]`, centre `x=1`, and
step 0.01, the returned derivative is **0.5000000000000004**, although the exact derivative
is **1**. This matters particularly because the pilot already landed near a bound.
**Required:** divide by the actual separation in search coordinates, or implement explicit
one-sided formulas; save effective points/steps and difference type per column. Reject
invalid/non-finite perturbations instead of merely saving an `invalid_cells` warning and
continuing inference. Test level and log links at both bounds and in the interior.

**A7 — OPEN, inference: `--at` is ignored, the J-test is absent, and rank is not guarded.**
[`standard_errors.jl`](../code/smm/standard_errors.jl#L65) parses `ATFILE` but never uses it.
An invocation with a **nonexistent** `--at` file completes successfully and still reports
`PARENT_DEFAULTS` from the supplied Jacobian. Passing `--at` neither recentres the Jacobian
nor enables a J-test; the J-test section only prints explanatory text. To centre SEs on a
fit, first generate a **new Jacobian with `jacobian.jl --at <fit>`**, then supply that
Jacobian directory to the SE script. The current script also reads covariance from the
current targets without checking it against the Jacobian's original target provenance.

The sandwich directly inverts `G'WG` without a parameter-rank guard. Running it on the
saved **10-moment/11-parameter** Jacobian completed and wrote invalid inference: e.g.
`se_equal(phi_3)=0`, `se_equal(R_0)≈1.06e7`, and reported correlations far outside `[-1,1]`.
Clipping negative variance diagonals to zero hides numerical failure.
**Required:** reject underidentified/rank-deficient or materially non-PSD calculations;
use stable factorizations; validate fitted-point, grid, target and acceptance provenance.
Either implement the advertised `--at` validation/J-test correctly or remove the unsupported
claims. Retain explicit limitations on simulation error and bound-constrained inference.
The efficient-weight comparison is not evidence that an efficient-weight estimate was run.

**A8 — OPEN, sensitivity: resume and point-status evidence are incomplete.**
Static inspection of [`sensitivity.jl`](../code/smm/sensitivity.jl#L231) establishes:

- Completed CSV rows are skipped without restoring their parameter vector to `z_prev`.
  After interruption between +1 and +2 SE, +2 restarts from `Z0`, not the completed +1
  neighbour. That changes the warm-start path, and also the seed passed to the alternative
  search. Rows therefore are not independent of the restart history as the comment claims.
- Resume does not validate the baseline, boxes, targets/scales, grids, seed or budgets;
  `meta.toml` is written only at the end and overwritten after resume. A partial CSV row
  is not checked for completeness before being marked done.
- If the alternative wins, `ret` records `alt.polish_ret`, even when that polish did not
  produce the retained point. Alternative restart failures/exception counts are discarded.
  `MAXEVAL_REACHED` and on-bound rows are still saved and skipped on future resume.
- `--report-grid` is parsed into `GRID_RPT` and never used. At `--grid 20`, the reported
  curves remain grid-20 results; there is no full-grid reporting/refinement stage.

**Required before the full exercise:** restore neighbour vectors; validate immutable metadata
at startup; recover incomplete rows; preserve winner provenance, all return codes and
exceptions; explicitly mark incomplete/nonconverged points and support retrying them.
Implement the report-grid option or reject unsupported differing grids. Include a solved
zero-offset baseline/check so optimizer variation can be assessed against perturbation effects.
The four-point pilot remains a useful machinery test, not validated response curves.

**A9 — OPEN, comparable fitted results: there is no complete comparison command.**
The previous phrase “`--at` everywhere” is incorrect: `run_smm.jl --report-only` always
reports `incumbent()` and has no `--at` parser; unknown flags are not rejected. Low-level
`report_fit(z, ...)` can re-evaluate a supplied vector, but no saved old/new comparison has
been demonstrated. **Required:** add a checked report/comparison path that loads complete
parameter vectors, holds corrected targets/grids/seeds fixed, saves moment residuals and
Q, and detects parameters outside the current box. `unpack` clamps to that box, so naively
feeding an old out-of-box estimate through it would evaluate a different parameter vector.
Reject unsupported CLI flags rather than silently ignoring them.

**A10 — OPEN, original Tier 2.4: notebook counterfactual calls still use removed keywords.**
[`transfer_CRRA_wage.ipynb`](../code/transfer_CRRA_wage.ipynb), code cell ID `5fb821e6`
(zero-based cell 29), still constructs `model_phi_high_1`, `model_phi_high_2` and
`model_phi_high_3` with `phi_1_0`, `phi_2_0`, `phi_3_0`. The current constructor accepts
`phi_1`, `phi_2`, `phi_3`. These are three live calls, not Markdown or commented code;
the historical “14 call sites” is not the current remaining count. The hardcoded
“unchanged” values also disagree with `PARENT_DEFAULTS`. **Required:** use current keywords
and current baseline values for unchanged parameters; check that each intended
counterfactual actually increases/decreases the named quantity. Then validate those cells
and update the stale notebook parameter table. This does not block `run_smm.jl`, but it
prevents declaring all original Tier 2 work complete.

### Numerical choices and the planned 1,000 / 30 run

- **Keep the current nine-parameter baseline explicit.** Choosing not to add a tenth
  parameter is a recorded baseline decision; evidence for freeing one is still conditional.
- **The `R_0` box changed.** All four saved identification directories used `[5,300]`,
  whereas the working code uses `[0.5,100]`. Their box-scaled condition numbers are not
  measurements under the new box. Recompute at the fitted point with the new bounds and
  consistent steps. Do not describe the old artifacts as a current-box audit.
- **Asset-tail accuracy remains unvalidated.** The correct diagnostics reproduce; that
  does not prove extrapolation harmless. Historical Tier 2.2's initial-wealth clamp was
  deliberately rejected because it changes the draw. Record a numerical grid-range/node
  placement sensitivity check within the 30-node cap, including the handoff. Merely
  widening a numerical grid is not automatically advisor-gated: `CLAUDE.md` explicitly
  permits numerical grid-bound/interpolation changes. Changes to the economic distribution
  or specification, and interpretation of results, are separate decisions.
- **Bounds are diagnostics, not automatic proof of nonconvergence.** The runner's 2%-of-box
  flag means “near a bound”; it does not establish that a constrained optimum failed to
  converge or that the box must be widened. Assess constrained optimality and numerical
  stability, and use inference appropriate to a binding constraint. Likewise `ftol_abs`
  is in objective units, not scale-free as claimed below.
- **After fitting:** repeat identification/step stability and grid coverage at the actual
  estimate, calculate validated uncertainty, run the full sensitivity exercise, compare
  fitted vectors under identical definitions, and retain the parental-time/leisure and
  parental-education interpretation caveats before results circulate.

Once the pre-run items above are closed, the requested fresh baseline command is:

```bash
cd /srv/project/speech/apps/Structural-estimation/code/smm
julia +1.11 --threads=1 --project=../.. run_smm.jl \
    --sobol 1000 --restarts 30 --grid 30 --procs 20 \
    --local-evals 2000 --polish-evals 4000
```

This selects the full search/report grid, the existing production simulation size of
2,000, and process-based Sobol parallelism. There are **1,000 Sobol evaluations plus one
incumbent seed**; the 30 local restarts are sequential. No `--quick` or old-run `--resume`
is intended. The 20-worker setting is the existing shared-server budget.

**Runtime is not certified by the old pilot.** At the earlier 8.79 seconds/evaluation and
165 evaluations/restart heuristic, this is about **12.2 hours** with 20 Sobol workers.
At the stated local/polish caps it could approach **156 hours** at that same evaluation
speed, before startup/report overhead. These are budget illustrations, not a fresh timing
measurement or guarantees; inspect actual termination traces and budget the job accordingly.
Increasing restarts does not establish winner convergence, identification or Tier 2 completion.

---

## Prior implementation assessment — 6 September 2026, code at `dca7980` + that pass

**The launch-readiness audit above supersedes this assessment wherever completion is
claimed.** This earlier pass implemented substantial portions of Tiers 0, 1 and 2, but
several claims of complete validity, resume and inference were not sustained by the later
audit. Tier 3 was deliberately left untouched in that pass (instruction, 2026-09-06).
The measurements and implementation account below are retained as a record, not a launch
checklist.

Three things to read before anything else:

1. **The parental-education correction changes the baseline.** It is a correctness fix, not
   a new specification — every other `kappa_ParEd` channel already operated and only the
   terminal continuation was evaluated as if `bc = 0` — but the numbers move.
   **Flagged for Sahber before any result built on it circulates.** The before/after below
   is a preliminary calibration check, not an estimate.
2. **There is still no completed nine-parameter estimation.** Every number here is measured
   at the incumbent calibration or on a pilot. Identification and sensitivity work centred
   on a calibration describes the calibration's neighbourhood, not the estimator's; both
   must be re-run centred on the fitted baseline once one exists.
3. **The identification numbers are now artefacts, not recollections.** `jacobian.jl` saves
   the matrix, the point, the steps, the boxes, the scales, the grids and the seed. One of
   the previously circulated numbers does not reproduce — see *New findings*.

---

## What was fixed in this pass

### Tier 0

| Item | Status | Evidence |
|---|---|---|
| Full simulation validity | Was fixed at `dca7980`; re-verified | 0 invalid cells at the incumbent, grid 30, `simN` 2000 |
| Asset **and HC** grid coverage | Fixed (diagnostic); grid range remains a specification question | per-period counts and maxima for both states, over all `T+1` columns |
| Final-solve acceptance | **Fixed** | return codes classified, persisted, and gated on |
| Parental-education consistency | **Fixed** | offset inside the max, family coefficient applied |
| Interpretation and stale explanations | **Fixed** | legacy entry point retired, not merely annotated |

**Parental-education consistency.** `terminal_value_surface(m; ip, bc)` now adds
`family_coef(m) * pared_value_offset(m, bc)` to the college branch **before**
`max(v_college, v_work)`. `family_coef(m) = (1 − mu) + mu*omega` is defined once
([`child_lifecycle.jl:408`](../code/src/child_lifecycle.jl#L408)) and applied at the two
sites where the offset is added to a **family-weighted** `sol_tr_v_college`
([`child_lifecycle.jl:1632`](../code/src/child_lifecycle.jl#L1632),
[`parent_family.jl:1888`](../code/src/parent_family.jl#L1888)) — and deliberately **not** at
[`child_lifecycle.jl:1431`](../code/src/child_lifecycle.jl#L1431), where it is added to the
child's own value and is already in the right units. That distinction is now written into
the code at all three sites so it cannot be "tidied" into consistency.

The parent needs no new state: its `k` **is** the `BothCollege` indicator, so
`terminal_value_spline` returns a `ChildTerminalValue` holding one surface per `bc`, and
`eval_child_value` selects on the parent's own `k`. The notebook's plotting path
(`V(a,hc)`, `Dierckx.derivative(V, …)`) forwards to the `bc = 0` surface, which is the
surface those figures already showed, so no notebook cell changes behaviour.

Measured, child grid 12×12×3:

| | |
|---|---|
| `V(bc=1) − V(bc=0)` | takes **6 distinct values**, not one — a pure level shift would be constant, so the max genuinely moved |
| enrolment cells flipping work → college | **2 of 432** |
| family coefficient at the current calibration | 0.650; raw offset 0.026765, weighted 0.017397 |

Effect on the parent block, grid 30, `simN` 2000 — **preliminary calibration check, not an
estimate**:

| | before | after |
|---|---:|---:|
| mean terminal HC | 884.99 | 885.95 |
| `mean_c_p` | 3.702224 | 3.702106 |
| `mean_t_p` | 0.358460 | 0.358541 |
| `mean_e_p` | 0.334173 | 0.334352 |

**Final-solve acceptance.** `TikTakResult` now carries `polish_ret`, `polish_improved` and
`n_eval_polish`; `ret_class` buckets an NLopt code into `:converged`, `:limit` (a budget
stopped it, not a criterion) or `:other`; `ret_tally` counts them. `run_smm.jl` prints how
every local search ended, warns when more restarts hit a budget than converged, warns
loudly on any exception, and writes an explicit **acceptance** verdict:

```
accepted = (converged restarts > 0) AND (no objective exceptions) AND (0 invalid cells at the final point)
```

persisted alongside `n_converged`, `n_hit_budget`, `n_ret_other`, `n_exception`,
`n_invalid_final` and the full `ret_tally`. Verified by deliberately under-budgeting a run
(`--local-evals 25`): it reported `n_converged = 0`, `n_hit_budget = 2`,
`accepted = false` — a finite `Q` of 0.3736 and a clean simulation did **not** certify it.

**Asset and HC grid coverage.** HC was never measured at all; it has a floor as well as a
ceiling, and a state below `hc_min` is extrapolated as silently as one above `hc_max`.
`moment_diagnostics` now reports both states, at both ends, per period, over all `T+1`
columns, and `report_fit` prints an off-grid-by-period table. At the incumbent, grid 30:

| | |
|---|---|
| households ever above the asset ceiling | **7 of 2,000** (2 at `t=1`, 7 at the handoff → **5 cross during the family stage**) |
| maximum simulated assets | **259.0** against a ceiling of 100 |
| human capital | entirely inside `[50, 1500]`; nothing off-grid at either end |

The initial asset draw is still **not** clamped, deliberately — clamping would distort the
wealth distribution to flatter a grid.

**Legacy entry point retired.** `code/smm.jl` → `archive/smm_14param_legacy.jl`, which is
where this repo keeps superseded code. Verified beforehand that nothing referenced it —
no notebook cell, no script, no doc except as a historical note. References in `CLAUDE.md`
and `code/smm/README.md` updated.

### Tier 1

| Item | Status | Evidence |
|---|---|---|
| Checkpoint metadata and actual resume | **Completed** | the final post-polish winner is now always checkpointed |
| Refinement reporting | **Fixed** | four distinguishable outcomes, return codes, full cost accounting |
| Expected continuation and combined value/gradient | **Fixed** | 1.28–1.32× faster, verified as an exact identity |
| Allocations and budgets | **Done, with a measured stopping point** | hot loop is now allocation-free |

**Checkpointing.** The final winner used to be checkpointed only when the search and report
grids differed — and both default to 30, so on a **default** run the last checkpoint was the
pre-polish incumbent and the polish's improvement lived only in `estimates.toml`. It is now
written unconditionally, with `stage = "final"` when the grids coincide and `"refined"`
when they do not.

**Refinement reporting.** `refined = GRID_SEARCH != GRID_FULL` said only that a stage was
*selected*; it read `true` after an exception and after a refinement that found nothing.
Replaced by `refine_status` ∈ `skipped | improved | no_improvement | failed`, plus
`refine_ret`, `n_eval_refine`, `n_eval_polish`, and `n_eval_total` — `result.n_eval` covered
TikTak only, so the refinement's evaluations were free in the reported cost. Verified:
`refine_status = "no_improvement"`, `refine_ret = "MAXEVAL_REACHED"`, `n_eval_total` 97
against `n_eval` 91.

**Expected continuation and combined value/gradient.** Two changes, both exact:

- `expected_interp` integrates over next period's shock **once per period** instead of once
  per objective call. For a fixed evaluation point the Hermite blend in `hc` and the
  bilinear blend in `(a,k)` are linear in the node arrays, so `Σⱼ πᵢⱼ Pⱼ(x) = P̃(x)` with
  `Ṽ = Σⱼ πᵢⱼ Vⱼ` and `D̃ = Σⱼ πᵢⱼ Dⱼ`. **The stored slopes are averaged, never refitted** —
  `_pchip_slopes` is nonlinear, so refitting from averaged values would be a different
  interpolant. The Fritsch–Carlson bound survives averaging.
- `value_and_gradient` returns both from one pass; the solver always wants both, and
  `P(x)` followed by `gradient(P,x)` located the same cell and evaluated the same four
  Hermite corners twice.

Verified by `continuation_selftest()` on 2,000 random points with random node values:
`value_and_gradient` differs from the separate calls by **exactly 0**, and `expected_interp`
from the explicit loop by **1.3e-15**.

| | before | after | |
|---|---:|---:|---|
| solve + simulate, grid 20 | 5.03 s | **3.81 s** | 1.32× |
| solve + simulate, grid 30 | 11.25 s | **8.79 s** | 1.28× |
| allocations, grid 20 | 1.60 GB | 1.40 GB | −12% |
| allocations, grid 30 | 3.57 GB | 3.13 GB | −12% |

**Allocations, and where the profiling stops.** `grad[:] = -grad[:]` materialised a
temporary vector on every objective call, at six sites; it is now `grad .= .-grad`. After
that the hot loop is allocation-free — `obj_work_period_full` runs at **242 ns/call and
0 bytes/call**, `value_and_gradient` at **58 ns/call and 0 bytes**.

The remaining 3.13 GB is NLopt's own per-`Opt` machinery: the solver constructs ~153,000
`Opt` objects per grid-30 solve, one per state. Reusing them across states would mean
sharing NLopt state, which is the exact hazard behind the silent exit-0 crash this project
already paid for — and **GC is only 5.5–8.0% of solve time** (measured: 6.0% at grid 10,
5.5% at 20, 8.0% at 30), so the whole prize is under 8%. Not taken. The measurement is
recorded here so the decision does not get re-litigated from intuition.

Budgets: `--local-evals` and `--polish-evals` are flags, and the run now reports
converged-versus-budget counts, so a production budget can be chosen from traces rather
than assumed. The absolute stopping safeguards (`local_ftol_abs = 1e-10`,
`local_xtol_rel = 1e-8`) are unchanged — they are scale-free and cost nothing.

### Tier 2

| Item | Status |
|---|---|
| Identification audit and tenth-parameter decision | **Done as a reproducible artefact** — [`code/smm/jacobian.jl`](../code/smm/jacobian.jl) |
| Sampling uncertainty and weighting | **Done** — clustered moment covariance + [`code/smm/standard_errors.jl`](../code/smm/standard_errors.jl) |
| Target-moment response exercise | **Script delivered and validated end to end; full run deliberately not launched** — [`code/smm/sensitivity.jl`](../code/smm/sensitivity.jl) |
| Comparable final results | **Machinery delivered; the comparison itself needs a fitted run** |

**Identification audit.** `jacobian.jl` computes the residual Jacobian at a stated point and
saves it: the matrix as CSV per finite-difference step, and in `jacobian.toml` the
evaluation point, parameter order, boxes, links, moment scales, grids, seed, steps, singular
values, condition numbers, numerical rank, right singular vectors and every pairwise cosine.
It takes `--at <estimates.toml>` so the exercise can be repeated at a fitted baseline, and
`--extend sigma_4_1,mu_1` so candidate columns are compared at the same point with identical
scaling of the shared columns.

Measured 2026-09-06, incumbent, grid 30, central differences at 0.5 / 1 / 2 % of each box
width, columns scaled to a full-box move:

| columns | condition number | smallest σ | thin-SVD rank |
|---|---:|---:|---|
| 9 (current) | **51.2** | 0.266 | 9 of 9 |
| 10, adding `sigma_4_1` | **228.9** | 0.060 | 10 of 10 |
| 10, adding `mu_1` instead | **198.4** | 0.082 | 10 of 10 |
| 11, adding both | 183.4 | 0.089 | **10 of 11** |

`σ_min` is stable across the three steps (spread 6–7% of its level), so it is resolved above
derivative noise — the question the triage asked, now answered with a number. The
eleven-column row is the case the triage predicted: a thin SVD reports ten positive singular
values while one parameter direction is unidentified **by construction**. The script now
says so rather than letting ten positive values read as full rank.

**Sampling uncertainty and weighting.** `tools/make_smm_targets.py` now emits a
`[moment_cov]` block: the cluster-robust covariance of the ten targeted moments, clustered
on `Fam_id` over **1,633 families**, built from the influence functions of the equal-age
means — so it carries both the repeat-observation dependence and the overlap between moments
measured on the same households.

| | |
|---|---|
| moment standard errors | **2.1–7.7% of the cross-sectional SDs** (e.g. `mean_c_p`: se 0.0355 against sd 1.699) |
| moment correlations | up to **+0.676**; \|corr\| > 0.3 in 4 of 45 pairs |

That settles the review's warning concretely: per-observation SDs are 13–48× too large to be
moment standard errors, and a diagonal weight built from them would have been wrong in
magnitude *and* in shape.

`standard_errors.jl` combines a saved Jacobian with that covariance in the minimum-distance
sandwich `(G'WG)⁻¹G'WΩWG(G'WG)⁻¹`, under both equal and efficient weights, and reports what
it does not cover: simulation error (fixed seed, common random numbers — the usual `1+1/S`
inflation assumes independent draws and is not folded in silently), the weighting choice
itself, specification error, and anything global. It refuses to compute a Jacobian of its
own, so the derivative step and evaluation point always travel with the numbers.

At the incumbent (grid 30, step 1%) — **not a fitted point, so these are the machinery
working, not results**:

| parameter | estimate | se (equal W) | se (optimal W) |
|---|---:|---:|---:|
| `phi_2` | 0.14184 | 0.00569 | 0.00549 |
| `phi_3` | 1.00000 | 0.18089 | 0.14333 |
| `lambda_2` | 1.00000 | 0.17872 | 0.15225 |
| `R_0` | 81.55000 | 6.03364 | 4.82516 |
| `sigma_1_0` | −0.45750 | 0.20236 | 0.15918 |
| `sigma_1_1` | −0.06340 | 0.00791 | 0.00639 |
| `sigma_2_0` | −3.39554 | 0.22281 | 0.18623 |
| `sigma_2_1` | −0.02870 | 0.01061 | 0.00787 |
| `sigma_4_0` | −4.50000 | 0.16489 | 0.13510 |

The J-test is deliberately **not** computed at a calibration; `--at` a fitted
`estimates.toml` makes it meaningful.

**Target-moment response exercise.** `sensitivity.jl` implements it as specified: perturb one
target, hold the rest fixed, **jointly re-estimate all nine parameters**, one row per point.
Held fixed across every point: the moment scales (frozen at the *baseline* targets, so moving
a target does not also move its own weight), the equal weights, the random draws, the grids,
the bounds, the links and the solver settings. The child solve is built once per process and
reused. Perturbations are in **standard errors of the sample moment** from `[moment_cov]`;
the script refuses to run if that block is missing rather than substituting a per-observation
SD.

Starting points follow the instruction: each point warm-starts from its neighbour along the
offset ladder, and **every point also gets an independent multistart check**, with the gap
recorded as `alt_gap`. `curves.csv` *is* the checkpoint — rows are independent estimations,
so `--resume` simply skips what is already there.

**The pilot ran to completion — 4 points, 88.2 min — and its result is a finding about the
method, not a set of curves.** Two moments (`mean_h_p`, `mean_hc_early`) at ±1 se, grid 20,
40 Sobol + 2 restarts, 80-evaluation caps:

| moment | offset | Q | Q warm | Q alt | `alt_gap` | ret | on bound |
|---|---:|---:|---:|---:|---:|---|---|
| `mean_h_p` | +1 se | 0.3064 | 0.4078 | 0.3064 | 0.1014 | MAXEVAL_REACHED | yes |
| `mean_h_p` | −1 se | 0.3109 | 0.4107 | 0.3109 | 0.0998 | MAXEVAL_REACHED | yes |
| `mean_hc_early` | +1 se | 0.3027 | 0.4090 | 0.3027 | 0.1062 | MAXEVAL_REACHED | yes |
| `mean_hc_early` | −1 se | 0.3089 | 0.4096 | 0.3089 | 0.1007 | MAXEVAL_REACHED | yes |

**Read the last two columns before the first three.** Every point stopped on its evaluation
budget rather than a convergence test, and every point has a parameter on a box edge
(`sigma_1_0` sits at or against its upper bound of −0.2 in all four). These are not
converged optima, so **the parameter differences across points cannot be attributed to the
target perturbation** — the perturbations are tiny (±1 se on `mean_hc_early` is ±0.0038 on a
target of 6.07, a 0.06% move) while the parameters moved by up to 3.4% of their boxes.
Optimizer noise at a 363-evaluation budget is by far the more likely explanation. The pilot
produced no interpretable curve and was never going to; that is what a pilot is for.

Three things it did establish:

1. **The machinery works end to end** — frozen scales, ladder warm starts, the independent
   check, `curves.csv` as its own checkpoint, and the on-bound and return-code flags that
   are the reason the paragraph above can be written at all.
2. **The alternative-start check is not optional.** The warm start lost at **every single
   point**, by a consistent ~0.10 in `Q` (0.408–0.411 against 0.303–0.311). Warm-starting
   from the calibration walks into a worse basin systematically, not occasionally. Without
   the check, all four rows would have entered the curve as parameter responses.
3. **The full run needs far larger per-point budgets than the pilot's.** `MAXEVAL_REACHED`
   at every point means 80-evaluation caps are nowhere near enough; the main estimation
   needs ~165 evaluations per restart to converge. The documented full-run command below
   uses the non-pilot defaults (600 per local search, 8 restarts, 200 Sobol) — budget from
   the traces it produces, and treat any point that still reports `MAXEVAL_REACHED` or
   `on_bound` as censored rather than as a slope.

**The full run was not launched, by instruction**, and should be centred on the eventual
fitted baseline:

```bash
cd code/smm
julia +1.11 --project=../.. sensitivity.jl \
    --at ../../output/smm_runs/<fitted-run>/estimates.toml \
    --moments all --offsets -2,-1,1,2 --restarts 8 --sobol 200 --procs 20
```

Forty points, one full re-estimation each. **Pilot output is preliminary and is not
identification or robustness evidence.**

---

## New findings from this pass

These were not in the original review and change how some of it should be read.

**1. The 1067.1 condition number does not reproduce; the cosines do.** Pairwise cosines are
invariant to column rescaling and come out as reported — `sigma_4_0`/`mu_1` **0.991**,
`sigma_4_0`/`sigma_4_1` **0.814**, `sigma_4_1`/`mu_1` **0.807**. Condition numbers are *not*
scale-invariant, and under a stated box for `sigma_4_1` of `[−0.05, 0.05]` the nine-to-ten
comparison is **51.2 → 228.9**, a 4.5× degradation, not the 21.7× implied by 49.2 → 1067.1.
The direction of the conclusion survives; the magnitude was never reproducible because the
box it depended on was never recorded. This is exactly the qualification the triage
predicted, now demonstrated rather than argued.

**2. The worst-separated pair is one that is already estimated.** Among the nine estimated
parameters, `sigma_1_0` vs `sigma_1_1` is **0.908** — *higher* than the
`sigma_4_0`/`sigma_4_1` 0.814 that is the stated reason for leaving `sigma_4_1` out. `t_p`
and `i_c` are split at the same two age groups, so the argument that ages 6–9 and 10–17 are
too close to separate a level from a slope applies with more force to a parameter already in
the set. Read this as an argument for richer age moments, not for dropping `sigma_1_1`.

**3. `mu_1` is the better-conditioned tenth parameter, not the worse one.** Adding `mu_1`
gives cond 198.4 / σ_min 0.082 against `sigma_4_1`'s 228.9 / 0.060 — the opposite of what the
0.991 cosine alone would suggest, because that cosine is with `sigma_4_0` specifically while
conditioning is a property of the whole column set. Neither is adopted; both now have
evidence attached.

**4. Estimate correlations are far worse than the column cosines.** From the sandwich:
`phi_3`/`R_0` **−0.998**, `R_0`/`sigma_4_0` **+0.987**, `R_0`/`sigma_1_0` **+0.986**.
Valuation against technology is the binding problem, it is worse than any pairwise Jacobian
cosine shows, and it is invisible without the covariance.

**5. Out of scope, found in passing: the by-age target generator crashed here, and my
first diagnosis of why was wrong.** `tools/make_smm_targets.py` was writing the targets and
*then* dying in `write_by_age()`, which is a bad failure mode regardless of cause; the guard
that makes that stage skip loudly instead stays. But I attributed the crash to a missing
column and concluded the committed `Input/smm_moments_by_age*.csv` were **stale**, and that
conclusion was wrong — corrected in `fe78940`. Regenerating them from the right extract
reproduces the committed files byte for byte. **They are current.**

Two things do remain, and they are not the same claim:

- **The `.dta` extracts differ between machines.** On `haflinger`,
  `Input/SMM_Moments_ByAge.dta` (md5 `d8688f1c…`, 18 rows × 133 columns) carries the full
  `mu_/sd_/md_/wmu_` set for consumption, time, investment and achievement and **no asset
  column at all**, and `SMM_Moments_ByAge_Cohort.dta` is absent — only its derived CSV is
  present. So the crash here was real and reproducible; this server simply has an older
  extract than the one the correction was checked against.
- **The `.dta` files are still untracked.** `fe78940` removed `Input/*.dta` from
  `.gitignore`, which is the right root-cause fix, but did not `git add` the files — so a
  fresh clone still gets no by-age inputs and the stated goal is not yet achieved. **They
  should be added from the machine with the newer extract, not from this one**, or the
  repository would capture the version without the asset columns.

Before that happens, note the warning `fe78940` itself records: this repository is public,
`SMM_Moments_Micro.dta` is individual-level PSID/CDS microdata, and git history is
permanent. That is a data-agreement question, not a git one.

---

## What is still open

**Tier 3 — untouched, by instruction.** State-level parallelism (8/10) and alternative local
methods / NLopt version (7/10). Unchanged from the historical assessment below.

**Everything that needs a fitted baseline.** These are not deferrals of work but of *inputs*:

| | |
|---|---|
| The identification audit, re-centred | `jacobian.jl --at <fitted estimates.toml>` |
| Standard errors that mean something | `standard_errors.jl --at <fitted estimates.toml>`; the J-test only becomes meaningful there |
| The full 90-curve sensitivity exercise | the command above; ~40 full re-estimations |
| Comparable final results | old and new vectors re-evaluated under the corrected definitions — the machinery exists (`--at` everywhere, `Q_search`/`Q_final` kept separate), the comparison needs two fitted points |

**Genuinely unresolved specification questions**, none of which code can settle:

- **The asset grid range.** 7 of 2,000 households leave the top of the grid and the maximum
  is 2.6× the ceiling, with the worst of it in the handoff column that becomes the child's
  initial assets. The diagnostic is now correct and per-period; whether to widen `a_max`,
  re-place the nodes within the 30-node cap, or accept the tail is a modelling decision.
  Small tail mass is not proof of small policy error.
- **The tenth parameter.** Keep nine as the baseline. Both candidates now have measured
  evidence and neither is clearly admissible; equal moment and parameter counts remain no
  argument at all.
- **`par_time_tot` and the leisure budget.** Unchanged: the targets force model leisure ~26
  hrs/wk below what the same data measures, and `phi_2` absorbs it.
- **The parental-education correction itself.** Landed as the default, flagged here for
  Sahber. It is a correctness fix rather than a specification change, but it moves the
  baseline and results built on it should not circulate before that conversation.

---

## Corrections to historical claims below

- There are eight level moments and two log-HC moments. The old rounded HC values
  6.5492 − 6.0802 imply 0.4690 log points, approximately 59.8% in levels. Changing a
  denominator from 6.1 to 0.05 changes squared-objective weight by about 14,884 times,
  not 150. A Jacobian column's squared-sensitivity share is not automatically a
  parameter's estimation influence or proof of identification.
- Pairwise correlation and full column rank concern local responses under chosen
  scales; they do not prove global identification. The study profile is endogenous
  and its early/late means are targeted even when both slope parameters are calibrated.
- Coarse-grid speedups are measurements, not guarantees. Fine-grid refinement is
  implemented, but neither its numerical risk nor the remaining speedup opportunity
  has been shown to be zero. Restarts hitting budgets need scrutiny regardless of
  whether the current design has more moments than parameters.
- The historical shared-optimizer bug supplies a plausible crash mechanism, not a
  reproduced causal diagnosis of the old silent exit. Checkpoints protect against
  termination/pre-emption; a normal tmux client disconnect itself does not kill the job.
- The claim of an exact `(sigma_4_1, mu_1)` identification ridge was wrong and was corrected
  at `dca7980`. The near-collinear pair is `(sigma_4_0, mu_1)`, and even 0.991 is a warning
  about local separation, not observational equivalence.

---

# Historical review — before implementation and the current assessment

Triage of an external review of the SMM pipeline, TikTak, both solvers and the target
generator. Every claim was checked against the code, and the ones that matter were
checked numerically. Measurements: Julia 1.11, `simN = 2000`, incumbent calibration,
one shared child solution, `seed = 1234`.

**Headline: the review is accurate. Its ranking is not.** The item that most changes
the estimates it scores 8/10 and buries in a table; the cheapest large speedup it
scores 8/10 and understates by 25%. Nothing in it was fabricated.

---

## 1. The review's own measurements, reproduced

| | review | measured here |
|---|---:|---:|
| solve, grid 20 | 3.96 s | 3.04 s |
| solve, grid 30 | 7.16 s | 7.27 s |
| allocations, grid 20 | 1.70 GB | 1.58 GB |
| allocations, grid 30 | 3.81 GB | 3.54 GB |
| SMM objective, grid 20 | 2.5273 | **2.5273** |
| SMM objective, grid 30 | 2.5485 | **2.5485** |
| optimizations per evaluation, grid 30 | 153,000 | **153,000** |

Objectives agree to four decimals and the optimization count is exact, so the review
ran the real pipeline. Timings differ by machine noise in one direction only (see §4 on
the grid-20 number, which is the one place the noise changed a conclusion).

---

## 2. Verdicts

### Confirmed, and they change the estimates

**W1. Log-scaled HC moments are effectively weightless.** *(review: 8/10, part of
"weighting and identification are stale" — this is the most consequential finding in the
report.)*

The objective is `((m - m̂)/s)²` with `s = max(|m̂|, 0.05)`. Nine moments are levels;
the two HC moments are logs of W-scores, so `m̂ ≈ 6.1`. Measured at the incumbent:

```
HC early: model 6.5492  data 6.0802  ->  log gap 0.4712
  = a +60.2% error in the LEVEL of human capital
  scored by the objective as a +7.7% miss
```

A 60% error in the level of skill costs the same `Q` as a 7.7% error in consumption.
From the residual Jacobian (central differences, scaled to a full-box parameter move):

```
HC moments' share of R_0's total identifying leverage:  13.9%
```

`R_0` is in the estimated set *specifically* to fix the HC level, and 86% of what moves
it is its side effects on time and investment. Rescaling the HC residual to Δlog:

| | now | units-free |
|---|---:|---:|
| condition number | 162 | **49** |
| smallest singular value | 0.080 | **0.271** (3.4× better) |
| `R_0` leverage from HC moments | 13.9% | **86.1%** |

The weakest direction is `lambda_2` against `sigma_1_0 + sigma_4_0 + sigma_2_0` — the
valuation-vs-technology collinearity the estimation memo says the HC moments resolve.
**They currently don't**, because the log scaling crushed them. The system is full rank,
so it is identified; it is badly conditioned exactly where the memo claims it isn't.
Note the scaling is also arbitrary: index HC to 1 instead of W-scores and `log HC ≈ 0`,
the 0.05 floor binds, and these moments get ~150× *more* weight.

**W2. Age weights and coverage differ between data and model.** *(review: 9/10 —
correct.)* The generator pools observations, so ages with more observations weigh more;
the simulation weighs every age equally. Verified against the microdata:

| moment | pooled (current target) | equal-age | diff |
|---|---:|---:|---:|
| parental time, early | 0.4672 | 0.4544 | −2.7% |
| parental time, late | 0.3232 | 0.3333 | +3.1% |
| investment, early | 0.3532 | 0.3429 | −2.9% |
| investment, late | **0.4414** | **0.3911** | **−11.4%** |

Separately, coverage: the model averages `log(sim_hc)` over ages **1–9**, while `x_gach`
has 0 observations at age 1 and 1 at age 2, so the data target is effectively **3–9**.
Measured cost:

```
model ages 1-9 = 6.5492   ages 3-9 = 6.6588   (+0.110 log points)
= 23% of the entire HC gap the estimation is trying to close
```

The review's catch of the single stray age-2 observation is correct.

**W3. σ₄ is not flat, and two documents say it is.** *(review: 7/10 — under-scored.)*
`sigma_4_1 = 0.02` is inherited from `PARENT_DEFAULTS`; `sigma_4_1` is not in
`SMM_PARAMS`, so `unpack` never overrides it. Measured: `sigma_4_vector` runs
0.01133 → 0.01412 over ages 6–17, a **+24.6% rise** — the review's figure exactly.
`moments.jl` says σ₄ is "held flat in t". **μ₁ is not estimated either**, so the age
profile of study time is currently an assumption on both sides. This has been corrected
in `ESTIMATION_MEMO.md`; the code comment still carries it.

### Confirmed, robustness rather than estimates

**R1. TikTak's local searches share one optimizer object.** *(10/10 — confirmed.)*
Inside `run_one`, `opt = Opt(...)` looks local, but `tiktak` assigns `opt` again for the
polish. Julia's scoping rule makes the inner assignment refer to the *enclosing* local,
which is then boxed. Verified in lowered code: `Core.Box()` is present for exactly that
scoping shape. Under concurrency, restarts overwrite an optimizer another restart is
configuring.

**R2. `parallel = false` does not disable local-stage threading.** *(9/10 — confirmed
empirically.)* `batch` defaults off `Threads.nthreads()`, and the local stage branches on
`nb > 1` without consulting `parallel`. `run_smm.jl` never passes `batch`, and
`Nstar = 100` gives `floor(√100) = 10`. Reproduced:

```
julia -t 4 :  parallel=false, batch defaulted to 4
  objective evaluated on thread ids: [1, 2, 3, 4]
```

R1 and R2 compose into one bug: multiple threads driving a shared NLopt `Opt`. That is a
sufficient explanation for the silent exit-0 crash already recorded in the repo.

**R3. Invalid entries vanish from the objective.** *(8/10 — confirmed.)* Injecting a NaN
into `sim_c` still returns a finite `mean_c_p` (3.703901). `nanmean` filters; nothing
counts what it filtered.

**R4. Parental education enters the college comparison inconsistently.** *(7/10 —
confirmed, and worse than described.)* The stored transfer value is
`coef*V_child + mu*V_parent` with `coef = (1−mu) + mu*omega = 0.650`, but
`pared_value_offset` is added unscaled at both choice sites:

```
offset(bc=1)      = +0.0268   applied as-is
coef * offset     = +0.0174   what the weighting implies
=> the ParEd shift is 54% larger than consistency allows
```

And `terminal_value_surface` omits the offset entirely, so **the parent solves its
terminal problem as if BothCollege = 0** while the simulation applies the shift. Policy
and outcome disagree. This is a specification question, not a numerical fix.

**R5. Solver status is accepted loosely; failures are caught broadly; no checkpoints.**
*(8/10 ×2 — accurate as descriptions.)* Parent convergence is 100% at the incumbent, so
these are contingency risks over a search that visits bad regions, not active bugs. The
checkpoint gap is the one that bites on a server.

**R6. `Q_final` is the coarse-grid objective.** *(7/10 — confirmed.)* `say_report` uses
`G_FULL_`; the saved `result.f` comes from the search grid. Printed fit and stored `Q`
are from different grids.

**R7. Legacy entry points are broken.** *(7/10 — confirmed.)* `code/smm.jl:343` passes
`phi_2_0`, `phi_3_0`, `lambda_2_0`; zero matches remain in `parent_family.jl`. The
notebook has **14** such call sites.

### Confirmed but over-scored

**D1. Simulated states outside the grid.** *(8/10 — real, but benign.)* Measured
0.100% of asset cells above `a_max = 100`, max 254.4. **All of it is at t = 1** — it is
the initial wealth draw, not a transition failure, and flat extrapolation of a policy at
the top of the grid is the intended behaviour there. One clamp fixes it; it is not
evidence the solver is wandering off its domain.

**D2. `par_time_tot` measures presence, not exclusive time.** *(9/10 interpretation
concern.)* Correct, and already documented at length in `moments.jl` and
`make_smm_targets.py` as a deliberate instruction from 2026-08-28. Not a new finding.

---

## 3. Speed proposals

| review | verdict |
|---|---|
| Precompute expected continuation (9/10) | **Sound.** Hermite interpolation is linear in (values, slopes) at fixed nodes, so averaging both with the transition probabilities and evaluating once is exact. The review's warning not to re-derive PCHIP slopes from averaged values is correct and easy to get wrong. |
| Combined value + gradient (9/10) | **Sound.** `PchipContinuation` does four `_herm` calls for the value, and `Interpolations.gradient` repeats all four, discarding what it just computed. Called `Np = 5` times per objective evaluation. |
| Reduce allocations (8/10) | **Sound.** `grad[:] = -grad[:]` at the objective wrapper confirmed. 3.54 GB per solve justifies profiling. |
| Coarse then fine (8/10) | **Correct but understated — see §4.** |
| Least-squares local method (8/10) | Reasonable, unproven here. Its caveat about derivative stability under CRN is the right one. |
| Budgets and tolerances (7/10) | **Partly stale — see §4.** |
| State-level parallelism (9/10) | **Disagree on priority — see §4.** |
| Newer NLopt (7/10) | Not mid-thesis. Changing a pinned solver dependency invalidates every result produced against it. |

---

## 4. Where the review is wrong

**The coarse-grid speedup is 2.4×, not 1.8×.** Measured 3.04 s vs 7.27 s; `run_smm.jl`'s
own recorded benchmark gives 4.78 s vs 11.62 s, also 2.4×. The review's 1.8× comes from a
grid-20 timing (3.96 s) that is high against both independent measurements. This matters:
**the run is ~99% sequential local stage, and `--grid 20` is a flag, not a refactor.** The
machinery to search coarse and report at the full grid already exists. This is the single
largest speedup available and it carries no implementation risk.

**State-level parallelism is wrong for a thesis.** It is a large refactor, it collides
with the NLopt hazard, and shipping the continuation to workers for 17 periods × 5 shocks
per evaluation will eat much of the gain. The review's own architecture note concedes the
communication problem.

**The tolerance rationale has already expired.** `local_ftol_abs = 1e-10` was added
because a *just-identified* SMM drives `Q → 0`, where `ftol_rel` can never fire. The
estimator is now over-identified with `Q ≈ 2.53`, so `ftol_rel = 1e-3` fires normally.
The 165-evaluations-per-restart budget was measured under the old design and is probably
now an **overestimate** — watch the first few restarts before trusting the runtime
projection.

**The NLopt thread-safety pushback is fair and self-defeating.** Independent `Opt`
objects are safe upstream; the repo's blanket claim is too strong. But the `opt` here is
boxed and shared, which explains the crash — so the conclusion (use processes) is right
until R1 is fixed, even though the stated reason is wrong.

---

## 5. What to do, in order

### Tier 0 — before launching. These change the estimates.

| | action | effort |
|---|---|---|
| 0.1 | Rescale HC residuals units-free (Δlog, not Δlog/6.1) | ~5 lines, `moments.jl` |
| 0.2 | Align age coverage (`1:9` → `3:9` for HC) and use equal-age weights in the generator | ~20 lines, two files |
| 0.3 | Decide σ₄₁ / μ₁ — estimate one, fix the other, and say which | config + memo |
| 0.4 | Run the search at `--grid 20` | a flag |

### Tier 1 — same sitting. Insurance; no effect on the estimates.

| | action | effort |
|---|---|---|
| 1.1 | Give each local search its own `Opt`; make `batch` respect `parallel` | ~10 lines, `tiktak.jl` |
| 1.2 | Checkpoint the incumbent best after every restart | ~10 lines, `run_smm.jl` |
| 1.3 | Count and report filtered/out-of-grid cells instead of dropping them silently | ~10 lines, `moments.jl` |

### Tier 2 — after the run, before anything circulates.

2.1 Report and store `Q` at the full grid · 2.2 Clamp the initial wealth draw to `a_max` ·
2.3 Resolve the ParEd inconsistency (R4) · 2.4 Delete `code/smm.jl`; fix the 14 notebook
call sites.

### Tier 3 — only with slack.

3.1 Combined value+gradient interpolation · 3.2 Expected-continuation precompute
(measure both before committing) · 3.3 Weighting matrix and standard errors — a thesis
section, not a bug fix.

### Do not

State-level parallelism inside backward induction · an NLopt version bump mid-thesis ·
reviving the legacy estimator.

### Advisor-gated

Per `CLAUDE.md`, specification changes go to Sahber before results built on them
circulate. **0.1, 0.2, 0.3 and 2.3 all qualify** — they change what is estimated and how.
They belong in the same message as the open μ₁ question, not fixed quietly. Everything
else on this list is a numerical fix and does not need to wait.

---

## 6. Server notes

**The threading bug is a server bug in practice.** If `JULIA_NUM_THREADS` is set in the
server environment — common on cluster images — the local stage silently threads across a
shared NLopt object, which is the exact silent exit-0 crash already in the notes. Until
1.1 lands, launch with an explicit thread count of one:

    cd code/smm && nohup julia +1.11 -t 1 --project=../.. run_smm.jl --grid 20 > /dev/null 2>&1 &

**More cores buy almost nothing.** `WORKER_BUDGET = 20` accelerates only the Sobol stage,
which is roughly 10 minutes of a ~20-hour run. The local stage is sequential by
construction. Do not request a larger allocation expecting it to help.

**Checkpointing matters more here than locally** — tmux disconnects, wall-clock limits
and pre-emption all cost the entire run at present, because parameters are written only
after the final report.

---

# Addendum — what was implemented, and what was deliberately left

Applied on branch `fix/estimation-consistency`.

## Implemented

| Area | Change | File |
|---|---|---|
| TikTak | Local search moved to a free function `_local_search` with its own `Opt`; polish binding renamed `polish_opt`. `Core.Box` is gone from the lowered code. | `code/src/tiktak.jl` |
| TikTak | `batch` now defaults to 1 unless `parallel = true`; the threaded branch checks `parallel`; `batch > 1` with `parallel = false` is refused with an error rather than silently threading. | `code/src/tiktak.jl` |
| TikTak | Return codes kept in the trace; exceptions counted in `n_exception` and warned at the moment they happen. The seed evaluation `f(x0)` is now guarded too — it was not, so one throw killed a whole run. | `code/src/tiktak.jl` |
| TikTak | `on_local` also receives the incumbent minimiser, so callers can checkpoint. | `code/src/tiktak.jl` |
| Moments | `moment_scale`: log moments are scaled by 1, level moments by their target. Removes the 6.1× shrinkage of the HC residuals. | `code/smm/moments.jl` |
| Moments | `SMM_AGE_HC_LO = 3`; the model's early-HC group is ages 3–9, matching the data. | `code/smm/moments.jl` |
| Moments | Non-finite and non-positive cells counted (`n_nonfinite`) and the evaluation refused rather than silently averaged over survivors. | `code/smm/moments.jl` |
| Moments | Grid-coverage diagnostics reported, split by t = 1 vs transitions. Header and σ₄ comments corrected. | `code/smm/moments.jl` |
| Targets | Equal weight per child age on the data side, matching the simulation. `mean_pooled` still written for audit. HC group starts at age 3. | `tools/make_smm_targets.py` |
| Model | `HC0_MEAN_LOG/SD` re-evaluated at **child age 1**, the child age of column 1 (5.9529 / 0.0667, was the age-0 intercept 5.9290 / 0.0698). | `code/src/parent_family.jl` |
| Run | Atomic per-restart `checkpoint.toml`. | `code/smm/run_smm.jl` |
| Run | Full-grid refinement: BOBYQA polish on the full-grid objective from the coarse winner. `Q_final` and `Q_search` stored separately, on the grids they were computed at. | `code/smm/run_smm.jl` |
| Run | `--quick` now bounds the local and polish evaluation caps (60 / 120 against 2000 / 4000), and both are settable with `--local-evals` / `--polish-evals`. The flag previously cut the *number* of restarts but not their length, so a "2 minute smoke test" could sit in one restart for a quarter of an hour. | `code/smm/run_smm.jl` |
| Moments | The per-startup `@warn` that the design is no longer just-identified is gone — over-identification is now the documented intent. Under-identification (fewer moments than parameters) is an error instead. | `code/smm/moments.jl` |
| Legacy | `code/smm.jl` carries a header saying it is superseded and non-functional, with the three specific reasons. Not deleted — retiring it is a call to make deliberately. | `code/smm.jl` |

## Deliberately not implemented

**Parental education in the college comparison** (`child_lifecycle.jl`). Confirmed real:
`pared_value_offset` is added unscaled to a value in which child utility carries
`coef = (1−mu) + mu*omega = 0.650`, so the shift enters 54% larger than the weighting
implies; and `terminal_value_surface` omits it entirely, so the parent solves as though
`BothCollege = 0` while the simulation applies it. **Fixing it requires deciding which of
the two is correct, which is a specification choice, and specification changes go through
the advisor** (`CLAUDE.md`). Making the terminal continuation depend on parental type also
adds a state dimension to the parent's terminal problem. Left for that decision.

**Asset grid range.** Not clamped, on the review's own reasoning — `sim_a_init` is
`LogNormal(0.2962, 1.4018)` and clamping its upper tail would distort the initial wealth
distribution to flatter a grid. Measured: 0.100% of cells above `a_max = 100`, max 254,
**all of it at t = 1**. The real choice is between widening `a_max` — which under the ≤ 30
node cap coarsens the region where the mass actually sits — and accepting flat
extrapolation for a 0.1% tail. Instrumented and reported on every run; the grid itself is
unchanged pending a deliberate decision.

**σ₄₁ as a tenth parameter.** The recommendation to estimate `sigma_4_1` while holding
`mu_1 = −0.04` is reasonable and its argument is sound — study time is already targeted
early and late, σ₄₀ is already estimated, and it stays parent-block so the cached child
solve remains valid. **But it is a specification change and it is the exact question
already open with the advisor.** The code comment that falsely claimed σ₄ was flat has
been corrected to state what the code does; the choice itself is not resolved here.

**Weighting matrix and standard errors.** Equal weights are now documented as a real
assumption rather than a harmless one. A covariance-based weighting matrix, the
target-perturbation response curves, and sampling uncertainty are analysis, not
maintenance, and belong in the thesis rather than in a pre-run fix list.

**Performance work** — expected-continuation precompute, combined value+gradient
interpolation, allocation reduction, state-level parallelism, an NLopt bump. All deferred.
The `--grid 20` search plus full-grid refinement now delivers most of the available
speedup with no numerical risk, and the correctness fixes above should establish the
baseline that any optimisation has to preserve.

## One bug introduced and caught

The first version of the refinement stage assigned `Z_FINAL` / `Q_FINAL` inside a
top-level `try` block. `try` is a soft scope, so those assignments bound locals and left
the globals at the search-grid values — `estimates.toml` reported the UNREFINED point
while the log printed the refined one. This is the same rule that bites top-level `for`
loops, which `CLAUDE.md` already warns about.

Caught by the smoke test, which showed `Q_final == Q_search` to every digit while the log
above it reported a different full-grid number. The refinement now lives in a function
that returns the pair, so the trap is unreachable rather than merely avoided. Worth
recording because the failure was silent and produced a plausible-looking file.

## Bearing on the estimates

Five of the changes move the numbers: the HC residual scaling, the HC age floor, the
equal-age weighting, the HC₀ re-indexing, and the refusal of invalid simulations. **All
previous estimates were produced under the old definitions and are not comparable.** The
first three are advisor-gated under `CLAUDE.md` and should reach Sahber with the open
μ₁/σ₄₁ question rather than arriving inside a set of results.


---

# Second round — the four follow-up findings

All four were verified and all four were valid. Three were errors in the first round's
work; one was an overstatement in its commit message.

## 1. Feasibility was checked only for human capital

The first round counted non-finite cells and called it a validity check. Measured by
injecting each pathology into a solved baseline:

```
sim_c  = NaN                          caught
sim_c  = -5.0  negative consumption   NOT caught
sim_h  = -0.3  negative hours         NOT caught
sim_h  =  1.8  hours > time budget    NOT caught
sim_a  = -50   below a_min = 0        NOT caught
sim_hc = -1.0  negative skill         caught
```

**Fixed.** `simulation_violations(p)` now checks every simulated series against the
model's own domain — consumption and skill strictly positive, investment non-negative,
each time share in the unit interval, both leisure residuals non-negative, assets at or
above `a_min` — and counts violations **by kind**, so the penalty log records which
economic law broke rather than that something did. Assets are checked over all `T+1`
columns. `smm_objective` refuses such a draw; `report_fit` prints the table. Zero
violations at the incumbent, so the check does not produce false positives.

## 2. Checkpoint metadata, and no resume

Two problems. The post-refinement call passed a **full-grid** objective while
`checkpoint!` unconditionally wrote `grid_search`, labelling a grid-30 number as grid-20.
And only saving was implemented, not continuation.

**Fixed.** `checkpoint!` takes `stage` and `grid`, and writes `objective_grid` alongside
`grid_search`/`grid_report`. Resume is now real and **exact**, not a warm start: the
pre-testing survivors are persisted once to `seeds.toml`, and `--resume <dir>` re-enters
the local stage at the next restart with the saved incumbent and those seeds, so restart
*j* sees precisely the mixture it would have seen. Verified on Rastrigin: a run resumed
at restart 11 reached the identical objective (0.9949590571) in 1096 evaluations against
the full run's 2447. `load_resume` refuses rather than guesses when the parameter count,
restart budget or search grid differ from the saved run. The run directory is reused and
its log appended, so a resume does not truncate a two-day transcript.

## 3. "All at t=1" was wrong

Measured at the incumbent, full grid, `simN = 2000`:

```
t=1..17   2 households above the ceiling, max 214.43 -> 254.41
t=18      7 households above,             max 259.04     <- the handoff
households ever above: 7      above at t=1: 2
```

The first round compared the fraction of **households** above at t=1 (0.1%) with the
fraction of **cells** above over t=1..17 (0.1%), read the equality as "all of it is the
initial draw", and reported that. They match only because 2 households × 17 periods /
34,000 cells equals 2 / 2,000 — different denominators, same digits. **Five of the seven
cross during the family stage**, and the old diagnostic excluded column `T+1` — the one
that becomes the child's initial assets — from both its share and its maximum.

**Fixed.** Coverage is reported as households (ever above, at t=1, at the handoff) over
all `T+1` columns, with the maximum taken over all of them, and the crossing count called
out explicitly. The grid itself is still unchanged: clamping would distort the initial
wealth draw, and widening `a_max` under the ≤ 30 node cap coarsens the region where the
mass sits. That trade-off is a decision, not a bug fix.

## 4. The identification ridge named the wrong pair

The first round claimed σ₄₁ and μ₁ form a ridge. Taking logs of the child's study FOC,
with μ₀ = 1 and λ₁ = 1 so that $(1-\tilde\mu_t) = -\mu_1(t-5)$:

$$\log\frac{\sigma_{4,t}}{1-\tilde\mu_t} = \sigma_{40} + \sigma_{41}(t-5) - \log(-\mu_1) - \log(t-5)$$

μ₁ enters **only the intercept**, which makes it collinear with σ₄₀, not with σ₄₁.
Measured |cos| between scaled residual-Jacobian columns:

| pair | \|cos\| |
|---|---:|
| **σ₄₀ vs μ₁** | **0.991** — worst pair of all eleven |
| σ₄₀ vs σ₄₁ | 0.813 |
| σ₄₁ vs μ₁ | 0.805 — correlated, not a ridge |

**Fixed** in the comment, with the measurements.

**A caveat on the recommendation to add σ₄₁.** It is well founded on rank, but it costs
conditioning:

```
 9 parameters (current)     condition number   49.2,  smallest sv 0.278
10 with sigma_4_1 added     condition number 1067.1,  smallest sv 0.052
```

because σ₄₀ and σ₄₁ are themselves 0.813 collinear — study time is targeted only at ages
6–9 and 10–17, too close together to separate the level of the elasticity from its slope.
**A study-time moment further apart in age would fix that; adding the parameter alone
would not.** This belongs with the specification question rather than being settled here.

## Still deferred

Unchanged from the first round, and for the same reasons: the parental-education
treatment (a specification choice that would add parental type as a state to the parent's
terminal problem), σ₄₁ itself, the asset-grid range, standard errors, the
target-moment sensitivity exercise, and all of the interpolation and parallelism
performance work.
