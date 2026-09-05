# Code review, verified — and what to do before the next estimation run

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
