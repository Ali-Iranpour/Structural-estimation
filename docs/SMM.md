# Estimation: SMM by TikTak

How the parent block is estimated: what TikTak does, what we set, which moments are
matched, and — the part that matters for the paper — **every place we depart from the
published algorithm, and what the fit does not deliver**.

```bash
export PATH="$HOME/.juliaup/bin:$PATH"          # Manifest.toml is resolved against 1.11
cd code/smm
julia +1.11 --project=../.. run_smm.jl --report-only    # ~1 min, no search
julia +1.11 --project=../.. run_smm.jl --quick          # smoke test
julia +1.11 --project=../.. run_smm.jl                  # the real run
```

Flags, runtimes and the parallelism story are in
[`code/smm/README.md`](../code/smm/README.md). This file is the method.

> **Status, 7 September 2026.** The completed nine-parameter run
> `2026-09-06_183119` is now the baseline in `PARENT_DEFAULTS`, with
> Q = 0.2500261422642604. Its outputs and original bounds are preserved; three
> limits were expanded for future searches. See [BASELINE_9PARAM.md](BASELINE_9PARAM.md)
> and the [run inspection](../output/smm_diagnostics/2026-09-06_183119/inspection_notes.md).
> Older calibration measurements below are explicitly historical. Remaining work is
> tracked in [REVIEW_TRIAGE.md](REVIEW_TRIAGE.md).

---

## Part 1 — What TikTak is

**TikTak**, from Arnoud, Guvenen & Kleineberg (2022), *Benchmarking Global Optimizers*,
which finds it the strongest performer on both test functions and their economic
application. Reference implementation: <https://github.com/serdarozkan/TikTak>. Ours is
[`code/src/tiktak.jl`](../code/src/tiktak.jl), standalone and reusable.

It is a **structured multistart** method: explore broadly, screen down to the promising
points, then progressively concentrate local searches around the best solution found so
far. Ordinary multistart launches local optimizers from arbitrary points; TikTak makes the
starting points systematic and lets the history of earlier searches steer later ones.

### Stage 0 — global pre-testing

Generate `N` **Sobol'** points over the parameter box and evaluate `Q` at all of them. A
Sobol' sequence is deterministic and low-discrepancy: it fills a multidimensional box more
evenly than independent uniform draws, leaving fewer large uncovered regions.

Sort ascending and keep the best `N*` as seeds `s₁ … s_{N*}`
([`tiktak.jl:323`](../code/src/tiktak.jl#L323)). This is cheap reconnaissance — the
rejected points are not proved useless, they are dropped because their neighbourhoods look
less promising under the available budget.

### Stage 1 — the first local search

Run the local optimizer from the best seed `s₁`. The result is the **incumbent**.

### Stage 2 — mixing, which is the whole idea

For restart `j > 1`, take the next unused seed `s_j` and the best local minimum so far,
`p*`, and start from the convex combination

```
S_j = (1 − θ_j)·s_j + θ_j·p*,        θ_j = clamp( (j/N*)^0.5, 0.1, 0.995 )
```

Small `θ` keeps the start near an independent global seed; large `θ` pulls it onto the
incumbent. So the algorithm shifts from exploration to exploitation on its own. At the
default `N* = 100`: `θ ≈ 0.1` early, `0.5` at `j = 25`, `0.9` at `j = 81`, capped at
`0.995` so the seed never stops contributing
([`tiktak.jl:352`](../code/src/tiktak.jl#L352)).

Why mix at all: if an early search finds a good basin, pure multistart keeps launching from
unrelated points and ignores the discovery. TikTak uses it immediately — but not
exclusively, so the incumbent can still be challenged. The seed also supplies **direction**:
late in the run, different seeds approach the incumbent from different sides, giving the
local optimizer repeated chances to fall into a neighbouring basin or enter a narrow valley
from a new angle.

### Stages 3–4 — local search, then polish

Run a local optimizer from `S_j` to convergence; if it improves on the incumbent it replaces
it. After all `N*` restarts, apply one final **polishing** search with a stringent tolerance
and return the best point.

The suffixes in `TikTak-nm3` / `TikTak-d8` are the local tolerance (`1e-3`, `1e-8`), not the
problem dimension.

### Why it suits SMM

`Q` need not be differentiable — ours contains discrete college decisions, simulation
counts and numerical solution artifacts. And it parallelises where it matters: the Sobol
stage is embarrassingly parallel, and expensive structural solves dominate the cost.

---

## Part 2 — What we actually did

### Four departures from the paper

State these as departures. The write-up above is the paper; this is us.

| | Paper | Ours | Why |
|---|---|---|---|
| **Local optimizer** | DFNLS preferred for SMM (exploits the least-squares structure) | Nelder-Mead, `ftol_rel = 1e-3` → **TikTak-nm3** | no maintained Julia DFNLS binding; `nm` is the faithful variant available |
| **Seeding** | Sobol' points only | Sobol' **+ the incumbent calibration** forced into the pool (`extra_seeds`) | it competes on function value like any Sobol point, and guarantees the estimate is weakly better than the calibration we started from. Pass `extra_seeds = []` for the published algorithm exactly |
| **Stopping rule** | `ftol_rel` | adds `local_ftol_abs = 1e-10`, `local_xtol_rel = 1e-8` | see below |
| **Post-search refinement** | not in the paper | when `--grid` differs from the report grid, a short BOBYQA polish **on the full-grid objective**, started from the coarse winner | the coarse and fine objectives have different minimisers, so the coarse argmin is a starting point, not an answer |

`N*/N` is **not** a departure any more: the defaults are `N = 1000`, `N* = 100`
([`run_smm.jl:70-71`](../code/smm/run_smm.jl#L70-L71)), i.e. the paper's own `N* = 0.1N`.
Restarts are sequential and cost ~15 min each, so cutting `--restarts` is the first thing
to give up on a budget — and `--sobol` is the thing to raise, since it divides by the
worker count.

**The stopping rule is not a detail.** `ftol_rel` tests `|Δf| ≤ ftol_rel·|f|`, so **as
`f → 0` the threshold collapses with it and the test can never be satisfied**. The earlier
just-identified design drove `Q` to ~0 by construction, so every restart ran to
`maxeval = 2000` regardless of having converged — measured, restart 1 reached `Q ≈ 0` at
evaluation 61 and was still running at 290. At ~15 s an evaluation that is 8 hours per
restart instead of 15 minutes. Absolute criteria work at any scale, and a collapsed simplex
means converged whatever `f` is worth there. The self-test still reaches `7.7e-45` on the
sphere, so accuracy is unaffected. The current design is over-identified and `Q` no longer
approaches zero, but the guard stays: it is scale-free and costs nothing.

Everything else matches the paper: Sobol' pre-testing, ascending sort, keep best `N*`, and
the benchmark schedule `clamp((j/N*)^0.5, 0.1, 0.995)`, with restart 1 pinned at `θ = 0` so
it starts purely from `s₁`. Polish is BOBYQA at `1e-10`
([`tiktak.jl:228`](../code/src/tiktak.jl#L228)).

### The nine parameters

**Over-identified by one**: ten moments against nine parameters. That is deliberate, and it
changes two things from the earlier just-identified design — `Q` **cannot** reach zero, and
the weighting matrix is **not** irrelevant at the optimum. A residual gap is therefore not
by itself evidence of a bug. Counting nine against ten also establishes nothing about
identification; what does is the residual Jacobian (below).

`phi_1` and `lambda_1` are **normalised to 1.0** and not estimated — utility is defined only
up to relative weights, so two of the five must be pinned (instruction 2026-08-30).
Preference weights are **time-invariant**: the `_1` slopes and per-period vectors on
`phi`/`lambda` were removed in `c27a049`.

| parameter | what it moves | bounds | link |
|---|---|---|---|
| `phi_2` | weight on parental leisure → `h_p` (work; `l = 1 − h − t`) | `[0.01, 20.0]` | log |
| `phi_3` | parents' weight on child skill → `t_p` and `e_p` | `[0.05, 20.0]` | log |
| `lambda_2` | the child's own weight on skill → `i_c` (study time) | `[0.05, 20.0]` | log |
| `R_0` | HC technology TFP → the **level** of `log HC` | `[0.5, 100.0]` | log |
| `sigma_1_0` | elasticity of HC to parental **time** → `t_p` level | `[−4.0, −0.1]` | level |
| `sigma_1_1` | its age slope → `t_p` early vs late | `[−0.20, 0.05]` | level |
| `sigma_2_0` | elasticity of HC to **money** → `e_p` level | `[−5.0, −0.5]` | level |
| `sigma_2_1` | its age slope → `e_p` early vs late | `[−0.10, 0.05]` | level |
| `sigma_4_0` | elasticity of HC to the child's **own study** → `i_c` | `[−8.0, −1.0]` | level |

Source of truth: [`moments.jl:506`](../code/smm/moments.jl#L506). Strictly-positive weights
are searched **in logs**, so a step can never propose a negative weight — and the log link
is also what concentrates the search. A Sobol sequence uniform in `log θ` is not uniform in
`θ`: its density in levels falls like `1/θ`. Over `R_0`'s box that puts 25% of pre-testing
points below 1.9, half below 7.1 and only a quarter above 26.6.

`R_0` remains in [0.5,100] and is searched in logs; its fitted baseline is now
50.60319558. The three expanded elasticity bounds are listed in
[BASELINE_9PARAM.md](BASELINE_9PARAM.md). The original fit and historical identification
matrices retain their original bounds. Elasticity coefficients are searched in levels.

**`R_0` became estimable only once HC was put in the data's units.** Before the rescaling
(`c27a049`) there was no HC moment to identify it against. That rescaling is what separates
the **valuation** parameters (`phi_3`, `lambda_2`) from the **technology** parameters
(`R_0`, `sigma_1`, `sigma_2`, `sigma_4`): both raise investment, and only the resulting HC
level can tell them apart.

**What is deliberately NOT estimated, and why it is not settled.** `sigma_4_1 = 0.02` and
`mu_1 = −0.04` are held at their calibrated values, so the age profile of study time is at
present an assumption on **both** sides. The reason `sigma_4_1` is out is **conditioning,
not rank** — and that claim is now a saved artefact rather than a recollection. Run
[`code/smm/jacobian.jl`](../code/smm/jacobian.jl), which writes the matrix, the evaluation
point, the boxes, the links, the moment scales, the grids, the seed and the finite-difference
steps beside every number it reports.

Measured 2026-09-06 at the incumbent, grid 30, central differences at 0.5/1/2% of each box
width, columns scaled to a full-box move
(`output/identification/jac_9col`, `jac_10col_sigma41`, `jac_10col_mu1`, `jac_11col`):

| columns | condition number | smallest singular value | thin-SVD rank |
|---|---:|---:|---|
| 9 (current) | **51.2** | 0.266 | 9 of 9 |
| 10, adding `sigma_4_1` | **228.9** | 0.060 | 10 of 10 |
| 10, adding `mu_1` instead | **198.4** | 0.082 | 10 of 10 |
| 11, adding both | 183.4 | 0.089 | **10 of 11** — one direction unidentified by construction |

`σ_min` is stable across the three steps (spread 6–7% of its level), so it is resolved above
finite-difference noise.

The pairwise cosines reproduce as reported — `sigma_4_0`/`mu_1` **0.991**,
`sigma_4_0`/`sigma_4_1` **0.814**, `sigma_4_1`/`mu_1` **0.807** — which is expected, since
cosines are invariant to column rescaling. **The condition-number ratio does not.** The
previously circulated 49.2 → 1067.1 (21.7× worse) comes out as 51.2 → 228.9 (4.5× worse)
under a stated `sigma_4_1` box of `[−0.05, 0.05]`. Condition numbers are *not* scale
invariant, and the box that produced the original figure was never recorded. The direction of
the conclusion survives; the magnitude was never reproducible.

Two things that change the emphasis:

- **The worst-separated pair is one already in the estimated set.** `sigma_1_0` vs
  `sigma_1_1` is **0.908**, *higher* than the `sigma_4_0`/`sigma_4_1` 0.814 that is the
  stated reason for leaving `sigma_4_1` out. `t_p` and `i_c` are split at the same two age
  groups, so "ages 6–9 and 10–17 are too close to separate a level from a slope" applies with
  more force to a parameter that is already free. Read it as an argument for richer age
  moments, not for dropping `sigma_1_1`.
- **`mu_1` is the better-conditioned tenth parameter**, not the worse one, despite the 0.991
  cosine — because that cosine is with `sigma_4_0` specifically, while conditioning is a
  property of the whole column set.

Taking logs of the child's study FOC ratio, with `mu_0 = lambda_1 = 1` and `s = t − 5 > 0`,

```
log[ σ_4,t / (1 − μ_t) ] = σ_40 + σ_41·s − log(−μ_1) − log s
```

so `mu_1` enters that ratio only through the intercept, which is what makes it near-collinear
with `sigma_4_0`. **This is a warning, not a proof of an exact ridge**: `mu_1` also moves the
child's leisure weight and `α̃_2,t = φ_3 + μ_1·s·(φ_3 − λ_2)` in `util_total`, so 0.991 means
nearly parallel *local* moment responses, not observational equivalence. An earlier comment
in `moments.jl` asserted an exact `(sigma_4_1, mu_1)` ridge; that was wrong and is corrected
at [`moments.jl:530-558`](../code/smm/moments.jl#L530-L558). **Keep nine as the baseline** —
neither candidate is clearly admissible, and equal moment and parameter counts are no
argument at all.

At the incumbent the nine-parameter residual Jacobian has **full column rank**, condition
number 49 and smallest singular value 0.278. The weakest direction is `lambda_2` against
`sigma_1_0 + sigma_4_0 + sigma_2_0` — valuation against technology — and the second weakest
is `sigma_2_1`. Both are identified; neither is sharply identified.

### The ten moments

Targets are frozen in [`Input/smm_targets_baseline.toml`](../Input/smm_targets_baseline.toml),
generated by [`tools/make_smm_targets.py`](../tools/make_smm_targets.py) from the Stata
files. Julia never reads `.dta`, so a run is reproducible and a change of target shows up as
a diff.

| moment | data source | target | N |
|---|---|---|---|
| `mean_c_p` | `cons_exhous_real_w99` | 3.1155 | 6,742 |
| `mean_h_p` | `(wh_mom + wh_dad)/2 / 112` | 0.3073 | 15,665 |
| `mean_t_p_early` | `par_time_tot / 112`, ages 1–9 | 0.4544 | 475 |
| `mean_t_p_late` | `par_time_tot / 112`, ages 10–17 | 0.3333 | 590 |
| `mean_e_p_early` | `m_method2_final_w99`, ages 1–9 | 0.3429 | 8,178 |
| `mean_e_p_late` | `m_method2_final_w99`, ages 10–17 | 0.3911 | 7,182 |
| `mean_i_c_early` | `study_hrs / 112`, ages 6–9 | 0.0393 | 171 |
| `mean_i_c_late` | `study_hrs / 112`, ages 10–17 | 0.0496 | 584 |
| `mean_hc_early` | `x_gach` (log PCA composite), ages 3–9 | 6.0737 | 252 |
| `mean_hc_late` | `x_gach` (log PCA composite), ages 10–17 | 6.2508 | 549 |

**Ages are matched on both sides, and each side is weighted the same way.** The targets are
means **over child ages**, equally weighted, because the simulation is — an
observation-weighted pooled mean is a different number and is carried in the file as
`mean_pooled`, unused. Three age ranges are not `1..17`, and each is enforced in the model
too:

- `mean_i_c_*` starts at **child age 6**: the child is not a decision maker before
  `T_CHILD_VOICE = 6`, so `sim_i` is not a choice there.
- `mean_hc_*` starts at **child age 3**: the Woodcock-Johnson composite is not administered
  earlier (`x_gach` has 0 observations at age 1). Averaging the model over 1–9 against a
  data group that is really 3–9 was worth **0.110 log points** on its own, 23% of the whole
  HC gap ([`moments.jl:134`](../code/smm/moments.jl#L134)).
- the early/late split at age 9 lives in **two** files; `load_targets` refuses to run if
  `SMM_AGE_SPLIT` and the generator's `AGE_SPLIT` have drifted apart.

**Units.** One model unit = **\$10,000/year** (`ASSET_RESCALE = 10`; the model's mean
after-tax household income of 5.24 units = \$52,441, a plausible US figure). Time is a
**share of the 112-hour non-sleep week, per parent** — 112 = 168 less a 56-hour sleep
allowance. Per *parent*, not per household, because `wage_func` multiplies by 2: one modelled
adult stands for two earners sharing one time allocation. Human capital is in the **data's
units** (the log W-score composite), not model units — that is what makes it targetable.

**Why `e_p`, `t_p`, `i_c` and `hc` are split early/late.** A single mean cannot separate an
age slope from a level: many `(σ_j0, σ_j1)` pairs give the same overall average, and the
optimizer would slide along that flat direction and return whatever its seed was near. Two
group means pin both. `h_p` is *not* split — it is flat in child age (0.3062 early vs 0.3080
late), so one pooled mean is right, and it carries 15,665 observations against `t_p`'s 1,065
because work hours are measured for everyone while time diaries exist only for the CDS
subsample.

**Why `h_p` and `t_p` rather than `l_p`.** `l_p = 1 − h_p − t_p` identically, so targeting
leisure pins the *sum* and says nothing about the split — and the split is where the model
was wrong. The 2026-08-27 estimate matched leisure exactly while working 29.6 hrs/wk against
34.4 in data and doing 23.2 hrs of childcare against 18.2: two errors that cancel inside
`l_p` and are invisible to it. `l_p` is still printed, as the residual check that the time
budget closes.

### The objective

Weighted relative distance, `Q(θ) = Σⱼ wⱼ((mⱼ − m̂ⱼ)/sⱼ)²`, with **two scales**
([`moments.jl:182`](../code/smm/moments.jl#L182)):

```
s_j = max(|m̂_j|, 0.05)   for a LEVEL moment   — puts it on a proportional footing
s_j = 1                   for a LOG moment     — it is already a proportional error
```

The log exception is not cosmetic. `x_gach` is a log W-score, so the HC targets are ~6.1;
dividing their residual by the target shrank it **6.1×** before squaring. Measured at the
incumbent, the model's human capital was **+60% in levels** and the objective scored it as a
**7.7% miss**, while `R_0` — the parameter in the set specifically to fix the HC level — got
only 13.9% of its identifying leverage from the HC moments. On the units-free scale that
becomes 86.1%, the Jacobian's condition number falls 162 → 49, and its smallest singular
value is 3.4× stronger. The old scaling was also arbitrary in the literal sense: index HC to
1 instead of W-scores and `log HC ≈ 0`, the 0.05 floor binds, and the same two moments get
~150× *more* weight than they had. A moment's weight must not depend on the units its log
happens to be in. `report_fit` prints the HC gap in **levels** (`exp(Δlog) − 1`) for the
same reason.

Weights are otherwise **equal**, and that is now a real assumption rather than a harmless
one: the system is over-identified, so `Q` cannot reach zero and the weighting does change
the answer at the optimum. Equal weights are the choice that adds nothing unexamined, not a
choice that is free. **A covariance-based weighting matrix is the principled successor and
is not implemented** — see [`REVIEW_TRIAGE.md`](REVIEW_TRIAGE.md), Tier 2.

**Common random numbers** throughout: every model is built with the same `seed`, so draws
and shock paths are identical across evaluations. Without this the objective is a step
function of simulation noise and no derivative-free method converges — it would be chasing
the RNG, not the parameters.

### What the objective refuses to score

Three gates, in order, before a number is returned
([`moments.jl:649-731`](../code/smm/moments.jl#L649-L731)). Each returns the large **finite**
penalty `SMM_PENALTY = 1e6` — never `Inf` and never an exception, so a derivative-free local
search can still form a descent direction away from a bad region — and each increments a
per-worker reason counter that `run_smm.jl` gathers and prints at the end. A high penalty
rate means the **box** is wrong, not that the model is.

1. **Economically infeasible draws, before paying for a solve.** `smm_feasible` checks that
   the Cobb-Douglas money share `σ_2,t = exp(σ_20 + σ_21(t−1))` stays below 1 at both
   endpoints of `t = 1..17`. Above 1 the technology is explosive and SLSQP wanders to a NaN
   iterate rather than failing cleanly; a Sobol point landed in that corner and killed the
   2026-08-27 run at evaluation 376 of 401.
2. **Simulations that leave the model's own domain.** `simulation_violations`
   ([`moments.jl:381`](../code/smm/moments.jl#L381)) counts violations **by kind** —
   consumption and skill strictly positive, investment non-negative, each time share in the
   unit interval, both leisure residuals (`1−h−t` and `1−t−i`) non-negative, assets at or
   above `a_min` over **all T+1 columns**, and non-finite cells anywhere. This replaced a
   check that counted only non-finite entries, which is not the same thing: measured by
   injection, negative consumption, negative hours, hours above the time budget and assets
   below `a_min` were **all accepted** as long as they were finite. A partly-failed solve
   could compete on the strength of the cells that happened to survive. Zero violations at
   the incumbent, so it is not producing false positives.
3. **Exceptions, classified by root cause.** `ErrorException` (the solver's 95%-convergence
   throw), `DomainError`, `AssertionError` and `InexactError` are scored as penalties;
   anything else is **re-thrown**, because a `MethodError` is a coding error and must not be
   laundered into a converged run. The classification unwraps `CapturedException` first
   ([`_root_cause`](../code/smm/moments.jl#L219)) — NLopt wraps anything thrown inside a
   callback, so testing the type directly is always false, and that cost two runs.

### Running the search: budget, grids, and surviving a kill

| | default | flag |
|---|---|---|
| Sobol points | 1000 | `--sobol` |
| restarts | 100 | `--restarts` |
| evals per local search / polish | 2000 / 4000 | `--local-evals`, `--polish-evals` |
| parent grid, search | 30 | `--grid` |
| parent grid, report | 30 (fixed) | — |
| full-grid refinement evals | 200 | `--refine` |
| `simN` | 2000 | — |
| worker processes | 20 | `--procs` |

**Search cheap, quote exact.** 98% of an evaluation is `solve_model!` and its cost scales
with `Na × Nhc`. Dropping 30 → 20 makes an evaluation 2.5× cheaper and moves the targeted
moments by 0.01–0.2%, against the gaps the estimation exists to close. So `--grid` sets the
grid the **search** runs on, the fit is **always** re-solved and reported at 30, and the
coarse winner is then re-optimised at the full grid by a short BOBYQA polish
([`run_smm.jl:643`](../code/smm/run_smm.jl#L643)). `estimates.toml` records **both**
objectives under separate names, `Q_search` and `Q_final`, on the grids they were computed
at. Never quote a `Q` minimised at `Na = 20`. `simN` is *not* the place to economise: it is
2% of the cost, and cutting it to 500 moved `c_p` more than halving the grid did.

**Processes, never threads.** NLopt.jl is not thread-safe in this project — with
`parallel = true` and 8 threads the objective killed the process with exit 0 and no error.
Each worker *process* owns its NLopt state. Two guards keep this from regressing: each local
search now builds an `Opt` that belongs to it (a closure shared one through a `Core.Box`,
which is a data race in NLopt's C state and a sufficient explanation for the silent exit),
and `batch > 1` is now **rejected** unless `parallel = true`, which it previously was not —
`julia -t 4` with `parallel = false` really did run the local stage on four threads.

**Only half the run is parallel.** The Sobol stage divides by the worker count; the restarts
are sequential by construction, since restart `j` starts from the best point found by
`1..j−1`. The run prints the two halves of the projected runtime separately before
committing. To spend a bigger machine on this problem, raise `--sobol`, not `--restarts`.
The 20-worker cap is a house rule for a shared server, not a hardware limit
([`run_smm.jl:122`](../code/smm/run_smm.jl#L122)).

**A killed run resumes exactly.** The local stage is ~99% of the wall clock and runs for the
better part of a day, so a disconnect or a pre-emption will eventually catch one.
`checkpoint.toml` is written after **every** restart (atomically, temp file plus rename) and
carries the stage, the objective **and the grid it was computed at**; `seeds.toml` is written
once, right after pre-testing, with the surviving seeds. `--resume DIR` reloads both and
re-enters the local stage at the next restart with exactly the mixture the original would
have used — continuation, not a warm start. Verified on Rastrigin: a run resumed at restart
11 reached the identical objective (0.9949590571) in 1096 evaluations against the full run's
2447. `load_resume` refuses rather than guesses if the parameter count, restart budget or
search grid differ.

### The three diagnostics that travel with an estimate

None of these existed before 6 September 2026; the numbers they now produce used to be
recollections from a review conversation.

| script | what it answers | what it does **not** |
|---|---|---|
| [`jacobian.jl`](../code/smm/jacobian.jl) | local separation: singular values, condition number, weak directions, every pairwise cosine — saved with the point, boxes, scales, grids, seed and steps | global identification; a small condition number is not precision |
| [`standard_errors.jl`](../code/smm/standard_errors.jl) | sampling uncertainty: the clustered minimum-distance sandwich under equal and efficient weights | simulation error, the weighting choice, specification error |
| [`sensitivity.jl`](../code/smm/sensitivity.jl) | how the argmin moves when one target moves — all nine parameters jointly re-estimated at each perturbed target | identification or robustness evidence. A point that reports `MAXEVAL_REACHED` or `on_bound` is censored, not a slope |

Each refuses to run on missing inputs rather than substituting a plausible one:
`standard_errors.jl` will not compute its own Jacobian, and `sensitivity.jl` will not
substitute a per-observation SD for a moment standard error.

**Moment standard errors are now available and are not the SDs.** `[moment_cov]` in the
targets file carries the cluster-robust covariance of the ten targeted moments, clustered on
the family over 1,633 clusters. The standard errors are **2.1–7.7% of the cross-sectional
SDs** — `mean_c_p` has se 0.0355 against sd 1.699 — and the moments correlate up to +0.676,
with 4 of 45 pairs above 0.3 in absolute value. A diagonal weight built from SDs would have
been wrong in magnitude and in shape.

---

## Part 3 — Fitted baseline and historical calibration

The current default is the completed fit in [BASELINE_9PARAM.md](BASELINE_9PARAM.md).
The following subsection records the pre-estimation calibration and its provenance;
its values are no longer `PARENT_DEFAULTS`.

### Historical calibration before the completed nine-parameter run

Before the completed run, `PARENT_DEFAULTS`
([`parent_family.jl:114`](../code/src/parent_family.jl#L114)) was an incumbent assembled from
three sources; it supplied the previous `--report-only` and search starting point:

| parameter | value | where it comes from |
|---|---|---|
| `phi_2` | 0.14183751 | the **2026-08-28 six-parameter run** (`output/smm_runs/2026-08-28_130810`) |
| `sigma_1_0` | −0.45749712 | same run |
| `sigma_1_1` | −0.06340019 | same run |
| `sigma_2_0` | −3.39554185 | same run |
| `sigma_2_1` | −0.02870211 | same run |
| `phi_3` | 1.0 | set, not estimated |
| `lambda_2` | 1.0 | set, not estimated |
| `R_0` | 81.55 | the HC rescaling, `1.6 × M^(1−σ_3)` with `M = 753.4` |
| `sigma_4_0` | −4.50 | calibrated |

**Those five estimates are not estimates of this specification.** The 2026-08-28 run matched
**six** moments with **six** parameters against *observation-weighted* targets, with HC in
model units and no HC or study-time moment; it also estimated `phi_1_0 = 0.754`, which the
`phi_1 = 1` normalisation has since discarded. Its `Q_final = 9.7e-12` is a just-identified
exact fit on a different objective and **is not comparable** to any `Q` reported now.

### The fit at the incumbent — measured 6 September 2026, grid 30, `simN = 2000`

After the parental-education correction. `Q = 2.9444` (it was 2.9447 before the fix — the
correction moves the objective by 0.009%, which is the honest size of it at the calibration).

| moment | model | data | gap |
|---|---|---|---|
| `mean_c_p` | 3.7021 | 3.1155 | +18.8% |
| `mean_h_p` | 0.3504 | 0.3073 | +14.0% |
| `mean_t_p_early` | 0.4469 | 0.4544 | −1.7% |
| `mean_t_p_late` | 0.2591 | 0.3333 | −22.3% |
| `mean_e_p_early` | 0.3326 | 0.3429 | −3.0% |
| `mean_e_p_late` | 0.3363 | 0.3911 | −14.0% |
| `mean_i_c_early` | 0.1159 | 0.0393 | **+194.9%** |
| `mean_i_c_late` | 0.0525 | 0.0496 | +5.9% |
| `mean_hc_early` | 6.6597 | 6.0737 | **+79.7%** (levels) |
| `mean_hc_late` | 6.6059 | 6.2508 | **+42.6%** (levels) |

Untargeted, at the same point: implied saving rate 30.9%, terminal assets 40.62 units
(\$406,229), leisure 32.6 hrs/wk, **0 invalid simulation cells**, and — from the coverage
diagnostics added in this pass — **7 households of 2,000 above the asset ceiling** (2 from
the initial draw, 5 crossing during the family stage, max 259.0 against a ceiling of 100),
with human capital entirely inside its grid at both ends.

The two HC moments and early study time are where the work is. That is the expected shape —
`R_0`, `phi_3`, `lambda_2` and `sigma_4_0` have never been estimated against anything.

---

## Part 4 — Caveats that must travel with any number from here

Read these before quoting anything above.

**1. `par_time_tot` overlaps leisure, so `phi_2` absorbs the inconsistency.** The `t_p`
target uses `par_time_tot` (active **plus** nearby/supervisory presence) by instruction.
That measure does not fit an exhaustive time budget — per parent,
`leisure + work + Mom_Total_Act = 112.00` exactly, but `leisure + work + par_time_tot = 133.25`,
21 hours over. Since the model enforces `l_p + h_p + t_p = 1` identically, targeting
`h_p` and `t_p` **forces** model leisure to ~33 hrs/wk against the **59.2 hrs/wk this same
dataset measures**. That ~26-hour gap lands in `phi_2`, which is why it fell 0.526 → 0.142.
**Do not read the estimated `phi_2` as a taste-for-leisure parameter.** To restore the
budget-consistent measure, target `(Mom_Total_Act + Dad_Total_Act)/2 / 112` instead, and the
identity closes exactly; `tools/make_smm_targets.py` carries the accounting and the one-line
revert. Related and separate: `t_p` is **parental presence**, not exclusive parental time —
keep that distinction when interpreting preference parameters.

**2. Terminal assets are \$406,426**, against the ~\$250k discussed as reasonable, on an
implied saving rate of 30.9%. Untargeted, and it has drifted further with each
re-specification (\$288,720 at the four-moment stage, \$403,989 at six). The targets
*jointly* imply this: the budget binds every period, so `c_p`, `e_p` and income leave a
residual that is mechanically the saving rate. The report prints both on every run so the
tension stays visible instead of hiding inside a converged objective.

**3. The consumption *profile* is not targeted — only its mean, and nothing in the
parameter set can tilt it.** The slope is the Euler equation,
`c_{t+1}/c_t = (β(1+r))^(1/ρ)`, which contains **no estimated parameter**. `phi_1` is
normalised to 1 and the age-varying `phi_1_1` lever described in earlier versions of this
file **no longer exists** — preferences became time-invariant in `c27a049`. The only lever
left is `beta_0`, calibrated at **0.98** by instruction (was 0.97):

| β | β(1+r) | growth/yr | over 16 yrs |
|---|---|---|---|
| 0.97 | 0.9991 | −0.06% | −1% |
| **0.98** | 1.0094 | +0.63% | **+10.5%** |
| 0.99 | 1.0197 | +1.31% | +23.1% |
| — | — | — | *data: +21.8%* |

So 0.98 recovers about half the observed tilt; ~0.989 would match it. To target the profile
properly, split `c_p` early/late and estimate `beta_0` against it — the same trick used for
`sigma_1_1` and `sigma_2_1`. **That changes the baseline for the notebook and the
counterfactuals too, not only the estimation**, so it goes through the advisor.

**4. Group means are not age profiles.** `σ_j,t = exp(σ_j0 + σ_j1(t−1))` is monotone by
construction, while the underlying investment profile is **U-shaped** — 0.353 at age 1, a
trough of 0.241 at 12, then nearly triples to 0.650 by 17. Two group means are the most this
functional form can honestly be asked to match. **A good fit on the two group means is not
the model reproducing the age profile.** The late `e_p` group also carries an
end-of-horizon spike: with the age-18 handoff approaching, investment pays off immediately
and the model front-loads it, which lifts the late mean by ~43%. That is the terminal
condition, not the elasticity slope, and `sigma_2_1` will partly absorb it.

**5. Simulated assets run off the solved grid.** Policies are interpolated with `Flat()`
extrapolation, so a state above the top asset node silently reuses the policy there.
Measured at the incumbent: **7 households of 2,000 (0.35%) are above the ceiling at some
point**, 2 of them from the initial draw onward and **5 crossing during `t = 1..17`**, with a
maximum of 259.0 against a grid ceiling of 100. The handoff column matters most — it becomes
the child's initial assets. The initial draw is `LogNormal(0.296, 1.402)` and is deliberately
**not** clamped, since clamping would distort the wealth distribution to flatter a grid.
Small tail mass is not by itself proof of small policy or estimation error; testing the grid
range and node placement within the 30-node cap is an open Tier-0 item.

**6. No standard errors, and no sensitivity analysis.** Equal weights are an assumption the
over-identified design makes binding; the moment covariance has not been estimated, the
weighting matrix has not been justified, and no parameter uncertainty is reported. A
full-rank local Jacobian is not inference. The target-moment response exercise — perturb one
target, jointly re-estimate all nine parameters, plot all 90 curves — has **not** been
implemented or run. Both are Tier 2 in [`REVIEW_TRIAGE.md`](REVIEW_TRIAGE.md) and are
required before conclusions travel, not optional if time remains.

**7. The parent's terminal continuation ignores parental education.** The college offset is
not applied inside `terminal_value_surface`'s `max(v_college, v_work)`, so the terminal value
is evaluated as if `BothCollege = 0` even though the parent carries that state and other
parental-education channels do operate. Open Tier-0 item; it will move the estimates.

---

## Why a run is affordable

The estimated parameters are **all** parent-block, so the child lifecycle, its transfer stage
and the terminal value spline depend on none of them. They are solved **once per process** at
startup and reused for every evaluation — exact, not an approximation. Each evaluation is
then just: build the parent, backward-induct, simulate. 98% of that is `solve_model!`.

This invariant is load-bearing, so it is **enforced**: `moments.jl` errors at load time if any
name in `SMM_PARAMS` is not a field of `PARENT_DEFAULTS`. Adding a child parameter (`rho`,
`omega`, `psi_terminal`, `kappa_terminal`, …) would silently keep reusing a stale child solve
and report a converged fit for a model it never solved — wrong answers, no error. If a child
parameter genuinely has to be estimated, the fix is to move `build_child_value()` inside
`smm_objective` and pay a full child solve per evaluation, not to delete the guard.

Only the Sobol stage parallelises; the restarts are sequential by construction. To spend a
bigger machine on this problem, raise `--sobol`, not `--restarts`.
