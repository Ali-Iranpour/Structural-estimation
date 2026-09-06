# SMM — ten parent moments

Estimates **nine** parent-block parameters so the simulated model reproduces **ten**
moments from PSID/CDS: consumption, work hours, parental time, monetary investment,
the child's study time, and the level of child skill — the last four split by child
age. Baseline only: nothing here touches the child lifecycle, the counterfactuals,
or the belief machinery.

The design is **over-identified by one**, not just-identified. `Q` cannot reach zero,
the weighting is a real assumption, and a residual gap is not by itself a bug.

If this is your first time running an SMM, read *What SMM is doing here* at the
bottom first — it is four paragraphs and the rest of this file will make more
sense afterwards.

## Run it

Julia 1.11 is installed for this account under `juliaup`. If `julia --version`
says 1.12 (or `julia` is not found), put juliaup's bin directory first and pin
1.11 — the `Manifest.toml` was resolved against 1.11:

```bash
export PATH="$HOME/.juliaup/bin:$PATH"
```

Then, always with `+1.11`:

```bash
cd code/smm
julia +1.11 --project=../.. run_smm.jl --report-only    # ~1 min, no search
julia +1.11 --project=../.. run_smm.jl --quick          # ~15 min, smoke test
julia +1.11 --project=../.. run_smm.jl                  # the real run
```

**Start with `--report-only`.** It solves the model once at the current
calibration, prints how far each moment is from its target, and stops. That is
the fastest way to see where you stand, and it exercises every part of the
machinery except the optimizer.

For anything long, don't hold it in your terminal — the run writes its own log,
so detach it and read the file:

```bash
nohup julia +1.11 --project=../.. run_smm.jl > /dev/null 2>&1 &
tail -f ../../output/smm_runs/<timestamp>/run.log
```

### Where the output goes

Every run creates a fresh timestamped folder — nothing is ever overwritten:

```
output/smm_runs/2026-08-27_155123/
├── run.log          full console transcript, exactly as it appeared
└── estimates.toml   estimated parameters, Q before/after, budget, git commit
```

Use `--outdir <path>` to send a throwaway run somewhere else, e.g.
`--outdir ../../temp/try1`, so experiments don't accumulate in `output/`.

### Flags

| Flag | Meaning |
|---|---|
| `--report-only` | Print the fit at the current calibration and stop. No search. |
| `--quick` | Small grids and a tiny budget. Smoke test, **not** an estimate. |
| `--procs N` | Worker processes. Default 20 — see below. |
| `--sobol N` | Pre-testing points (default 1000). Cheap: these run in parallel. |
| `--restarts N` | Local searches (default 100). **Each one costs ~15 min.** |
| `--grid N` | Parent `Na = Nhc` for the *search* (default 30). `20` is 2.4× faster, and the winner is then re-optimised at the full grid. |
| `--refine N` | Evaluations for that full-grid re-optimisation (default 200). Only runs when `--grid` differs from the report grid. |
| `--local-evals N` | Cap per local search (default 2000). |
| `--polish-evals N` | Cap for the final polish (default 4000). |
| `--every N` | Seconds between progress lines (default 2). |
| `--outdir P` | Write the run folder to `P` instead of `output/smm_runs/<stamp>/`. |
| `--serial` | Everything on one process. Slowest, but the easiest to debug. |
| `--resume DIR` | Continue a killed run from `DIR/checkpoint.toml`. Exact continuation, not a warm start — see below. |

### Resuming a killed run

The local stage is ~99% of the wall clock and runs for the better part of a day, so on a
server a disconnect, a wall-clock limit or a pre-emption will eventually catch one. The run
writes `checkpoint.toml` after **every** restart (atomically, via a temp file and rename),
and `seeds.toml` once after pre-testing.

```bash
cd code/smm && julia --project=../.. run_smm.jl --resume ../../output/smm_runs/2026-09-06_014210
```

This re-enters the local stage at the next restart with the saved incumbent **and the
saved pre-testing seeds**, so restart *j* sees exactly the mixture it would have seen in
the original run — continuation is exact, not approximate. The Sobol stage is skipped
(its evaluations are the expensive part; the points themselves are deterministic). The run
directory is reused and its log appended, so the original transcript survives.

Pass the same `--restarts` and `--grid` you used originally. The loader refuses rather than
guesses if the parameter count, restart budget or search grid differ from the saved run.

## How many cores does it use?

**20 worker processes, out of 112 on the machine (18%).** `haflinger` is a
shared server — the cap is a house rule in `run_smm.jl` (`WORKER_BUDGET = 20`),
not a hardware limit, and it binds long before RAM or cores would:

```
workers = min( 20 ,  CPU_THREADS - 1 ,  RAM_GB / 2 )
            ^^        ^^^^^^^^^^^^^^^    ^^^^^^^^^^
        house rule     111 here          251 here
```

The run prints which of the three bound. Override with `--procs N`, but on a
machine other people are using, raise it by agreement, not because the cores
look idle. Each worker also gets **one BLAS thread** — without that, 20 processes
each open a BLAS pool sized to all 112 cores and the machine thrashes.

**Threads are deliberately not used.** NLopt.jl is not thread-safe in this
project — with threads the objective killed the process with exit 0 and no error
message. Worker *processes* each own their NLopt state, so the hazard cannot
arise. `Threads.@threads` must not be reintroduced here.

### Only half the run is parallel — this matters for your budget

| Stage | Parallel? | Why |
|---|---|---|
| Sobol pre-testing | **yes** | N independent evaluations, nothing shared. |
| Local restarts | **no** | Restart `j` starts from the best point found by restarts `1..j-1`. It cannot begin before `j-1` finishes. |

So the local stage runs on one core no matter how many workers you start, and on
a wide machine it dominates the wall clock. The run prints the two halves
separately before the search begins:

```
projected runtime
  sobol stage      201 evals / 20 workers  =    1.7 min   (parallel)
  local stage      600 evals, sequential    =   49.5 min   (cannot be parallelised)
  total                                        51.2 min
```

**The practical consequence: to spend a big machine on this problem, raise
`--sobol`, not `--restarts`.** More Sobol points are nearly free (they divide by
20) and they give the local stage better seeds to start from. More restarts are
paid for one at a time.

## Making the local stage faster

Since workers cannot help here, the only levers are *cheaper evaluations* and
*fewer of them*. Measured on this machine, in order of value:

**1. Search on a coarser grid — 2.5× (`--grid 20`).** 98% of an evaluation is
`solve_model!`, and its cost scales with the parent's `Na × Nhc`:

| parent grid | solve | simulate | `mean_c_p` | `mean_l_p` | `mean_e_p` |
|---|---|---|---|---|---|
| `Na=Nhc=30` | 11.62 s | 0.24 s | 3.0153 | 0.4703 | 2.1884 |
| `Na=Nhc=20` | 4.78 s | 0.03 s | 3.0149 | 0.4701 | 2.1924 |
| `Na=Nhc=15` | 2.71 s | 0.01 s | 3.0279 | 0.4693 | 2.2179 |

At 20 the targeted moments move by **0.01%, 0.04% and 0.2%** — against the 3%,
11% and 461% gaps the estimation exists to close. The optimizer does not need
resolution the answer needs, so `--grid 20` searches cheap while the reported fit
is still re-solved at 30. Below 20 the drift starts to show (`e_p` 1.3% at 15).

**2. Fewer restarts — linear (`--restarts 5`).** Each restart costs ~15 min at
full grid. In testing, restart 1
alone drove `Q` from 0.081 to 1e-5. Ten restarts is insurance against local
minima that this objective has not shown any sign of. Pair a cut here with a
larger `--sobol`, which is free and buys back the same insurance.

**3. Do not touch `simN`.** It is 2% of the cost, and cutting it to 500 moved
`c_p` *more* (3.0275) than halving the grid did — that is simulation noise, which
is the one thing common random numbers exist to keep out of the objective.

Combining 1 and 2, `--grid 20 --restarts 5 --sobol 400` is roughly **30 minutes**
instead of 2.6 hours, and the reported fit is still at full resolution.

What is *not* worth doing: parallelising the restarts. `tiktak.jl` supports a
`batch` argument, but its measured cost on Rastrigin is severe (`f` 2.985
sequential → 6.965 at batch 4) — it degrades TikTak toward plain multistart,
which is the one thing the algorithm exists to beat. The two levers above are
larger and cost nothing.

### A stopping-rule bug this uncovered — fixed 2026-08-27

`tiktak.jl`'s local searches originally stopped on `ftol_rel` and `maxeval` only.
`ftol_rel` tests `|Δf| ≤ ftol_rel · |f|`, so **as `f → 0` the threshold collapses
with it and the test can never be satisfied.** A just-identified SMM drives `Q` to
~0 by construction, so every restart ran to `maxeval = 2000` regardless of having
converged. Measured: restart 1 reached `Q ≈ 0` at evaluation 61 and was still
running at 290.

At 15.5 s an evaluation that is **8 hours per restart instead of 15 minutes** —
the default run would have taken days, not hours. `local_ftol_abs` and
`local_xtol_rel` now stop it: absolute criteria work at any scale, and a collapsed
simplex means converged whatever `f` is worth there. The self-test still passes
(sphere reaches 7.7e-45), so accuracy is unaffected.

## Watching a run

Progress prints every 2 seconds (`--every N` to change), to both the console and
`run.log`:

```
  sobol      140/201  70%   best Q    0.38734   0.9 min elapsed, ~0.4 min left
  sobol    complete: 201 evaluations, best Q 0.31552, 1.8 min
  restart   1/10   eval   240   this Q    0.04120   best Q    0.03310   6.2 min
  restart   1/10  DONE   this    0.03310   best Q    0.03310   8.1 min, ~73 min left
```

`best Q` should fall and then flatten. If it is still dropping at the last
restart, the budget was too small — raise `--sobol` first.

The Sobol lines need a word of explanation. `pmap` hands the whole batch to the
workers and returns only when all of it is done, so the optimizer's own callback
fires *after* the stage rather than during it. Instead each worker reports its
value through a `RemoteChannel` the moment it finishes, and the master prints as
they land — which is why the count can jump by several at a time.

## What is being matched

| Moment | Data source | Data mean | N |
|---|---|---|---|
| mean consumption | `cons_exhous_real_w99` | 3.158 (= $31,577/yr) | 6,742 |
| mean **work** hours | `(wh_mom+wh_dad)/2 / 112` | 0.307 (= 34.4 hrs/wk) | 15,665 |
| mean **child time**, ages 1–9 | `par_time_tot / 112` | 0.4672 (= 52.3 hrs/wk) | 475 |
| mean **child time**, ages 10–17 | `par_time_tot / 112` | 0.3232 (= 36.2 hrs/wk) | 590 |
| mean investment, ages 1–9 | `m_method2_final_w99` | 0.353 (= $3,532/yr) | 8,178 |
| mean investment, ages 10–17 | `m_method2_final_w99` | 0.441 (= $4,414/yr) | 7,182 |

**Leisure `l_p` is no longer targeted.** `l_p = 1 - h_p - t_p` identically, so
targeting leisure pins the *sum* of work and child time and says nothing about the
split — and the split is where the model was wrong. The 2026-08-27 estimate
matched leisure *exactly* while working 29.6 hrs/wk against 34.4 in data and doing
23.2 hrs of childcare against 18.2: two errors that cancel inside `l_p` and are
invisible to it. `l_p` is still printed, as the residual check that the time budget
closes.

**Which active-time variable.** `Mom_Total_Act` / `Dad_Total_Act`, **not**
`par_time_act` or `parent_Act`. Only the first pair closes the identity
`leisure = 112 - work - active` per parent — verified on the data:

| candidate | mom | dad | want |
|---|---|---|---|
| `Mom_Total_Act` / `Dad_Total_Act` | **112.00** | **112.00** | 112.00 ✓ |
| `par_time_act` | 117.28 | 124.60 | ✗ |
| `parent_Act` | 117.28 | 124.60 | ✗ |

`par_time_act` and `parent_Act` are identical household-level "any parent"
measures; either would break the time budget.

**Why `t_p` is split but `h_p` is not.** `h_p` is flat in child age (0.3062 early
vs 0.3080 late) — one pooled mean, on 15,665 observations. `t_p` **halves** over
the family stage (30.3 → 9.7 hrs/wk, late/early **0.512×**) and does so
*monotonically*, which is exactly the shape `exp(sigma_1_0 + sigma_1_1(t−1))` can
produce. That makes `sigma_1_1` better identified than its `sigma_2_1`
counterpart, whose investment profile is U-shaped and cannot be matched by a
monotone form.

Targets are frozen in `Input/smm_targets_baseline.toml`, generated by
`tools/make_smm_targets.py`. Julia never reads Stata — regenerate with:

```bash
python3 tools/make_smm_targets.py
```

### Units

**One model unit = $10,000/year.** Confirmed three ways: `ASSET_RESCALE = 10`;
the model's mean after-tax household income is 5.23 model units = $52,264, a
plausible US figure; and the older targets in `docs/SMM.md` use the same scale.

**Time is a share of the 112-hour non-sleep week, per parent.** The model splits
`l_p + h_p + t_p = 1`; the data builds leisure as `112 − own work − own active
childcare`, where 112 = 168 less a 56-hour sleep allowance. Same identity —
verified on the data, `mean(leisure + work + active) = 112.00` exactly.

**Per parent, not per household.** `wage_func` multiplies by 2, so one modelled
adult stands for two earners sharing one time allocation. The data counterpart is
the *average* of mother and father. Using `leis_hh` would double the target.

## What is being estimated

**Nine parameters against ten moments — over-identified by one.** `Q` cannot reach
zero and the weighting matrix is *not* irrelevant at the optimum, so equal weights are a
real assumption. Counting nine against ten establishes nothing about identification on its
own; what does is the residual Jacobian — **and that is now a saved artefact, not a
recollection**. Run `jacobian.jl` and read `output/identification/<dir>/jacobian.toml`.

Measured 2026-09-06 at the incumbent, grid 30, central differences at 0.5/1/2% of each
box width, columns scaled to a full-box move:

| columns | condition number | smallest singular value | thin-SVD rank |
|---|---:|---:|---|
| 9 (current) | 51.2 | 0.266 | 9 of 9 |
| 10, adding `sigma_4_1` | 228.9 | 0.060 | 10 of 10 |
| 11, adding `sigma_4_1` and `mu_1` | 183.4 | 0.089 | 10 of **11** — one direction is unidentified by construction |

`sigma_min` is stable across the three steps (spread 7% of its level), so it is the model's
and not the finite difference's.

**The pairwise cosines reproduce; the condition-number ratio does not.** Cosines are
invariant to column scaling and come out as reported — `sigma_4_0`/`mu_1` **0.991**,
`sigma_4_0`/`sigma_4_1` **0.814**, `sigma_4_1`/`mu_1` **0.807**. Condition numbers are
*not* scale-invariant, and the previously circulated 49.2 → **1067.1** does not reproduce:
under a stated box for `sigma_4_1` of [−0.05, 0.05] the same comparison is 51.2 → **228.9**,
a 4.5× degradation rather than 21.7×. The direction of the conclusion survives; the
magnitude was never reproducible because the box it depended on was never recorded.

**A finding that changes the emphasis.** Among the nine parameters actually estimated, the
worst-separated pair is **`sigma_1_0` vs `sigma_1_1` at 0.908** — *higher* than the
`sigma_4_0`/`sigma_4_1` 0.814 that is the stated reason for leaving `sigma_4_1` out. Both
`t_p` and `i_c` are split at the same two age groups, so if 0.814 disqualifies a slope
parameter, 0.908 is a problem for one already in the set. Take this as an argument for
richer age moments, not for dropping `sigma_1_1`.

**Estimate correlations are worse than the cosines suggest.** From the sandwich
(`standard_errors.jl`): `phi_3`/`R_0` **−0.998**, `R_0`/`sigma_4_0` **+0.987**,
`R_0`/`sigma_1_0` **+0.986**. Valuation against technology is the binding problem, and it
is not visible in a pairwise column cosine.

**Two scales, deliberately.** Level moments are scaled by their own target so every
residual is a proportional error. The two HC moments are means of *logs*, where the
residual is already proportional, so they are scaled by 1 — dividing them by a log
W-score of ~6.1 shrank them 6.1× and made a 60% error in the level of human capital
score like a 7.7% miss. See `moment_scale` in `moments.jl`.

**Ages are matched on both sides.** The data is weighted equally per child age, as
the simulation is, and the HC moments start at child age 3 because the composite is
not administered earlier. See `SMM_AGE_HC_LO` and `AGE_HC_LO`.

| Parameter | Moves | Bounds | Link | Incumbent |
|---|---|---|---|---|
| `phi_2` | leisure weight → `h_p` | [0.01, 20.0] | log | 0.1418 |
| `phi_3` | parents' weight on child skill → `t_p`, `e_p` | [0.05, 20.0] | log | 1.0 |
| `lambda_2` | child's weight on skill → `i_c` | [0.05, 20.0] | log | 1.0 |
| `R_0` | HC technology TFP → the **level** of log HC | [5.0, 300.0] | log | 81.55 |
| `sigma_1_0` | **level** of HC elasticity to parent *time* → early `t_p` | [−4.0, −0.2] | level | −0.4575 |
| `sigma_1_1` | **age slope** of that elasticity → late `t_p` | [−0.20, 0.05] | level | −0.0634 |
| `sigma_2_0` | **level** of HC elasticity to *money* → early `e_p` | [−5.0, −0.5] | level | −3.3955 |
| `sigma_2_1` | **age slope** of that elasticity → late `e_p` | [−0.05, 0.05] | level | −0.0287 |
| `sigma_4_0` | HC elasticity to the child's *own study* → `i_c` | [−6.0, −1.0] | level | −4.50 |

`phi_1` and `lambda_1` are **normalised to 1** — utility is defined only up to relative
weights, so two of the five must be pinned. `sigma_4_1 = 0.02` and `mu_1 = −0.04` are held
fixed; see *Identification* below for why, and for the qualification that goes with it.

`sigma_1_0` is the right partner for `t_p` on the model's own evidence:
`parent_family.jl` records that `tau_p` sits at 0.011–0.023 for *every* `phi_2`
from 0.05 to 3.0, because the FOC scales with `phi_2` on both sides — "tau_p is
set by sigma_1 and the value of the child's HC". So `phi_2_0` identifies work and
`sigma_1_0` identifies child time, through separate channels.

### β is calibrated, not estimated

`beta_0 = 0.98` (was 0.97), set by instruction — **not** estimated, because
consumption enters as a single pooled mean and nothing in the objective identifies
patience. Consumption was flat because `β(1+r) = 0.97×1.03 = 0.9991`, i.e. the
Euler condition was almost exactly balanced:

| β | β(1+r) | growth/yr | over 16 yrs |
|---|---|---|---|
| 0.97 | 0.9991 | −0.06% | −1% |
| **0.98** | 1.0094 | +0.63% | **+10.5%** |
| 0.99 | 1.0197 | +1.31% | +23.1% |
| — | — | — | *data: +21.8%* |

So 0.98 recovers about half the observed tilt; ~0.989 would match it. To target the
profile properly, split `c_p` early/late and let `beta_0` be estimated against it —
the same trick used for `sigma_1_1` and `sigma_2_1`. **Note this changes the
baseline for the notebook and counterfactuals too, not only the estimation.**

Bounded parameters are searched on a linked (log) scale so a step can never
produce a negative weight.

### Why investment is split by age

`sigma_2_t = exp(sigma_2_0 + sigma_2_1·(t−1))`, so `sigma_2_1` is an **age slope
that compounds from age 1** — there is no kink at 9, and the split is a property
of the *moments*, not the model. A single pooled mean of `e_p` cannot separate a
slope from a level: many `(sigma_2_0, sigma_2_1)` pairs reproduce the same
average, so adding `sigma_2_1` to a 3-moment design would be **under-identified**
— it would return an answer determined by the Sobol seed, not the data. The
second investment moment is what pins the slope down.

### Two caveats on the slope

**The data profile is U-shaped; the model's is monotone.** Investment falls from
0.353 at age 1 to a trough of 0.241 at 12, then nearly triples to 0.650 by 17.
`exp(sigma_2_0 + sigma_2_1·(t−1))` cannot bend. Two group means are the most this
functional form can honestly be asked to match — a good fit on them is *not* the
model reproducing the age profile.

**The late moment contains an end-of-horizon spike.** The model's `e_p` runs
2.83 → 4.09 → 7.17 over ages 15–17: with the age-18 handoff approaching,
investment pays off immediately and parents front-load it. Ages 16–17 are 2 of 8
years in the late group but lift its mean from 2.07 to 2.96 — a 43% distortion.
That is the *terminal condition*, not the elasticity slope, so `sigma_2_1` will
partly absorb it. The data rises at 16–17 too (college spending), roughly 11×
less steeply. To exclude it, set the late group to ages 10–15 in **both**
`AGE_SPLIT` handling in `tools/make_smm_targets.py` and `model_moments`.

## Two things to know before you read a result

**1. The moments are not independent.** The budget binds every period:

```
c_p + e_p + saving = (1+r)·a + after-tax income + y
```

The model currently spends 2.19 on `e_p` against a data value of 0.39. Cutting
investment frees ~1.8 units, which pushes consumption up *on its own* — so fixing
the investment moment may largely fix consumption for free. Do not read the three
moments as three independent successes.

The flip side: the targets *jointly* imply a saving rate. At the current wage
process they leave `5.83 − 3.16 − 0.39 = 2.27` per period, i.e. **39% of
resources**, which over 17 years accumulates to far more than the ~$250k terminal
assets discussed elsewhere. The run prints the implied saving rate and terminal
assets on every report so this tension stays visible instead of hiding inside a
converged objective.

**2. SDs are reported but not targeted.** The model's only cross-sectional
heterogeneity is a 5-node wage shock plus initial asset, HC and college draws.
It cannot reach the data's dispersion — leisure SD is 7.4× too small, consumption
4× too small. Targeting SDs now would push parameters to extremes chasing
variance the model structurally cannot generate, and damage the means doing it.
Adding heterogeneity is a model change, not an estimation setting.

## Reading the output

`Q` is the weighted relative distance — a sum of squared percentage gaps, so
`Q = 0` is a perfect match and `Q ≈ 21` (the current incumbent) means the moments
are badly off. The report prints each moment in model units *and* in dollars or
hours per week, because "0.53" is hard to sanity-check and "59 hours a week" is
not.

The untargeted block underneath is what tells you whether a good `Q` is
believable: if the saving rate or terminal assets have gone somewhere absurd to
buy a good fit on ten moments, that is worth knowing before the numbers travel.

## What SMM is doing here, in four paragraphs

The model has parameters nobody can observe directly — how much parents value
leisure, how productive money is at building a child's human capital. But for any
*guess* at those parameters the model can be solved and a cohort simulated, which
produces simulated counterparts of things that *are* observed: average
consumption, average leisure, average investment.

Simulated Method of Moments picks the parameters that make the simulated averages
line up with the averages in the data. "Method of moments" because it matches
moments (here, means) rather than a likelihood; "simulated" because this model has
no closed form, so the moments have to come out of a simulation.

The thing being minimised is `Q`, the summed squared relative gap between the
ten simulated moments and their ten data counterparts. Minimising it is hard because `Q`
has no derivative anyone can write down and may have several local minima — hence
TikTak (`../src/tiktak.jl`), which scatters Sobol points over the parameter box to
find promising regions, then runs local searches from the best of them.

One detail that makes the whole thing work: **common random numbers**. Every
evaluation builds the model with the same `seed`, so the simulated shocks are
identical across parameter guesses. Without that, `Q` would jump around with
simulation noise and no derivative-free optimizer could converge — it would be
chasing the random number generator instead of the parameters.

## Files

| File | What |
|---|---|
| `run_smm.jl` | Driver: worker setup, budget, progress, TikTak, logging, reporting. |
| `moments.jl` | Targets, model moments, objective, fit report. The economics. |
| `jacobian.jl` | Saves the residual Jacobian with its full metadata: singular values, condition number, weak directions, pairwise cosines. **Run this before arguing about identification** — the numbers in this file's text came from it. |
| `standard_errors.jl` | The clustered minimum-distance sandwich, from a saved Jacobian plus `[moment_cov]`. Sampling uncertainty only — read its header for what it does not cover. |
| `sensitivity.jl` | The target-moment response exercise: perturb one target, jointly re-estimate all nine, 90 curves. Checkpoint/resume built in. |
| `../src/tiktak.jl` | The optimizer (Arnoud, Guvenen & Kleineberg 2022). Shared. |
| `../../archive/smm_14param_legacy.jl` | **Retired** 2026-09-06: the older 14-parameter, 12-moment estimation. Non-functional against the current model; kept in `archive/` as a record, referenced by nothing. |
