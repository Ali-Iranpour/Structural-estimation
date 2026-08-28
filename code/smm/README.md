# SMM — three parent moments

Estimates three parent-block parameters so the simulated model reproduces three
averages from PSID/CDS. Baseline only: nothing here touches the child lifecycle,
the counterfactuals, or the belief machinery.

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
| `--sobol N` | Pre-testing points (default 200). Cheap: these run in parallel. |
| `--restarts N` | Local searches (default 10). **Each one costs ~15 min.** |
| `--grid N` | Parent `Na = Nhc` for the *search* (default 30). `20` is 2.5× faster. |
| `--every N` | Seconds between progress lines (default 2). |
| `--outdir P` | Write the run folder to `P` instead of `output/smm_runs/<stamp>/`. |
| `--serial` | Everything on one process. Slowest, but the easiest to debug. |

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
full grid. This system is just-identified and well-behaved: in testing, restart 1
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
| mean parental leisure | `(leis_mom + leis_dad)/2 / 112` | 0.529 (= 59.2 hrs/wk) | 1,046 |
| mean investment, **child ages 1–9** | `m_method2_final_w99` | 0.353 (= $3,532/yr) | 8,178 |
| mean investment, **child ages 10–17** | `m_method2_final_w99` | 0.441 (= $4,414/yr) | 7,182 |

The pooled investment mean (0.394, N = 15,360) is still generated and printed, but
it is **not targeted** — it carries no information the two age groups don't, and
including it would make the system over-identified. `SMM_MOMENTS` in `moments.jl`
is the single place that decides which set is live.

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

Four parameters, one per moment — **just-identified**, so the objective can in
principle reach zero. That is deliberate for a first estimation: if the fit is
bad, you know it is the *model* failing, not a shortage of free parameters. It
also makes the weighting matrix irrelevant at the optimum.

| Parameter | Moves | Bounds | Incumbent |
|---|---|---|---|
| `phi_1_0` | consumption weight → mean `c_p` | [0.2, 5.0], log scale | 1.00 |
| `phi_2_0` | leisure weight → mean `l_p` | [0.05, 20.0], log scale | 0.50 |
| `sigma_2_0` | **level** of HC elasticity to `e_p` → early investment | [−5.0, −0.5] | −1.80 |
| `sigma_2_1` | **age slope** of that elasticity → late investment | [−0.05, 0.05] | 0.02 |

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
buy a good fit on three means, that is worth knowing before the numbers travel.

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
three simulated means and the three data means. Minimising it is hard because `Q`
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
| `../src/tiktak.jl` | The optimizer (Arnoud, Guvenen & Kleineberg 2022). Shared. |
| `../smm.jl` | **Legacy**: the older 14-parameter, 12-moment estimation. Left in place; not used by anything here. |
