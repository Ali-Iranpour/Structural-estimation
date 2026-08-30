# Estimation: SMM by TikTak

How the parent block is estimated: what TikTak does, what we set, which moments are
matched, and — the part that matters for the paper — **every place we depart from the
published algorithm, and what the fit does not deliver**.

```bash
cd code/smm
julia +1.11 --project=../.. run_smm.jl --report-only    # ~1 min, no search
julia +1.11 --project=../.. run_smm.jl --quick          # smoke test
julia +1.11 --project=../.. run_smm.jl                  # the real run
```

Flags, runtimes and the parallelism story are in
[`code/smm/README.md`](../code/smm/README.md). This file is the method.

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

Sort ascending and keep the best `N*` as seeds `s₁ … s_{N*}`. This is cheap reconnaissance —
the rejected points are not proved useless, they are dropped because their neighbourhoods
look less promising under the available budget.

### Stage 1 — the first local search

Run the local optimizer from the best seed `s₁`. The result is the **incumbent**.

### Stage 2 — mixing, which is the whole idea

For restart `j > 1`, take the next unused seed `s_j` and the best local minimum so far,
`p*`, and start from the convex combination

```
S_j = (1 − θ_j)·s_j + θ_j·p*,        θ_j = min( max( 0.1, √(j/N*) ), 0.995 )
```

Small `θ` keeps the start near an independent global seed; large `θ` pulls it onto the
incumbent. So the algorithm shifts from exploration to exploitation on its own. At
`N* = 100`: `θ ≈ 0.1` early, `0.5` at `j = 25`, `0.9` at `j = 81`, capped at `0.995` so the
seed never stops contributing.

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
| **`N*` / `N`** | `N* = 0.1N` | `5 / 400 = **1.25%**` | inside the paper's 1–10% guidance but at its floor — a budget choice, since each restart is sequential and costs ~15 min |
| **Stopping rule** | `ftol_rel` | adds `local_ftol_abs = 1e-10`, `local_xtol_rel = 1e-8` | see below |

**The stopping rule is not a detail.** `ftol_rel` tests `|Δf| ≤ ftol_rel·|f|`, so **as
`f → 0` the threshold collapses with it and the test can never be satisfied**. A
just-identified SMM drives `Q` to ~0 by construction, so every restart ran to
`maxeval = 2000` regardless of having converged — measured, restart 1 reached `Q ≈ 0` at
evaluation 61 and was still running at 290. At ~15 s an evaluation that is 8 hours per
restart instead of 15 minutes. Absolute criteria work at any scale, and a collapsed simplex
means converged whatever `f` is worth there. The self-test still reaches `7.7e-45` on the
sphere, so accuracy is unaffected.

Everything else matches the paper exactly: Sobol' pre-testing, ascending sort, keep best
`N*` ([`tiktak.jl:244-245`](../code/src/tiktak.jl)), and the benchmark schedule
`clamp((j/N*)^0.5, 0.1, 0.995)` ([`tiktak.jl:265`](../code/src/tiktak.jl)), with restart 1
pinned at `θ = 0` so it starts purely from `s₁`. Polish is BOBYQA at `1e-10`.

### The six parameters

Just-identified: one parameter per moment, so `Q` can in principle reach zero. That is
deliberate — if the fit is bad you know it is the **model** failing, not a shortage of free
parameters, and the weighting matrix cannot matter at the optimum.

| parameter | what it moves | bounds | link |
|---|---|---|---|
| `phi_1_0` | weight on consumption → `c_p` level | `[0.2, 5.0]` | log |
| `phi_2_0` | weight on parental leisure → the time split | `[0.05, 20.0]` | log |
| `sigma_1_0` | elasticity of HC to parental **time** → `t_p` level | `[−4.0, −0.2]` | level |
| `sigma_1_1` | its age slope → `t_p` early vs late | `[−0.20, 0.05]` | level |
| `sigma_2_0` | elasticity of HC to **money** → `e_p` level | `[−5.0, −0.5]` | level |
| `sigma_2_1` | its age slope → `e_p` early vs late | `[−0.05, 0.05]` | level |

Strictly-positive weights are searched **in logs**, so a step can never propose a negative
weight. The `sigma`s are already log-elasticities (`σ_j,t = exp(σ_j0 + σ_j1·(t−1))`) and are
searched in levels inside a box.

### The six moments

Targets are frozen in [`Input/smm_targets_baseline.toml`](../Input/smm_targets_baseline.toml),
generated by [`tools/make_smm_targets.py`](../tools/make_smm_targets.py) from the Stata
files. Julia never reads `.dta`, so a run is reproducible and a change of target shows up as
a diff.

| moment | data source | target | N |
|---|---|---|---|
| `mean_c_p` | `cons_exhous_real_w99` | 3.1577 | 6,742 |
| `mean_h_p` | `(wh_mom + wh_dad)/2 / 112` | 0.3070 | 15,665 |
| `mean_t_p_early` | `par_time_tot / 112`, ages 1–9 | 0.4672 | 475 |
| `mean_t_p_late` | `par_time_tot / 112`, ages 10–17 | 0.3232 | 590 |
| `mean_e_p_early` | `m_method2_final_w99`, ages 1–9 | 0.3532 | 8,178 |
| `mean_e_p_late` | `m_method2_final_w99`, ages 10–17 | 0.4414 | 7,182 |

**Units.** One model unit = **\$10,000/year** (`ASSET_RESCALE = 10`; the model's mean
after-tax household income of 4.69 units = \$46,924, a plausible US figure). Time is a
**share of the 112-hour non-sleep week, per parent** — 112 = 168 less a 56-hour sleep
allowance. Per *parent*, not per household, because `wage_func` multiplies by 2: one modelled
adult stands for two earners sharing one time allocation.

**Why `e_p` and `t_p` are split early/late.** A single mean cannot separate an age slope from
a level: many `(σ_j0, σ_j1)` pairs give the same overall average, and the optimizer would
slide along that ridge and return whatever its seed was near. Two group means pin both.
`h_p` is *not* split — it is flat in child age (0.3062 early vs 0.3080 late), so one pooled
mean is right, and it carries 15,665 observations against `t_p`'s 1,065 because work hours
are measured for everyone while time diaries exist only for the CDS subsample.

### The objective

Weighted relative distance, `Q(θ) = Σⱼ wⱼ((mⱼ − m̂ⱼ)/sⱼ)²` with `sⱼ = max(|m̂ⱼ|, 0.05)`, so
every moment contributes on a comparable scale regardless of units — without it consumption
(~3) would dominate a time share (~0.3) purely by size. Weights are **equal**: the system is
just-identified, so at the optimum they cannot change the answer, and equal weights add no
unexamined assumption.

**Common random numbers** throughout: every model is built with the same `seed`, so draws
and shock paths are identical across evaluations. Without this the objective is a step
function of simulation noise and no derivative-free method converges — it would be chasing
the RNG, not the parameters. A failed solve returns a large **finite** penalty, never `Inf`
or an exception, so a local search can still form a descent direction away from it.

### The run behind the current calibration

`output/smm_runs/2026-08-28_130810`

| | |
|---|---|
| budget | 400 Sobol' points, 5 restarts, **2,283 evaluations** |
| workers | 20 processes, 166 minutes |
| grids | search at `Na = Nhc = 20`, fit **re-solved and reported at 30** |
| penalised draws | 15 (the model could not be solved there) |
| objective | `Q` **0.497 → 9.7e-12** |

| parameter | was | estimated |
|---|---|---|
| `phi_1_0` | 0.84172991 | **0.75417767** |
| `phi_2_0` | 0.52555078 | **0.14183751** |
| `sigma_1_0` | −0.90 | **−0.45749712** |
| `sigma_1_1` | −0.08 | **−0.06340019** |
| `sigma_2_0` | −3.32678034 | **−3.39554185** |
| `sigma_2_1` | −0.02287337 | **−0.02870211** |

| moment | model | data | gap |
|---|---|---|---|
| `mean_c_p` | 3.1598 | 3.1577 | +0.1% |
| `mean_h_p` | 0.3076 | 0.3070 | +0.2% |
| `mean_t_p_early` | 0.4650 | 0.4672 | −0.5% |
| `mean_t_p_late` | 0.3238 | 0.3232 | +0.2% |
| `mean_e_p_early` | 0.3496 | 0.3532 | −1.0% |
| `mean_e_p_late` | 0.4428 | 0.4414 | +0.3% |

Search on a coarser grid, report on the full one: never quote a `Q` minimised at `Na = 20`.
Dropping 30 → 20 makes an evaluation 2.5× cheaper and moves the targeted moments by
0.01–0.2%, against gaps of 3–461% the estimation exists to close.

---

## Part 3 — Three caveats that must travel with these numbers

An exact fit on six means is exactly as informative as the six means. Read these before
quoting anything above.

**1. `par_time_tot` overlaps leisure, so `phi_2_0` absorbs the inconsistency.** The `t_p`
target uses `par_time_tot` (active **plus** nearby/supervisory presence) by instruction. That
measure does not fit an exhaustive time budget — per parent,
`leisure + work + Mom_Total_Act = 112.00` exactly, but `leisure + work + par_time_tot = 133.25`,
21 hours over. Since the model enforces `l_p + h_p + t_p = 1` identically, targeting
`h_p = 0.3070` and `t_p` **forces** model leisure to 32.9 hrs/wk against the **59.2 hrs/wk
this same dataset measures**. That 26-hour gap lands in `phi_2_0`, which is why it fell
0.526 → 0.142. **Do not read the estimated `phi_2_0` as a taste-for-leisure parameter.** To
restore the budget-consistent measure, target `(Mom_Total_Act + Dad_Total_Act)/2 / 112`
instead, and the identity closes exactly.

**2. Terminal assets are \$403,989** against the ~\$250k discussed as reasonable. The implied
saving rate is 32.9%. Untargeted, and drifting further with each re-estimation (\$288,720 at
the four-moment stage).

**3. The consumption *profile* is not targeted — only its mean.** The model rises 4% from
child age 1 to 17; the data rises **34%**. Profile correlation +0.381. This is not fixable by
re-estimating what is currently in the parameter set: the slope is the Euler equation,

```
c_{t+1}/c_t = (β(1+r))^(1/ρ) = (0.97 × 1.03)^(1/1.5) = 0.9994
```

which contains **no estimated parameter**. `phi_1_0` sets the level; nothing sets the slope.
Two levers exist, and both have a cost:

- **`phi_1_1`** — already wired (`phi_1_vector = [phi_1_0 + phi_1_1·(t−1)]`), currently fixed
  at 0. A measured sweep gives consumption growth of **+4.4 / +22.1 / +38.2 / +53.3%** at
  `phi_1_1 = 0.000 / 0.015 / 0.030 / 0.045`, so ≈0.027 reproduces the data's +34% without
  touching `r`. Interpretable as an age-varying equivalence scale: a 17-year-old costs more
  than a toddler, and `cons_exhous_real` is household consumption.
- **`r`** — would need ≈6%, which fights the instruction to keep `r` low and pushes terminal
  assets further past target.

The same warning applies to `e_p` and `t_p`: their underlying age profiles are **U-shaped**
(investment falls from 0.353 at age 1 to a trough of 0.241 at 12, then nearly triples to
0.650 by 17), while `σ_j,t = exp(σ_j0 + σ_j1(t−1))` is monotone by construction. Two group
means are the most the model can be asked to match. **A good fit on the two group means is
not the model reproducing the age profile.**

---

## Why a run is affordable

The estimated parameters are **all** parent-block, so the child lifecycle, its transfer stage
and the terminal value spline depend on none of them. They are solved **once per process** at
startup and reused for every evaluation — exact, not an approximation. Each evaluation is
then just: build the parent, backward-induct, simulate. 98% of that is `solve_model!`.

Only the Sobol stage parallelises; the restarts are sequential by construction, since
restart `j` starts from the best point found by `1..j−1`. To spend a bigger machine on this
problem, raise `--sobol`, not `--restarts`.
