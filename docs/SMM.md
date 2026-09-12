# Estimation: SMM by TikTak

How the model is estimated: what TikTak does, what we set, which moments are matched,
how the objective is built — and, the part that matters for the paper, **every place we
depart from the published algorithm, and what the fit does not deliver**.

```bash
export PATH="$HOME/.juliaup/bin:$PATH"          # Manifest.toml is resolved against 1.11
uv run --with pandas --with numpy python tools/make_smm_targets.py     # freeze targets
julia +1.11 --project=. code/smm/run_smm.jl --report-only --targets <targets.toml>   # ~1 min, no search
julia +1.11 --project=. code/smm/run_smm.jl --quick --targets <targets.toml>         # smoke test
julia +1.11 --project=. code/smm/run_smm.jl --targets <targets.toml>                 # the real run
```

Flags, runtimes and the parallelism story are in
[`code/smm/README.md`](../code/smm/README.md). This file is the method.

> **Status, 12 September 2026.** The specification is **sixteen parameters against
> seventeen moments** (eleven parent, five child; ten parent moments, seven TAS moments).
> The latest run, `2026-09-11_182836_exp16b`, finished at `Q = 61.80` from an incumbent of
> 313.34, `accepted = true`, no parameter on a bound (`kappa_ParEd` within 5% of one).
> The parent block fits to |t| ≤ 2; what remains is two child-block moments — the
> completion–wealth gap (`kse_w_gap`, −94%) and the ability gap (`kth_ga17_gap`, −47%) —
> which together carry 70% of `Q`. See [Part 3](#part-3--results).
>
> **The 2026-09-11 additions (`sigma_eta`, `sigma_eps`, the gap moments) are a
> preliminary experiment that has not been through the advisor.** Results built on them
> do not circulate. The motivating diagnosis is [`ERRORS.md`](ERRORS.md) P13.
>
> **Baseline promoted 2026-09-12.** `PARENT_DEFAULTS` and `CHILD_DEFAULTS` now carry the
> `exp16b` vector at full precision, so the notebook, `run_all.jl` and the next search's
> incumbent are the same model. **Before the next run** three boxes changed —
> `kappa_0 [−3, 1]`, `kappa_ParEd [−1, 0.5]`, `sigma_4_1 [−0.05, 0.30]` — and the run
> needs no `--init-from`; see [the next run](#the-next-run).

### How the specification got here

| date | parameters / moments | what changed |
|---|---|---|
| 2026-08-28 | 6 / 6 | just-identified parent block, observation-weighted targets, HC in model units |
| 2026-09-06 | 9 / 10 | HC put in the data's units; `R_0`, `phi_3`, `lambda_2`, `sigma_4_0` added; HC moments added; equal weights on a relative scale |
| 2026-09-07 → 09 | 9 / 10 | pilot with `i_c = c_time_hrs` (school **plus** study); **reverted** — see [the parent moments](#the-ten-parent-moments) |
| 2026-09-09 | 10 / 10 | own study only; fixed school schedule deducted from the child's leisure; `sigma_4_1` estimated; HC late window 12–17 |
| 2026-09-10 | 14 / 17 | four child parameters (`kappa_*`) and seven TAS moments; inverse-variance weights from the joint clustered covariance |
| 2026-09-11 | **16 / 17** | `sigma_eta` (HC shock) and `sigma_eps` (taste-shock scale); ability tertiles replaced by `kth_ga17_gap`; `kse_w_gap` and `sd_ga17` added |

Estimates from different rows are **not comparable** — different targets, different
weights, different objectives. Never compare a `Q` across rows.

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
| **Seeding** | Sobol' points only | Sobol' **+ the incumbent** forced into the pool (`extra_seeds`; `--init-from <estimates.toml>` loads it by name) | it competes on function value like any Sobol point, and guarantees the estimate is weakly better than the point we started from. Pass `extra_seeds = []` for the published algorithm exactly |
| **Stopping rule** | `ftol_rel` | adds `local_ftol_abs = 1e-10`, `local_xtol_rel = 1e-8` | see below |
| **Post-search refinement** | not in the paper | when `--grid` differs from the report grid, a short BOBYQA polish **on the full-grid objective**, started from the coarse winner | the coarse and fine objectives have different minimisers, so the coarse argmin is a starting point, not an answer |

`N*/N` is **not** a departure any more: the defaults are `N = 1000`, `N* = 100`
([`run_smm.jl:100-101`](../code/smm/run_smm.jl#L100-L101)), i.e. the paper's own
`N* = 0.1N`. Restarts are sequential and cost ~15 min each, so cutting `--restarts` is the
first thing to give up on a budget — and `--sobol` is the thing to raise, since it divides
by the worker count.

**The stopping rule is not a detail.** `ftol_rel` tests `|Δf| ≤ ftol_rel·|f|`, so **as
`f → 0` the threshold collapses with it and the test can never be satisfied**. The
2026-08-28 just-identified design drove `Q` to ~0 by construction, so every restart ran to
`maxeval = 2000` regardless of having converged — measured, restart 1 reached `Q ≈ 0` at
evaluation 61 and was still running at 290. At ~15 s an evaluation that is 8 hours per
restart instead of 15 minutes. Absolute criteria work at any scale, and a collapsed simplex
means converged whatever `f` is worth there. The self-test still reaches `7.7e-45` on the
sphere, so accuracy is unaffected. The current design is over-identified and `Q` no longer
approaches zero, but the guard stays: it is scale-free and costs nothing.

Everything else matches the paper: Sobol' pre-testing, ascending sort, keep best `N*`, and
the benchmark schedule `clamp((j/N*)^0.5, 0.1, 0.995)`, with restart 1 pinned at `θ = 0` so
it starts purely from `s₁`. Polish is BOBYQA at `1e-10`
([`tiktak.jl:228`](../code/src/tiktak.jl#L228)); `--skip-polish` is the explicit bypass
(`--polish-evals 0` is refused, because NLopt reads 0 as *no limit*).

### The sixteen parameters

Seventeen moments against sixteen parameters: **over-identified by one**. That is
deliberate, and it changes two things from a just-identified design — `Q` **cannot** reach
zero, and the weighting matrix is **not** irrelevant at the optimum. A residual gap is
therefore not by itself evidence of a bug. Counting sixteen against seventeen also
establishes nothing about identification; what does is the residual Jacobian
([below](#identification)).

`phi_1` and `lambda_1` are **normalised to 1.0** and not estimated — utility is defined only
up to relative weights, so two of the five must be pinned (instruction 2026-08-30).
Preference weights are **time-invariant**: the `_1` slopes and per-period vectors on
`phi`/`lambda` were removed in `c27a049`. `R_1`, the age slope of HC productivity, is
**fixed at 0** and `moments.jl` errors if it ever appears in `SMM_PARAMS`. `mu_1 = −0.04`
is held at its calibrated value ([why](#identification)).

Source of truth: `SMM_PARAMS` in [`moments.jl`](../code/smm/moments.jl) (the boxes carry
their history in comments beside each line). `moments.jl` asserts the split is
`(11 parent, 5 child)` at load time.

**Parent block** — eleven, all fields of `PARENT_DEFAULTS`:

| parameter | what it moves | bounds | link |
|---|---|---|---|
| `phi_2` | weight on parental leisure → `h_p` (work; `l = 1 − h − t`) | `[0.01, 20.0]` | log |
| `phi_3` | parents' weight on child skill → `t_p` and `e_p` | `[0.05, 20.0]` | log |
| `lambda_2` | the child's own weight on skill → `i_c` (own study) | `[0.05, 100.0]` | log |
| `R_0` | HC technology TFP → the **level** of `log HC` | `[0.5, 100.0]` | log |
| `sigma_1_0` | elasticity of HC to parental **time** → `t_p` level | `[−4.0, −0.1]` | level |
| `sigma_1_1` | its age slope → `t_p` early vs late | `[−0.20, 0.05]` | level |
| `sigma_2_0` | elasticity of HC to **money** → `e_p` level | `[−5.0, −0.5]` | level |
| `sigma_2_1` | its age slope → `e_p` early vs late | `[−0.30, 0.05]` | level |
| `sigma_4_0` | elasticity of HC to the child's **own study** → `i_c` level | `[−10.0, −1.0]` | level |
| `sigma_4_1` | its age slope → study and HC age profiles | `[−0.05, 0.30]` (was 0.15) | level |
| `sigma_eta` | SD of the i.i.d. log shock to HC → dispersion of `log HC` at 17 | `[0.0, 0.08]` | level |

**Child block** — five, routed to the child constructor (`CHILD_ESTIMATED`). The start
is the block default, `CHILD_DEFAULTS` in [`child_lifecycle.jl`](../code/src/child_lifecycle.jl),
which since 2026-09-12 is the `exp16b` fit:

| parameter | what it moves | bounds | start (exp16b) | link |
|---|---|---|---|---|
| `kappa_0` | psychic cost of college **at mean ability** → the completion level | `[−3, 1]` (was `[−2, 5]`) | −0.3566 | level |
| `kappa_theta` | its ability gradient → completion by ability | `[−10, 0]` | −3.6194 | level |
| `kappa_ParEd` | its parental-education shift → completion by `BothCollege` | `[−1, 0.5]` (was `[−3, 0]`) | −0.1081 | level |
| `kappa_terminal` | parent's terminal weight on retained assets → the transfer, hence retained wealth **and** the college share | `[0.5, 40]` | 8.787 | log |
| `sigma_eps` | SD of the college taste shock `ε₀` → how sharply completion responds to any shifter | `[0.1, 2.0]` | 1.142 | log |

**The three box changes of 2026-09-12**, all numerical, none a specification change:
`kappa_0` narrowed to the measured transition region (71% of the old box was a flat zero
where every completion moment has derivative exactly zero, and exp16b's 1,000 Sobol
points found nothing that beat the seeded incumbent); `kappa_ParEd` given a floor above
its saturation region and a top that admits a small positive value, because exp16b landed
3.6% from the wall at 0 and a sign restriction the data does not ask for is not a reason
to stop there — if it settles at ~0 again, fix it at 0 and drop a parameter; `sigma_4_1`
raised after two fits (0.133, 0.130) sat at 90% of the old ceiling.

These are joint identifying relationships, not exclusive assignments: `kappa_terminal`
moves the college share strongly (a parent that values retained assets transfers less), and
every kappa moves every completion moment.

**Three things about the boxes and links.**

Strictly-positive parameters are searched **in logs**, so a step can never propose a
negative weight — and the log link is also what concentrates the search. A Sobol sequence
uniform in `log θ` is not uniform in `θ`: its density in levels falls like `1/θ`. Over
`R_0`'s box that puts 25% of pre-testing points below 1.9, half below 7.1 and only a
quarter above 26.6. Elasticity coefficients are searched in levels.

Several boxes have been widened after an estimate pinned to a wall — `sigma_2_1` three
times in 0.05 steps, each costing a full estimation, before a large step let it settle at
−0.155; `lambda_2` 20 → 100 after landing at exactly 20.0; `sigma_4_0` −6 → −10 after a
univariate sweep put the optimum near −7.5. A wall is only informative once
[`profile_param.jl`](../code/smm/profile_param.jl) has shown `Q` is not simply flat in
that direction. And `lambda_2` climbing is a **specification question**, not only a box
one: `lambda_1 = 1`, so `lambda_2 = 14` means the child weights skill fourteen times its
leisure — flag it if the fit only holds at implausible values.

**`kappa_0` is on a centred scale.** The psychic cost is
`kappa_0 + kappa_theta·(log θ − m_psychic)` with `m_psychic = 6.263397` frozen in the target
file. Behaviourally neutral — the same device the wage equation already uses — but
uncentred, `log θ` has mean 6.26 and SD 0.035, so `[1, log θ]` has a condition number near
180 and `kappa_0` must move ~180 units to undo one unit of `kappa_theta`. The legacy
uncentred 0.2728 corresponded to 0.0587 at this centring, which was the pilot start.
`CHILD_DEFAULTS.m_psychic` records the centring the fitted `kappa_0` belongs to, and
`check_psychic_centring` errors if a target file carries a different one.

**`R_0` became estimable only once HC was put in the data's units.** Before the rescaling
(`c27a049`) there was no HC moment to identify it against. That rescaling is what separates
the **valuation** parameters (`phi_3`, `lambda_2`) from the **technology** parameters
(`R_0`, `sigma_1`, `sigma_2`, `sigma_4`): both raise investment, and only the resulting HC
level can tell them apart.

### The ten parent moments

Targets are frozen as `targets.toml` inside each timestamped run folder under
`output/smm_runs/`, generated by [`tools/make_smm_targets.py`](../tools/make_smm_targets.py)
from the Stata files. Julia never reads `.dta`, so a run is reproducible and a change of
target shows up as a diff. Values below are from the
[current targets](../output/smm_runs/2026-09-11_182836_exp16b/targets.toml).

| moment | data source | target | N |
|---|---|---|---|
| `mean_c_p` | `cons_exhous_real_w99` | 3.1155 | 6,742 |
| `mean_h_p` | `(wh_mom + wh_dad)/2 / 112` | 0.3073 | 15,665 |
| `mean_t_p_early` | `par_time_tot / 112`, ages 1–9 | 0.4544 | 475 |
| `mean_t_p_late` | `par_time_tot / 112`, ages 10–17 | 0.3333 | 590 |
| `mean_e_p_early` | `m_method2_final_w99`, ages 1–9 | 0.3429 | 8,178 |
| `mean_e_p_late` | `m_method2_final_w99`, ages 10–17 | 0.3911 | 7,182 |
| `mean_i_c_early` | `study_hrs / 112` (own study only), ages 6–9 | 0.0393 | 171 |
| `mean_i_c_late` | `study_hrs / 112` (own study only), ages 10–17 | 0.0496 | 584 |
| `mean_hc_early` | `x_gach` (log PCA composite), ages 3–9 | 6.0737 | 252 |
| `mean_hc_late` | `x_gach` (log PCA composite), ages **12–17** | 6.2589 | 459 |

**The child's time input is own study, with school time fixed** (since 2026-09-09).
`study_hrs` is own study alone — homework, self-study, academic clubs — averaging 3.4
hrs/wk. School hours enter the model as an **exogenous schedule** (`SCHOOL_TIME_BY_AGE` in
`parent_family.jl`: the median `school_hrs` by (Year, Age), ~37 hrs/wk from age 6, frozen
in each run's `targets.toml`), deducted from the child's leisure
`l_c = 1 − t_p − i_c − school_t` and **not** entering the HC technology. Only own study is
chosen and only own study is in the `i_c^σ₄` term.

The pilot of 2026-09-07 to 09-09 targeted `c_time_hrs` = school **plus** study instead
(targets 0.365 / 0.387, ninefold higher). It was reverted because in this model the child
*chooses* `i_c` against its own leisure, so the only way to make a child voluntarily spend
41 hrs/wk is to make it value skill enormously — `lambda_2` climbed 8.7 → 16.8 → exactly
20.0, the box ceiling, across three runs. School attendance is compulsory, not chosen;
reproducing a mandate through a taste parameter fits the moment while attributing it to
the wrong mechanism, and every counterfactual that moves the return to skill inherits that.
Estimates from that pilot are not comparable to anything since.

**Ages are matched on both sides, and each side is weighted the same way.** The targets are
means **over child ages**, equally weighted, because the simulation is — an
observation-weighted pooled mean is a different number and is carried in the file as
`mean_pooled`, unused. Three age ranges are not `1..17`, and each is enforced in the model
too:

- `mean_i_c_*` starts at **child age 6**: the child is not a decision maker before
  `T_CHILD_VOICE = 6`, so `sim_i` is not a choice there.
- `mean_hc_early` starts at **child age 3**: the Woodcock-Johnson composite is not
  administered earlier (`x_gach` has 0 observations at age 1). Averaging the model over 1–9
  against a data group that is really 3–9 was worth **0.110 log points** on its own, 23% of
  the whole HC gap.
- `mean_hc_late` is **12–17**, not 10–17 (since 2026-09-09), so the two HC windows are
  separated by the trough of the investment profile and the slope `sigma_4_1` has
  something to pull against.
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

### The seven TAS moments

A **different sample frame** from the parent block: one row per TAS-linked child (4,248
children, 1,481 family clusters, cluster key `famclust`) against the parent block's one row
per child-year. Every moment is a ratio of means over the whole frame with the subgroup
indicator in the denominator — how the source do-file estimates them — or a difference of
two such ratios, or a sample SD. Order must match `TAS_MOMENTS` in the generator and
`SMM_TAS_MOMENTS` in `moments.jl`; `load_targets` refuses otherwise.

Three frames:

- **CF** (completion frame): `hf_complete == 1`, N = 2,771.
- **A17** (age-17 achievement): CF & `ach_age == 17` & `g_ACH` non-missing, N = 317.
  Value `x = log(g_ACH)` — log, not level, not standardised. This is the frame `m_psychic`
  is computed on.
- **W** (wealth): CF & `ever_strict == 1` & `pwx_strict` non-missing, N = 665. Value
  `W99 = min(pwx_strict, cut) / 10,000`, `cut` = the p99 of `pwx_strict` on the
  `ever_strict` sample (4,448,695 USD); negatives retained.

| moment | frame | definition | data | N | model counterpart | informs |
|---|---|---|---|---|---|---|
| `k0_complete` | CF | share completing a four-year degree by 25 | 0.3226 | 2,771 | share of resimulated children choosing college | `kappa_0` |
| `kth_ga17_gap` | A17 | `mean(x \| y=1) − mean(x \| y=0)` | 0.0312 | 317 | `mean(log hc17[college]) − mean(log hc17[work])`, `hc17 = parent.sim_hc[:, 17]` | `kappa_theta` |
| `kpe_g0_c` | CF | completion where `pared_col == 0` | 0.2108 | 1,319 | college share where `BothCollege == 0` | `kappa_ParEd` |
| `kpe_g1_c` | CF | completion where `pared_col == 1` | 0.6013 | 913 | college share where `BothCollege == 1` | `kappa_ParEd` |
| `kterm_x_strict_w99` | W | `E[min(W, cut)]`, model units | 33.198 | 665 | mean of `sim_a[:, T+1] − transfer`, same cut | `kappa_terminal` |
| `kse_w_gap` | W | `mean(W99 \| y=1) − mean(W99 \| y=0)` | 36.843 | 665 | `mean(retained[college]) − mean(retained[work])` | `sigma_eps` |
| `sd_ga17` | A17 | SD of `x` | 0.0329 | 317 | `std(log hc17)` | `sigma_eta` |

Two estimator forms beyond the plain ratio, both built on `ratio_influence` (which returns
the estimate and the per-cluster influence `psi`): a **difference of two ratios** has
`psi = psi₁ − psi₂` (Stata `lincom`); a **standard deviation** comes from `m₁ = mean(x)`
and `m₂ = mean(x²)` as two ratio moments, `sd = sqrt(m₂ − m₁²)`,
`psi = (psi₂ − 2·m₁·psi₁) / (2·sd)` (Stata `nlcom`). The gap and SD rows of the covariance
are built from these, not treated as independent.

**Why absolute units for `kse_w_gap`.** `sigma_eps` is a scale, so the shifter has to be
in known units; rank-based wealth tertiles would be scale-free and lose that. Timing
caveat: TAS wealth is measured at a median child age of ~29, the model's object is at 18.

**Why the ability tertiles were replaced by the gap** (2026-09-11). The 14-parameter run
`2026-09-10_183649` could not fit completion by ability and by parental education at the
same time: all 430 simulated completers were children of `BothCollege = 1` parents, and
`kpe_g0_c = 0` carried 64% of `Q`. The cause is structural: without an idiosyncratic shock
the model's HC at 17 is a deterministic function of parental resources, and its SD of
`log HC` at 17 was 0.0081 against 0.0329 in the data — **four times too narrow**. Rank
tertiles could not see that miss. The fix is the shock (`sigma_eta`) plus a free
taste-shock scale (`sigma_eps`), and the three moments that identify them: `sd_ga17` for
the dispersion, `kth_ga17_gap` in **absolute** log units for the ability gradient, and
`kse_w_gap` for the scale. Full diagnosis in [`ERRORS.md`](ERRORS.md) P13.

#### The measurement decisions (2026-09-10)

Each foreclosed a different silent error.

**Completion, not entry.** The model's college path is binary and has no dropout: enrol,
study `t_college = 4` years, earn the graduate wage `E = 1`. Nobody enrols without
finishing, so the path *is* a completed degree. Matching it to entry (0.616) would compare a
mechanism that always pays the college premium against a population where 30 percentage
points of it never does. The codebook reaches the same conclusion independently.

**Parental education: use the supplied numbers, change nothing, fix later** (by
instruction). TAS `pared_col = 1` means *either* parent has 16+ years; the model's state is
`BothCollege`. These are different groups, and the reconstructed both-parents rates are
0.7861 / 0.2013 against the targeted 0.6013 / 0.2108 — a gap 50% wider than the one being
targeted. Recorded as [`ERRORS.md`](ERRORS.md) P7c, with the three ways out costed. Until it
is taken up, the estimated `kappa_ParEd` is *not* the effect of parental education.

**Terminal wealth: winsorised at p99.** The raw moment is a mean of \$429,803 on a median of
\$49,243, an SD of \$1.9m and a maximum of \$41.3m — the top 1% alone moves the mean by ~\$98k,
and the model's asset grid stops at \$1m. This repository already winsorises consumption and
investment at p99 for the same reason. The cut is exported in model units and the model side
applies the *same* functional, `E[min(W, cut)]`, so the two are the same estimator. Negative
net worth (13.0% of the sample) is retained, not dropped.

**Wealth is post-transfer.** Pre-transfer assets, mean transfer and retained assets are all
printed (37.60 / 22.03 / 15.57 at the 2026-09-10 incumbent). The target sat *between* the
model's pre- and post-transfer values; substituting pre-transfer assets would have
"improved" the fit while measuring the wrong object. Validation group 9 pins this.

**Achievement: rank-based, within the model** (for the untargeted tertile diagnostics).
The model's HC level is already targeted by `mean_hc_late`; cutting the simulation at the
data's absolute W-score cut points would fold any level or dispersion miss into
`kappa_theta`. Ties break by stable rank (`MergeSort`). Tertiles are cut *within* age group
in the data, so only the within-panel gradient is meaningful.

#### Untargeted exports

Written to every target file for reporting, never entering `Q`: the three ability tertiles
`kth_ga17_t{1,2,3}_c` (0.113 / 0.245 / 0.629), the two halves of the gap
`kth_ga17_mean_{c,n}` (6.2843 / 6.2532), the wealth tertiles `kse_w_t{1,2,3}_c`
(0.207 / 0.358 / 0.617), completion on the wealth frame `k0_w_c` (0.394 — selection into
`ever_strict`; the model counterpart of `kse_w_gap` is the full population), the
unknown-education group `kpe_gu_c`, and `sd_ga_age{3..17}` — the SD of `x_gach` by child
age from the CDS file (0.0827 at 3 falling to a flat 0.02–0.03 from age 9), which has no
model counterpart and is the evidence for the shock's form: initial heterogeneity decaying
under `σ₃ = 0.41` onto a stationary floor, which is what an i.i.d. `η` produces. The
entry moments, LW subscale and age-18 variants are no longer produced.

#### Staging

The target file is generated once; only the `targeted` flags change per stage.

| stage | model | parameters | targeted TAS moments |
|---|---|---|---|
| 0 (2026-09-10) | as is | 14 | `k0_complete`, `kth_ga17_t{1,2,3}_c`, `kpe_g{0,1}_c`, `kterm_x_strict_w99` |
| 1 | `sigma_eps` freed | 15 | stage 0 + `kse_w_gap` |
| **2 (current)** | + `sigma_eta` | **16** | `k0_complete`, `kth_ga17_gap`, `kpe_g{0,1}_c`, `kterm_x_strict_w99`, `kse_w_gap`, `sd_ga17` |

Stages 1 and 2 were run together on 2026-09-11 as a preliminary estimation. If more slack
is wanted, re-add the ability tertiles beside the gap (+3, not linear in it) or the wealth
tertiles (+2 independent) — both already in the export. `spec_version`
(`smm16_tas7_gap_v1`) must be bumped whenever the target list changes; `selftest.jl` fails
otherwise.

### The objective

```
Q(θ) = Σ_j w_j (m_j − m̂_j)²,        w_j = 1 / se_j²
```

— each residual in standard errors of its own moment ([`moments.jl`](../code/smm/moments.jl),
`moment_weights`; decision 2026-09-10). The residual vector every tool must agree on is
`sqrt(w_j)·(m_j − m̂_j)`, in `SMM_MOMENTS` order.

**The standard errors come from a joint cluster-robust covariance the generator rebuilds
every run.** The missing `SMM_TAS_VCov.dta` was reconstructed, not requested: the codebook
specifies the estimator exactly (a ratio of means, clustered on `famclust`), which is a
closed-form influence function, and it reproduces all seven published estimates *and* their
standard errors to six decimal places. Influence functions from both micro files are
stacked on the shared `Fam_id` key — 488 families appear in both frames (1,794 parent,
1,481 TAS, 2,629 distinct).

**The standard errors are not the cross-sectional SDs.** They are 2–8% of them —
`mean_c_p` has se 0.0355 against sd 1.699. A diagonal weight built from SDs would have been
wrong in magnitude and in shape.

**The diagonal weight is a first-stage choice, and it is not free.** Across all 70
cross-block pairs the correlation runs from −0.035 to +0.045, so the two *blocks* are close
to independent — but that does not justify the diagonal, and an earlier version of this
document said it did. The correlations a diagonal weight throws away are the *within*-block
ones, and they are large: the largest in absolute value is **+0.676**. The case for the
diagonal first stage is the ordinary one — clean and well-conditioned, where a two-step
optimal weight estimated on 2,629 clusters can be noisy enough to move the estimate more
than the efficiency it buys. Whether it costs much here is an open question that needs a
second-stage comparison. The full 17×17 covariance is exported and is what standard errors,
sensitivity and the identification diagnostics use; `standard_errors.jl` inverts the
**correlation** form, per the codebook's warning that the raw form is ill-conditioned
because the vector mixes probabilities with dollars.

**Q is dominated by the TAS block.** At the 2026-09-10 incumbent, six completion moments
carried 98% of `Q`; at the current estimate the top three moments carry 79%. `report_fit`
prints the concentration on every run. If the parent moments drift during a search, the
answer is a deliberate scale decision, not a wider box.

**How the objective got here — the lesson that still applies.** Until 2026-09-10 the
objective was a weighted *relative* distance, `Σ w_j((m_j − m̂_j)/s_j)²` with equal `w_j`,
`s_j = max(|m̂_j|, 0.05)` for a level moment and `s_j = 1` for a **log** moment
(`moment_scale`, now used only by the frozen parent-block regression in
`tools/test_smm_baseline.jl`). The log exception was the important part. `x_gach` is a log
W-score, so the HC targets are ~6.1; dividing their residual by the target shrank it
**6.1×** before squaring. Measured at the 2026-09-06 incumbent, the model's human capital was
**+60% in levels** and the objective scored it as a **7.7% miss**, while `R_0` — the
parameter in the set specifically to fix the HC level — got only 13.9% of its identifying
leverage from the HC moments; on the units-free scale that became 86.1%, the Jacobian's
condition number fell 162 → 49, and its smallest singular value was 3.4× stronger. A
moment's weight must not depend on the units its log happens to be in. That is why
`report_fit` still prints the HC gap in **levels** (`exp(Δlog) − 1`), and why the
inverse-variance weight — which is units-free by construction — is the principled successor.

**Common random numbers** throughout: every model is built with the same `seed`, so draws
and shock paths are identical across evaluations. The HC shock `z` is drawn from a
**dedicated seeded stream** so the wage and taste draws do not move when `sigma_eta`
changes. Without this the objective is a step function of simulation noise and no
derivative-free method converges — it would be chasing the RNG, not the parameters.

### What the objective refuses to score

Three gates, in order, before a number is returned. Each returns the large **finite**
penalty `SMM_PENALTY = 1e6` — never `Inf` and never an exception, so a derivative-free local
search can still form a descent direction away from a bad region — and each increments a
per-worker reason counter that `run_smm.jl` gathers and prints at the end. A high penalty
rate means the **box** is wrong, not that the model is (`n_penalized = 953` of 3,060 on the
current run — most of them Sobol points).

1. **Economically infeasible draws, before paying for a solve.** `smm_feasible` checks that
   the Cobb-Douglas money share `σ_2,t = exp(σ_20 + σ_21(t−1))` stays below 1 at both
   endpoints of `t = 1..17`. Above 1 the technology is explosive and SLSQP wanders to a NaN
   iterate rather than failing cleanly; a Sobol point landed in that corner and killed the
   2026-08-27 run at evaluation 376 of 401.
2. **Simulations that leave the model's own domain.** `simulation_violations` counts
   violations **by kind** — consumption and skill strictly positive, investment
   non-negative, each time share in the unit interval, both leisure residuals
   (`1−h−t` and `1−t−i−school`) non-negative, assets at or above `a_min` over **all T+1
   columns**, and non-finite cells anywhere. This replaced a check that counted only
   non-finite entries, which is not the same thing: measured by injection, negative
   consumption, negative hours, hours above the time budget and assets below `a_min` were
   **all accepted** as long as they were finite. A partly-failed solve could compete on the
   strength of the cells that happened to survive.
3. **Exceptions, classified by root cause.** `ErrorException` (the solver's 95%-convergence
   throw), `DomainError`, `AssertionError` and `InexactError` are scored as penalties;
   anything else is **re-thrown**, because a `MethodError` is a coding error and must not be
   laundered into a converged run. The classification unwraps `CapturedException` first
   (`_root_cause`) — NLopt wraps anything thrown inside a callback, so testing the type
   directly is always false, and that cost two runs.

### How an evaluation works

The specified order, implemented in `run_pipeline` (`moments.jl`):

```
solve child lifecycle and transfer problems      (rebuilt per evaluation; see below)
  -> initial child simulation                    (demonstration; outputs ERASED)
  -> construct the terminal-value object          (from the solved value functions)
  -> solve and simulate parents
  -> initialize children from simulated parent outcomes
  -> resimulate children
  -> calculate all seventeen moments
```

The demonstration simulation cannot leak: its `sim_college` and `sim_tr_init` are filled
with NaN before the parent block runs, and `model_moments` errors if the resimulation has
not refilled them. Validation group 6 runs the pipeline with and without it and asserts
every targeted moment is bit-identical.

Handoff, asserted exactly (validation group 5):

```julia
child.sim_a_init  .= parent.sim_a[:,  parent.T + 1]
child.sim_k_init  .= parent.sim_hc[:, parent.T + 1]
child.sim_bc_init .= parent.sim_k[:, 1]
```

Matching `simN` and a common `seed` are both asserted.

**The child solve is rebuilt, not reused — and it is nearly free.** Until 2026-09-10 every
estimated parameter was a parent-block parameter, so the child lifecycle, its transfer
stage and the terminal value spline depended on none of them. They were solved **once per
process** and reused, which was exact, and `moments.jl` enforced it by erroring at load
time if any `SMM_PARAMS` name was not a field of `PARENT_DEFAULTS`. That invariant is now
false by design — all five child parameters change the child solve. **The guard was
replaced, not deleted**, because deleting it is precisely what would produce a converged
fit for a model that was never solved. The child solve has four stages and the estimated
parameters do not touch all of them:

| stage | cost (grid 30, warm) | reads an estimated parameter? |
|---|---|---|
| `solve_model_work!` (high-school path) | 6.31 s | **no** |
| `solve_model_college!` stage 1 (graduate life, `E = 1`) | ~5.8 s | **no** |
| `solve_model_college!` stage 2 (the `t_college` study years) | ~0.5 s | `kappa_0`, `kappa_theta`, `m_psychic` |
| `optimal_transfer_work!` + `optimal_transfer_college!` | 0.50 s | `kappa_terminal`, `sigma_eps` |
| `terminal_value_spline` | ~0.00 s | `kappa_ParEd` |

So 12.1 s of the 12.9 s is invariant across the whole search and only ~1.2 s has to be
redone per evaluation — about **+7%** on an evaluation, not +100%. `build_child_solution`
caches the two invariant stages under a **complete dependency key** (`child_config`: every
non-estimated child setting and no estimated one) and rebuilds the rest. A second cache
tier holds the **full** child solution keyed on the configuration plus the five child
parameters, so a finite difference in a *parent* parameter — which changes no kappa — does
not redo the study years and transfer stage either (bounded at two entries, ~90 MB each).
Cache keys do **not** include `simN` or `seed`: neither can change a solution, and
including them produced cache misses, not stale hits.

**This is exact, and it was verified rather than argued.** Refreshing from the cached work
and graduate blocks reproduces a full re-solve **bit-identically** — `max |diff| = 0.000e+00`
on `sol_v_college`, `sol_c_college` and `sol_h_college`, with an identical NaN feasibility
pattern — at 16.9× the speed. There is no tolerance to tune: the arrays are copied, not
refitted. `tools/test_smm_tas.jl` groups 4 and 10 are the standing regression.

The cache stores **solution arrays, never a model object**, deliberately: a cached model
would carry mutable `sim_*` state and any simulator touching it would contaminate every
later evaluation with another draw's simulation. There is no cached object to simulate.

Each evaluation is therefore: rebuild ~1.2 s of child solve, build the parent,
backward-induct (integrating the HC shock by Gauss–Hermite, `Neta ≤ 5` nodes), simulate,
hand off, resimulate the child, compute seventeen moments. The parent solve is still most
of it.

**One evaluation pipeline, five tools.** `jacobian.jl`, `sensitivity.jl`,
`profile_param.jl`, `grid_sensitivity.jl` and `standard_errors.jl` each used to hold a
private copy of `build_child_value()` and its own residual. Four copies of an objective is
four chances for one to drift from the one being optimised, and a Jacobian of the wrong
objective is a diagnostic of nothing. They now share `evaluate_at` / `residuals_from`.

### Running the search: budget, grids, and surviving a kill

| | default | flag |
|---|---|---|
| Sobol points | 1000 (+1: the incumbent) | `--sobol` |
| restarts | 100 | `--restarts` |
| evals per local search / polish | 2000 / 4000 | `--local-evals`, `--polish-evals`, `--skip-polish` |
| parent grid, search | 30 | `--grid` |
| parent grid, report | 30 (fixed) | — |
| full-grid refinement evals | 200 | `--refine` |
| `simN` | 2000 | — |
| worker processes | 20 | `--procs` (**not** `--workers`) |
| warm start | — | `--init-from <estimates.toml>` |
| targets | — | `--targets <targets.toml>` (old target files are rejected) |

Every flag is validated; an unknown one is an error rather than silently ignored. The
2026-09-11 pilot budget was `--sobol 1000 --restarts 5 --local-evals 500 --skip-polish`;
the current run used three restarts with polish (444 min on 20 workers).

**Search cheap, quote exact.** 98% of an evaluation is `solve_model!` and its cost scales
with `Na × Nhc`. Dropping 30 → 20 makes an evaluation 2.5× cheaper and moves the targeted
moments by 0.01–0.2%, against the gaps the estimation exists to close. So `--grid` sets the
grid the **search** runs on, the fit is **always** re-solved and reported at 30, and the
coarse winner is then re-optimised at the full grid by a short BOBYQA polish.
`estimates.toml` records **both** objectives under separate names, `Q_search` and
`Q_final`, on the grids they were computed at. Never quote a `Q` minimised at `Na = 20`.
`simN` is *not* the place to economise: it is 2% of the cost, and cutting it to 500 moved
`c_p` more than halving the grid did.

**Processes, never threads.** NLopt.jl is not thread-safe in this project — with
`parallel = true` and 8 threads the objective killed the process with exit 0 and no error.
Each worker *process* owns its NLopt state. Two guards keep this from regressing: each local
search builds an `Opt` that belongs to it (a closure shared one through a `Core.Box`,
which is a data race in NLopt's C state and a sufficient explanation for the silent exit),
and `batch > 1` is **rejected** unless `parallel = true`.

**Only half the run is parallel.** The Sobol stage divides by the worker count; the restarts
are sequential by construction, since restart `j` starts from the best point found by
`1..j−1`. The run prints the two halves of the projected runtime separately before
committing. To spend a bigger machine on this problem, raise `--sobol`, not `--restarts`.
The 20-worker cap is a house rule for a shared server, not a hardware limit.

**A killed run resumes exactly.** The local stage is ~99% of the wall clock and runs for the
better part of a day, so a disconnect or a pre-emption will eventually catch one.
`checkpoint.toml` is written after **every** restart (atomically, temp file plus rename) and
carries the stage, the objective **and the grid it was computed at**; `seeds.toml` is written
once, right after pre-testing, with the surviving seeds. `--resume DIR` reloads both and
re-enters the local stage at the next restart with exactly the mixture the original would
have used — continuation, not a warm start. Verified on Rastrigin: a run resumed at restart
11 reached the identical objective (0.9949590571) in 1096 evaluations against the full run's
2447.

`load_resume` refuses rather than guesses. A checkpoint must match on `spec_version`,
`source_sha` (`child_lifecycle.jl` + `parent_family.jl` + `moments.jl` — the model source is
part of the objective now that child parameters are estimated), `moment_names`,
`m_psychic`, the parameter count, restart budget, search grid, child grid, `simN` and
`seed`. **A missing field is itself grounds to refuse** — "cannot verify" is not
"compatible". This matters because an accepted checkpoint hands its `Q_best` to the new
run, and if the new run cannot reproduce that objective every later restart is measured
against an unbeatable number. `tools/test_smm_resume.jl` covers 8 missing-field cases, 8
changed-value cases and a control. Do not resume a checkpoint from an earlier
specification; start fresh.

### The next run

What `exp16b` says about how to spend the next budget, and what was changed for it
(2026-09-12):

- **Start from the baseline, not from a file.** The block defaults *are* `exp16b`, so
  the seeded incumbent is the fit; `--init-from` is only for starting somewhere else.
- **The Sobol stage bought nothing.** 1,000 points in sixteen dimensions and the best of
  them did not come close to the seeded incumbent; the box was spending its budget in
  `kappa_0`'s dead region (71% of `[−2, 5]` is a flat zero) and above `kappa_ParEd`'s
  saturation. Hence `kappa_0 → [−3, 1]` and `kappa_ParEd → [−1, 0.5]`.
- **`kappa_ParEd` sat 3.6% from its wall at 0** and `sigma_4_1` at 90% of its ceiling
  for the second fit running; both boxes now leave room. If `kappa_ParEd` settles at ~0
  again, fix it at 0 and drop a parameter rather than widen further.
- **All three restarts converged and the polish met FTOL**, so the local budget
  (2,000 / 4,000) is right; more *restarts* is what buys a better basin, not more
  evaluations per restart.
- **What no box change fixes.** `kse_w_gap` (−94%) and `kth_ga17_gap` (−47%) carry 69%
  of `Q`: the model does not sort into college on wealth or on ability nearly as strongly
  as the data, and `sigma_eps` doubled (0.5 → 1.14) to blur the margin rather than
  sharpen it. That is a specification question — the analytic taste-shock integration,
  probability-weighted moments and BOTH-vs-NOT-BOTH targets in `Structural-estimation-v2`
  are the response, and it is where the estimate should be read next. `mean_a_p`-type
  wealth levels (retained assets −24%) are the `beta_0` question of caveat 5.

```bash
julia --project=. code/smm/run_smm.jl --sobol 1000 --restarts 5 --grid 30 --procs 20 \
      --seed 1234 --targets output/smm_runs/2026-09-11_182836_exp16b/targets.toml \
      --outdir output/smm_runs/<stamp>_exp16c      # ~5 × 2,000 + 4,000 evals ≈ 1.5 days
```

Run `code/smm/selftest.jl` first: it pins the new boxes and asserts every start is
inside its box.

### Identification

**Parameter and moment counts establish nothing.** What does is the residual Jacobian —
central differences of the weighted residual vector, columns scaled to a full-box move.
[`tools/check_jacobian_rank.jl`](../tools/check_jacobian_rank.jl) takes it at a point and
three step sizes and sweeps simulation sizes and seeds (default 400/1000 × two seeds),
because full numerical rank on one draw of 400 households is not evidence of robust
identification.

**The 16-parameter Jacobian has not yet been taken at the current estimate.** That is the
first thing to do with `2026-09-11_182836_exp16b`, and the [verification checklist](#appendix--the-2026-09-11-respecification-work-order)
requires full rank at the stage's parameter count before the run's numbers are read as
estimates.

**At the 14-parameter stage (2026-09-10)**, measured at the report-only incumbent and at a
point with an interior college share:

| point | step | rank | cond | smallest sv |
|---|---|---|---|---|
| incumbent (share 0) | 2% | **11/14** | Inf | 0 |
| incumbent | 5% | **12/14** | 1.2e20 | 5.9e-17 |
| incumbent | 10% | 14/14 | 8.2e4 | 0.085 |
| interior college | 2% | 14/14 | 1.4e4 | 0.618 |
| interior college | 5% | **14/14** | **1.1e3** | **6.87** |
| interior college | 10% | **14/14** | **922** | **7.78** |

**The model was locally unidentified at that incumbent.** With the college share pinned at
zero, none of the completion moments responds to a small perturbation and the Jacobian
loses three columns — which is why `kappa_0`'s box is ~71% dead: completion is 0.000 at
`kappa_0 = 0.059` and 1.000 by −1.0, and everything above ~0 is a flat zero where the
derivative is identically zero. At an interior-college point it is full rank at every step
size, and rank, conditioning and the least-identified direction are stable between the 5%
and 10% steps — the 2% step is finite-difference noise on a simulated objective. The
weakest direction is consistently **`kappa_theta` against `kappa_ParEd`** — the two
psychic-cost gradients trading off; `kappa_ParEd` is the weakest column throughout. That
trade-off is exactly what P13 turned out to be, and what `sd_ga17` and the absolute gap are
there to break.

**At the nine-parameter stage (2026-09-06)**, the parent-block Jacobian had full column
rank, condition number 49 and smallest singular value 0.278. The weakest direction was
`lambda_2` against `sigma_1_0 + sigma_4_0 + sigma_2_0` — valuation against technology —
and the second weakest `sigma_2_1`. Both identified; neither sharply. The worst-separated
*pair* was `sigma_1_0` vs `sigma_1_1` at a cosine of **0.908**: `t_p` is split at the same
two age groups as everything else, so "ages 6–9 and 10–17 are too close to separate a level
from a slope" applies to a parameter that is already free. Read it as an argument for
richer age moments, not for dropping `sigma_1_1`. The saved matrices are in
`output/identification/jac_9col` and siblings, each with the evaluation point, boxes, links,
scales, grids, seed and steps beside every number.

**Why `sigma_4_1` was out, and why it is now in.** On 2026-09-06 `sigma_4_1` and `mu_1`
were both held fixed because adding either worsened conditioning — 51.2 → 228.9 for
`sigma_4_1`, → 198.4 for `mu_1`, with pairwise cosines `sigma_4_0`/`mu_1` 0.991 and
`sigma_4_0`/`sigma_4_1` 0.814. That was **conditioning, not rank**: every set was full rank
except the eleven-column one, which has one unidentified direction by construction. (A
previously circulated 21.7× ratio was not reproducible — condition numbers are not scale
invariant and the box that produced it was never recorded; the direction survived, the
magnitude did not.) On 2026-09-09 the HC late window moved to 12–17 and `sigma_4_1` was
freed against the study and HC age profiles; **the Jacobian has not been re-taken for the
parent block since**, and the 2026-09-06 numbers describe the nine-parameter specification
only. `mu_1` stays fixed: taking logs of the child's study FOC ratio,

```
log[ σ_4,t / (1 − μ_t) ] = σ_40 + σ_41·s − log(−μ_1) − log s ,      s = t − 5 > 0
```

`mu_1` enters only through the intercept, which is what makes it near-collinear with
`sigma_4_0`. **This is a warning, not a proof of an exact ridge**: `mu_1` also moves the
child's leisure weight and `α̃_2,t` in `util_total`, so 0.991 means nearly parallel *local*
moment responses, not observational equivalence.

### The diagnostics that travel with an estimate

None of these existed before 6 September 2026; the numbers they now produce used to be
recollections from a review conversation.

| script | what it answers | what it does **not** |
|---|---|---|
| [`jacobian.jl`](../code/smm/jacobian.jl) | local separation: singular values, condition number, weak directions, every pairwise cosine — saved with the point, boxes, scales, grids, seed and steps | global identification; a small condition number is not precision |
| [`standard_errors.jl`](../code/smm/standard_errors.jl) | sampling uncertainty: the clustered minimum-distance sandwich under the diagonal and the efficient weight. **Rank-aware**: a tolerance-explicit pseudo-inverse of `G'WG` that reports its rank and warns that intervals are conditional on the unidentified directions | simulation error, the weighting choice, specification error |
| [`sensitivity.jl`](../code/smm/sensitivity.jl) | how the argmin moves when one target moves — all parameters jointly re-estimated at each perturbed target | identification or robustness evidence. A point that reports `MAXEVAL_REACHED` or `on_bound` is censored, not a slope |
| [`profile_param.jl`](../code/smm/profile_param.jl) | whether a box wall is binding or `Q` is flat: fix one parameter on a ladder, jointly re-optimise the rest | anything about the joint optimum's location |
| [`grid_sensitivity.jl`](../code/smm/grid_sensitivity.jl) | whether the numerical grid (`a_max`, nodes) moves the targeted moments, in residual units | — |
| [`tools/check_jacobian_rank.jl`](../tools/check_jacobian_rank.jl) | numerical rank across step sizes, `simN` and seeds | — |

Each refuses to run on missing inputs rather than substituting a plausible one:
`standard_errors.jl` will not compute its own Jacobian, and `sensitivity.jl` will not
substitute a per-observation SD for a moment standard error.

### Validation

```bash
uv run --with pandas --with numpy python tools/make_smm_targets.py
julia --project=. tools/test_smm_tas.jl <targets.toml>          # 120 checks, ~90 s
julia --project=. tools/test_smm_resume.jl <targets.toml>        # 17 refusal cases + control
julia --project=. code/smm/selftest.jl                           # the specification is frozen
julia --project=. tools/check_jacobian_rank.jl <targets.toml>
julia --project=. code/smm/run_smm.jl --report-only --targets <targets.toml>
julia --project=. code/smm/run_smm.jl --quick --sobol 8 --restarts 1 --procs 6 --targets <targets.toml>
```

`tools/test_smm_tas.jl` groups, and what each rules out:

| group | what it rules out |
|---|---|
| 1 target reproduction | the target file disagreeing with the published moments; a covariance whose diagonal is not its own SE vector |
| 2 parameter routing | a child parameter being absorbed by a same-named parent field while the block keeps its default |
| 3 child-solution refresh | a kappa that changes nothing — a converged fit for a parameter that does not act |
| 4 cache-key completeness | the cache serving a stale college solution; a simulated array in the cache |
| 5 handoff arrays | a mis-sized or mis-ordered handoff |
| 6 demonstration isolation | the initial child simulation supplying a final moment |
| 7 determinism | a broken common-random-numbers path; removed entry points silently working |
| 8 centring neutrality | the reparameterisation changing behaviour |
| 9 wealth accounting | pre-transfer assets being substituted for post-transfer |
| 10 cache parity | the cached refresh differing from an INDEPENDENT full solve |
| 11 mutation isolation | simulating a returned child poisoning the cache for the next draw |
| 12 partial points | an omitted child parameter falling back to the constructor's default instead of the SMM's |
| 13 `parent_extra` | a non-estimated parent setting failing to reach the constructor |

Also standing: `test_smm_target_paths`, `test_smm_boundaries`, `test_smm_own_study` (the
own-study routing and the parent/child split), `test_hc_process_shock.jl` (the shock is
zero-mean in logs, `sigma_eta = 0` is bit-identical to the deterministic solver, and the
dedicated stream leaves the other draws alone). `test_smm_baseline` is **deliberately scoped
to the parent block on the old proportional scale** — its frozen `Q` is the only frozen
reference to the parent solve that exists, and re-pinning it to the new objective would
discard it.

#### Fixes from external review (2026-09-10)

An independent review found six defects. All are fixed and have regression coverage; the
two graded P1 would have produced confidently wrong diagnostics.

| # | Defect | Effect | Now covered by |
|---|---|---|---|
| P1 | `grid_sensitivity.jl` accepted `a_max` but never forwarded it — `run_pipeline` had no channel for non-estimated parent settings | every rung of the asset-ceiling sweep solved at the SAME ceiling, so the tool concluded "the asset grid does not move the moments" vacuously | test group 13; `solve_at` reads `p.a_max` back and errors on a mismatch |
| — | the same file aborted on `Main.GS_AMAX = $am` — an interpolation outside a quote, in the `--reoptimize` branch; Julia lowers a top-level `if` whether or not the branch is taken, so it fired on every run | the tool had never completed | dead code removed; the ceiling travels as an argument |
| P1 | its summary computed `Δ / 1.0 / sqrt(w)`, and `sqrt(w) = 1/se`, so it MULTIPLIED by the standard error | a 0.001 move on a 0.01 SE reported 0.00001 residual units instead of 0.1 — always under the "grid does not matter" threshold; the two bugs pointed the same way, which is why neither showed | fixed to `Δ · sqrt(w)` |
| P1 | resume checks read `haskey(ck, f) && <mismatch> && refuse(...)`, which ACCEPTS a checkpoint that lacks the field; `child_grid`, `sim_n` and `seed` were never compared | a checkpoint from a different numerical problem could hand its `Q_best` to a run that could not reproduce it | `tools/test_smm_resume.jl` |
| P2 | `build_child_solution` used `merge(cfg, ckw)` without completing omitted kappas | a partial point — the documented interface every diagnostic uses — fell back to CONSTRUCTOR defaults (`kappa_0 = 0.2728`, the legacy UNCENTRED value; `kappa_terminal = 10.0`) | test group 12 |
| P2 | the printed cross-block correlation divided by `sqrt(se_i·se_j)` | reported −0.0011/+0.0044 against a true −0.0350/+0.0453; the exported `cov` and `corr` were always correct | the generator prints within-block correlations too |
| P2 | `selftest.jl` still asserted ten parameters and ten moments | the self-test's closing verdict was permanently "do not run the estimation" | updated, plus routing and moment-order assertions |

**What the fixed grid sweep says.** `a_max` 80 against 200 at grid 14 / simN 400 moves the
targeted moments by **2.04 residual units** (Q 2414.6 → 2437.1, terminal assets 37.92 →
35.98). The parent's asset ceiling is therefore a live specification choice, not a settled
one; `Q` is not comparable across ceilings. It has **not** been swept at the production
grid — see [caveat 8](#part-4--caveats-that-must-travel-with-any-number-from-here).

**Not adopted:** a handoff-only simulation path for estimation — skipping the child's
51-year lifecycle, which the seventeen moments do not need. A real saving, but the
specification explicitly requires the complete simulation sequence inside each evaluation.
That is a specification change and belongs with the advisor.

---

## Part 3 — Results

### Current: `2026-09-11_182836_exp16b` — 16 / 17, preliminary

Grid 30, `simN = 2000`, seed 1234, 1,000 Sobol + the seeded incumbent
(`--init-from` the `exp16` pilot's warm start), 3 restarts, polish. `Q_incumbent = 313.34`
→ `Q_final = 61.80`, winner at the polish, `FTOL_REACHED`, `accepted = true`, 0 invalid
cells, no parameter on a bound (`kappa_ParEd` within 5% of its upper wall at 0).

| moment | model | data | gap | t | Q share |
|---|---|---|---|---|---|
| `mean_c_p` | 3.0473 | 3.1155 | −2.2% | −1.9 | 6.0% |
| `mean_h_p` | 0.3092 | 0.3073 | +0.6% | 0.9 | 1.2% |
| `mean_t_p_early` | 0.4456 | 0.4544 | −1.9% | −1.2 | 2.4% |
| `mean_t_p_late` | 0.3427 | 0.3333 | +2.8% | 1.4 | 3.1% |
| `mean_e_p_early` | 0.3456 | 0.3429 | +0.8% | 0.2 | 0.1% |
| `mean_e_p_late` | 0.3861 | 0.3911 | −1.3% | −0.3 | 0.2% |
| `mean_i_c_early` | 0.0375 | 0.0393 | −4.6% | −0.7 | 0.7% |
| `mean_i_c_late` | 0.0506 | 0.0496 | +1.9% | 0.4 | 0.2% |
| `mean_hc_early` | 6.0743 | 6.0737 | +0.1% | 0.2 | 0.0% |
| `mean_hc_late` | 6.2588 | 6.2589 | −0.0% | −0.1 | 0.0% |
| `k0_complete` | 0.3275 | 0.3226 | +1.5% | 0.4 | 0.3% |
| **`kth_ga17_gap`** | 0.0167 | 0.0312 | **−46.5%** | −4.6 | **34.5%** |
| `kpe_g0_c` | 0.2128 | 0.2108 | +1.0% | 0.2 | 0.0% |
| `kpe_g1_c` | 0.5895 | 0.6013 | −2.0% | −0.6 | 0.7% |
| `kterm_x_strict_w99` | 25.157 | 33.198 | −24.2% | −2.5 | 9.9% |
| **`kse_w_gap`** | 2.139 | 36.843 | **−94.2%** | −4.6 | **34.7%** |
| `sd_ga17` | 0.0365 | 0.0329 | +10.8% | 1.9 | 5.9% |

| parameter | estimate | started from | | parameter | estimate | started from |
|---|---|---|---|---|---|---|
| `phi_2` | 0.1963 | 0.1897 | | `sigma_4_1` | 0.1301 | 0.1327 |
| `phi_3` | 1.5337 | 1.2478 | | `sigma_eta` | 0.0315 | 0.0 |
| `lambda_2` | 13.862 | 10.672 | | `kappa_0` | −0.3566 | 0.0587 |
| `R_0` | 48.336 | 50.452 | | `kappa_theta` | −3.6194 | −0.0342 |
| `sigma_1_0` | −0.8886 | −0.7374 | | `kappa_ParEd` | −0.1081 | −0.0070 |
| `sigma_1_1` | −0.0927 | −0.0851 | | `kappa_terminal` | 8.7868 | 5.0 |
| `sigma_2_0` | −3.6918 | −3.6233 | | `sigma_eps` | 1.1422 | 0.5 |
| `sigma_2_1` | −0.0980 | −0.0830 | | | | |

**What the shock bought.** With `sigma_eta = 0.0315`, completion by parental education now
fits — `kpe_g0_c` went from 0.000 to 0.213 against 0.211 — and the dispersion of `log HC`
at 17 is right (0.0365 vs 0.0329). That was the P13 failure, and it is closed.

**What it did not.** Two moments carry 69% of `Q`, and they are the two *gaps*: the
completion–ability gap is half its target and the completion–wealth gap is essentially
zero (2.1 against 36.8). Selection into college in the model runs almost entirely through
the taste shock (`sigma_eps` doubled to 1.14) rather than through ability or through
parental wealth, and retained wealth is a quarter low. `kappa_theta` moved two orders of
magnitude from its calibrated start (−0.03 → −3.6) and should be read with the
[16-parameter Jacobian](#identification), which has not been taken.

### The 14-parameter stage: `2026-09-10_183649` and the P13 diagnosis

The first estimation with the TAS block, from the report-only incumbent `Q = 2405` to
`Q = 427`, accepted. The parent block fitted to |t| ≤ 1.3 everywhere; 92% of the remaining
`Q` was three moments that were one problem:

| moment | model | data | t | Q share |
|---|---|---|---|---|
| `kpe_g0_c` | **0.000** | 0.211 | −16.5 | 64% |
| `k0_complete` | 0.215 | 0.323 | −9.4 | 21% |
| `kpe_g1_c` | 0.706 | 0.601 | +5.7 | 8% |

All 430 simulated completers were children of `BothCollege = 1` parents. Moving
`kappa_ParEd`, `kappa_0`, `Nt` or `sigma_eps` did not free `g0`; the SD of `log HC` at 17
was 0.0081 against 0.0329. Structural, not a bad basin — [`ERRORS.md`](ERRORS.md) P13.

At the report-only incumbent before that run (`2026-09-10_151548`), the college share was
**exactly zero**: every simulated parent cleared the college asset threshold
(`col_min = 2.28` model units against a minimum terminal asset of 10.03), so it was the
value comparison, not feasibility — college was simply never preferred at the calibrated
kappas, which is exactly what estimating them was for. The six completion moments carried
98% of `Q = 2405`.

### Historical calibrations

**The nine-parameter fits (2026-09-06 to 09-09)** on the old relative-distance objective
and the school-plus-study targets: run `2026-09-07_114138` reached `Q = 0.0116` with
`sigma_2_1` pinned at −0.10, and run `2026-09-09_003312` is the fit `PARENT_DEFAULTS` still
carries — **a starting vector, not an estimate under the current specification**, since the
targets, the weights and the child block have all changed since. Inspection notes are under
`output/smm_diagnostics/`. Before any of it was estimated, the calibration measured at the
2026-09-06 incumbent had `Q = 2.94`, human capital **+80% / +43%** in levels and early
study **+195%** — the expected shape, since `R_0`, `phi_3`, `lambda_2` and `sigma_4_0` had
never been estimated against anything.

**The 2026-08-28 six-parameter run** (`output/smm_runs/2026-08-28_130810`) supplied the
first incumbent (`phi_2 = 0.142`, `sigma_1_0 = −0.457`, `sigma_1_1 = −0.063`,
`sigma_2_0 = −3.40`, `sigma_2_1 = −0.029`). It matched **six** moments with **six**
parameters against *observation-weighted* targets, with HC in model units and no HC or
study-time moment; it also estimated `phi_1_0 = 0.754`, which the `phi_1 = 1`
normalisation has since discarded. Its `Q_final = 9.7e-12` is a just-identified exact fit
on a different objective and **is not comparable** to any `Q` reported now.

---

## Part 4 — Caveats that must travel with any number from here

Read these before quoting anything above. Ordered by how much they could move a result.

**1. The 2026-09-11 specification has not been through the advisor.** `sigma_eta` is a
change to the HC technology, not a numerical fix. Until it is approved the 16-parameter
estimates are a preliminary experiment, and the results do not circulate.

**2. `kappa_ParEd` targets the wrong group.** EITHER-parent college in the data against
`BothCollege` in the model. Open by instruction; [`ERRORS.md`](ERRORS.md) P7c. The
estimated `kappa_ParEd` is not the effect of parental education.

**3. `par_time_tot` overlaps leisure, so `phi_2` absorbs the inconsistency.** The `t_p`
target uses `par_time_tot` (active **plus** nearby/supervisory presence) by instruction.
That measure does not fit an exhaustive time budget — per parent,
`leisure + work + Mom_Total_Act = 112.00` exactly, but `leisure + work + par_time_tot = 133.25`,
21 hours over. Since the model enforces `l_p + h_p + t_p = 1` identically, targeting
`h_p` and `t_p` **forces** model leisure to ~33 hrs/wk against the **59.2 hrs/wk this same
dataset measures**. That ~26-hour gap lands in `phi_2`, which is why it fell 0.526 → 0.142
when the targets were first introduced. **Do not read the estimated `phi_2` as a
taste-for-leisure parameter.** To restore the budget-consistent measure, target
`(Mom_Total_Act + Dad_Total_Act)/2 / 112` instead, and the identity closes exactly;
`tools/make_smm_targets.py` carries the accounting and the one-line revert. Related and
separate: `t_p` is **parental presence**, not exclusive parental time.

**4. No standard errors and no sensitivity analysis have been reported for the current
estimate.** The tools exist ([above](#the-diagnostics-that-travel-with-an-estimate)) and
the covariance is estimated, but the 16-parameter Jacobian, the sandwich standard errors
and the target-perturbation exercise have not been run on `2026-09-11_182836_exp16b`. A
full-rank local Jacobian is not inference, and a diagonal first-stage weight is a choice
whose cost has not been measured. Required before conclusions travel, not optional.

**5. The consumption *profile* is not targeted — only its mean, and nothing in the
parameter set can tilt it.** The slope is the Euler equation,
`c_{t+1}/c_t = (β(1+r))^(1/ρ)`, which contains **no estimated parameter**. `phi_1` is
normalised to 1 and preferences are time-invariant, so the only lever is `beta_0`,
calibrated at **0.98** by instruction (was 0.97):

| β | β(1+r) | growth/yr | over 16 yrs |
|---|---|---|---|
| 0.97 | 0.9991 | −0.06% | −1% |
| **0.98** | 1.0094 | +0.63% | **+10.5%** |
| 0.99 | 1.0197 | +1.31% | +23.1% |
| — | — | — | *data: +21.8%* |

So 0.98 recovers about half the observed tilt; ~0.989 would match it. To target the profile
properly, split `c_p` early/late and estimate `beta_0` against it. **That changes the
baseline for the notebook and the counterfactuals too**, so it goes through the advisor.

**6. Group means are not age profiles.** `σ_j,t = exp(σ_j0 + σ_j1(t−1))` is monotone by
construction, while the underlying investment profile is **U-shaped** — 0.353 at age 1, a
trough of 0.241 at 12, then nearly triples to 0.650 by 17. Two group means are the most this
functional form can honestly be asked to match. **A good fit on the two group means is not
the model reproducing the age profile.** The late `e_p` group also carries an
end-of-horizon spike: with the age-18 handoff approaching, investment pays off immediately
and the model front-loads it. That is the terminal condition, not the elasticity slope, and
`sigma_2_1` will partly absorb it.

**7. Terminal and retained wealth.** Untargeted terminal assets have drifted up with each
re-specification (\$288k at the four-moment stage, \$404k at six, \$406k at nine) on an
implied saving rate of ~31%: the budget binds every period, so `c_p`, `e_p` and income
leave a residual that is mechanically the saving rate. Retained wealth is now targeted
(`kterm_x_strict_w99`) and is a quarter low, while the data is measured at a median child
age of ~29, a median 4 years after qualifying independence; the model's object is assets at
the transfer, child age 18, and the model has no post-separation parent to age forward. The
model also cannot reproduce negative retained assets (13% of the sample; `delta_P` floors
it at zero).

**8. The parent's asset ceiling is unsettled, and assets run off the grid.** `a_max` 80 vs
200 moves the moments by 2.04 residual units at a coarse grid; it has not been swept at the
production grid, and `Q` is not comparable across ceilings. Policies are interpolated with
`Flat()` extrapolation, so a state above the top asset node silently reuses the policy
there; the initial draw is deliberately **not** clamped, since clamping would distort the
wealth distribution to flatter a grid. Small tail mass is not by itself proof of small
policy error.

**9. Dead regions in the child boxes.** Completion is a near-step in `kappa_0` (0.000 at
0.059, 1.000 by −1.0), so most of `[−2, 5]` is flat zero, and `kpe_g1_c` saturates at 1.0
below `kappa_ParEd ≈ −0.30`. A Sobol point in a flat region contributes nothing; a local
search that wanders into one slides to whatever wall it is given. Narrowing to `[−3, 1]` and
`[−1, 0]` was recommended and has not been done.

**10. Data-side approximations in the TAS block.** "By age 25" is an approximation — a
child last observed at 23 or 24 contributes a zero not actually observed through 25. The
last CDS assessment is not demonstrably pre-entry for every child. The covariance conditions
on the fitted PCA weights and the sample cut points as fixed; propagating that needs a
family-cluster bootstrap, not done. Support flags cover parents *or other relatives*.

---

## Appendix — the 2026-09-11 respecification work order

Kept because code comments (`moments.jl`, `make_smm_targets.py`, `test_hc_process_shock.jl`)
point here. P7c / P7b were **not** part of this — by decision, the `kappa_ParEd` targets
stay as they are.

### Files and conventions

| file | role |
|---|---|
| `Input/SMM_TAS_Micro.dta` | one row per TAS-linked child (4,248). Cluster key `famclust`. Used: `hf_complete`, `y_complete`, `ach_age`, `g_ACH`, `tert_ga`, `ever_strict`, `pwx_strict` |
| `Input/SMM_Moments_Micro.dta` | CDS child-year panel (parent block). Used for `x_gach` by `Child_Age` (untargeted `sd_ga_age*`) |
| `Input/SMM_Assets_ByChildAge.dta` | carries `age_bin` (16 rows: 0, 2, …, 30, the lower edge of a two-year bin) rather than `Child_Age`; the generator's CSV export is bin-aware. Undocumented in the codebook |
| `tools/make_smm_targets.py` | generates `targets.toml`. Every TAS moment is `add(name, mask, value)` = `mean(value·mask)/mean(mask)` on the FULL frame; per-cluster influence from `ratio_influence`; joint covariance from `joint_covariance` |
| `code/smm/moments.jl` | model counterparts (`tas_moments`) |

### The model change

`log HC_{t+1} = log F_t(inputs, HC_t) + sigma_eta · z_{t+1}`, `z ~ N(0,1)` i.i.d. across
households and periods, realised after period-`t` investment, applied to every family-stage
transition including the age-18 handoff (`hc_apply_shock` in `parent_family.jl`). Zero mean
in logs, so the conditional mean of HC in levels rises by `exp(sigma_eta²/2)` (≈ 1.00045 at
0.03). The parent integrates over `z` in its continuation once per period
(`eta_expected_interp`, Gauss–Hermite, `Neta ≤ 5`) and by quadrature on the handoff
(`eval_child_value_eta`); the simulator draws `z` from a dedicated seeded stream.
`PARENT_DEFAULTS.sigma_eta = 0.0` is the deterministic technology, bit-identical to the
pre-shock solver. `sigma_eps` became a struct field of the child model and part of
`CHILD_ESTIMATED`; the cached work and graduate blocks are eps-free.

### Code side, per stage

1. `moments.jl`: `SMM_TAS_MOMENTS` → the stage's list, same order as the generator's
   `TAS_MOMENTS`; `tas_moments` computes the new counterparts; the tertiles stay as
   diagnostics.
2. `spec_version` in `estimates.toml` / `selftest.jl` bumped whenever the target list
   changes.
3. `tools/test_smm_tas.jl`: routing / parity checks extended to the new moments.
4. `tools/check_jacobian_rank.jl` at each stage — local identification is tested, not
   assumed.

### Verification checklist

- [x] Every expected value in the [TAS moment table](#the-seven-tas-moments) and the
  untargeted exports reproduces to four decimals.
- [x] The seven published estimates and SEs (`Input/SMM_TAS_Moments.csv`) reproduce to six
  decimals — the `ratio_influence` regression test.
- [x] Joint covariance is positive semi-definite with the new rows; the gap and SD rows are
  built from the influence differences / delta method, not treated as independent.
- [x] `tools/test_smm_tas.jl` green; `selftest.jl` green after the `spec_version` bump.
- [ ] `tools/check_jacobian_rank.jl` full rank at 16 parameters, at the fitted point.

### The preliminary run

Both stages at once. Eleven parent parameters (the ten existing plus `sigma_eta`, start
0.03, box `[0, 0.08]`, level) and five child (the four existing plus `sigma_eps`, start 0.5,
box `[0.1, 2.0]`, log); the fourteen existing boxes unchanged; `R_1` asserted fixed at 0.
Pilot budget: 1,000 Sobol points plus the seeded incumbent, five restarts at ≤ 500
evaluations each, no polish, grid 30, 20 workers, seed 1234, incumbent loaded from the
14-parameter `estimates.toml` (`2026-09-10_183649`) by name via `--init-from` with the two
new parameters at their starts (`2026-09-11_142509_exp16`); then a second run from its
warm start with three full restarts and polish (`2026-09-11_182836_exp16b`, the current
estimate). The 14-parameter run and its `targets.toml` are untouched; the new targets are
frozen under `output/smm_runs/2026-09-11_134542_120129_targets/`. The notebook
`code/transfer_CRRA_wage.ipynb` was not executed; every new constructor argument defaults to
baseline behaviour so it needs no edit.
