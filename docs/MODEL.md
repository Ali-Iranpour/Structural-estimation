# Model ↔ Code Map, and how to run it

The current model. A family (two parents, one child) is solved over `t = 1..17` (child
ages 1–17); at age 18 the family chooses college vs. work and the parents transfer assets;
the child is then solved to age 68. The specification is in [`model.txt`](model.txt);
this file maps each equation to the code that implements it, records the specification
decisions that are frozen, lists the current defaults, and says how to run the thing.

- **P** = [`code/src/parent_family.jl`](../code/src/parent_family.jl)
- **C** = [`code/src/child_lifecycle.jl`](../code/src/child_lifecycle.jl)

References are by **section**, not line number. Both files are organised
primitives → constraints → objectives → solver → simulation, with banner comments, so a
section name stays valid when code moves.

---

## 1. Childhood and adolescence (`t = 1..17`, child ages 1–17)

| Model | Equation | Code |
|---|---|---|
| Parent utility | `U_p = φ₁c^(1−ρ)/(1−ρ) + φ₂l_p^(1−η)/(1−η) + φ₃ ln HC`, `l_p = 1 − h_p − τ_p` | `util_parent` — P, *Primitives* |
| Family utility | `U_f = φ₁c^(1−ρ)/(1−ρ) + φ₂l_p^(1−η)/(1−η) + α̃₁ ln l_c + α̃₂ ln HC` | `util_total` — P, *Primitives* |
| `α̃₁ = (1−μ̃)λ₁`, `α̃₂ = μ̃φ₃ + (1−μ̃)λ₂` | | inline in `util_total` |
| Parental leisure | CRRA with curvature `η`, linearised below `LEISURE_FLOOR` | `crra_leisure` / `d_crra_leisure` — P, *Primitives* |
| Child leisure | logarithmic, `l_c = 1 − τ_p − i_c − school_t` | `child_leisure`, `log_leisure` = `crra_leisure(·, 1.0)` — P |
| School time | exogenous schedule by age, zero before `T_CHILD_VOICE`, deducted from `l_c`, **not** in the technology | `SCHOOL_TIME_BY_AGE` / `default_school_time` — P; frozen per run in `targets.toml` |
| Welfare weight | `μ̃_t = 1` for `t < T_CHILD_VOICE`; `μ₀ + μ₁(t − 5)` after | `mu_vector`, keyed off `T_CHILD_VOICE = 6` |
| HC technology | `HC' = R·τ_p^σ₁·e_p^σ₂·HC^σ₃·i_c^σ₄`, `σ_j,t = exp(σ_j0 + σ_j1(t−1))`, `R_t = R₀ + R₁(t−1)` | `HC_technology_full` (t ≥ 6) / `HC_technology_parentonly` (t ≤ 5) — P, *Primitives* |
| HC shock | `log HC_{t+1} = log F_t(·) + σ_η·z_{t+1}`, `z ~ N(0,1)` i.i.d., realised after investment, every transition incl. the handoff | `hc_apply_shock` (simulation); `eta_expected_interp` (continuation, Gauss–Hermite `Neta ≤ 5`); `eval_child_value_eta` (handoff) — P |
| Budget | `a' = (1+r)a + y + λ(wh)^(1−τ) − c_p − e_p`, `a' ≥ 0` | `asset_constraint_full` (5 controls) / `asset_constraint_parentonly` (4) — P, *Constraints* |
| Wage | `ln w = β₀ + β_BC·BothCollege + β_age·t + β_age²·t² + interactions + z`, `× 2` | `wage_func` — P, *Primitives* |
| Wage shock | AR(1), Rouwenhorst `Np = 5` | `rouwenhorst` in the constructor — P |
| Terminal value | the child's own lifecycle at age 18, one surface per `BothCollege` state | `eval_child_value` on a `ChildTerminalValue` — P, *Interpolation*; built by `terminal_value_spline` — C |

**`T_CHILD_VOICE = 6`.** Parent-only periods are `t = 1..5`; the child's own study time
enters the technology and the child bargains from `t = 6`. Everything keys off this
constant rather than repeating the literal. `σ₄,t = 0` for `t < 6`.

**States** `(a, k, HC, z)`: household assets; the **`BothCollege` indicator ∈ {0,1}**, a
fixed household type drawn once from `Bernoulli(0.3)` (not parental human capital — it
does not accumulate and enters only the wage equation; `Nk = 2` is exact); the child's
cognitive skill; the AR(1) wage shock. Three regimes: `t = 1..5` parents decide alone (4
controls); `t = 6..16` the child bargains and study time is added (5 controls); `t = 17`
terminal, continuation is the college/transfer value.

**`sigma_eta = 0` is the deterministic technology**, bit-identical to the pre-shock solver
— the quadrature nodes collapse to the point. Since 2026-09-12 `PARENT_DEFAULTS.sigma_eta`
is the fitted 0.0315 (run `exp16b`, preliminary, [`SMM.md`](SMM.md)); pass `sigma_eta = 0.0`
explicitly for the deterministic model.

## 2. The half period at 18

| Model | Code |
|---|---|
| `E_{ε₀}[ max_{d,tr} E_{z₀}[ W_d(tr; ε₀, z₀) ] ]` | `optimal_transfer_work!` / `optimal_transfer_college!` — C, *Transfer stage* |
| Parent keeps at least `δ_P` | `compute_min_assets`, `delta_P` — C, *College feasibility* |
| Taste shock `ε₀ ~ N(0, σ_ε²)`, Gauss–Hermite `Nt = 5` | `sigma_eps`, `Nt` — C |
| Handoff | `parent.sim_a[:, T+1] → child.sim_a_init`, `parent.sim_hc[:, T+1] → child.sim_k_init`, `parent.sim_k[:, 1] → child.sim_bc_init` |

The parent's `hc_grid` and the child's `k_grid` are **the same object** on either side of
the handoff; keep `hc_max` and the child's `k_max` equal (both 1500) or the handoff clips.

## 3. The child's lifecycle (`T = 51`, ages 18–68)

| Model | Code |
|---|---|
| Wage | `ln w = lnw₀ + β_E·E + (α_θ + α_θE·E)(ln θ − m_θ) + (γ₁ + γ₁E·E)age + (γ₂ + γ₂E·E)age²` | `wage_func` — C, *Primitives* |
| Progressive tax | `λ(wh)^(1−τ)` (HSV/Benabou) | `after_tax_income` — C, *Primitives* |
| Psychic cost of college | `κ₀ + κ_θ(ln θ − m_psychic) + κ_ParEd·BothCollege` | `pared_value_offset` — C, *Primitives* |
| College vs work | four college years (18–21), work from 22 | `solve_model_college!` / `solve_model_work!` — C, *Solver* |
| Terminal period | works and consumes everything, no bequest; **no retirement** | — |

A graduate's working life is solved with `E = 1` into the **college** arrays, so
post-graduation policies are read from `sol_*_grad`, not `sol_*_work`. The child's HC
(`θ`) is fixed over its lifecycle — the learning-by-doing term of an earlier version was
removed (`k_next = capital`).

**`kappa_0` is on a centred scale.** With `m_psychic = 0` the psychic cost is the original
uncentred expression; the SMM freezes `m_psychic = 6.2634` (mean log ability at 17 in the
data) in the target file. `CHILD_DEFAULTS` (in `child_lifecycle.jl` since 2026-09-12)
carries the fitted kappas **and** the `m_psychic` they were fitted at, the constructor
defaults read from it, and `check_psychic_centring` errors if a target file carries a
different centring.

---

## Frozen specification decisions

Taken **2026-08-05** so that no work was done against a specification later discarded.
Implement against these.

| # | Decision | Resolution |
|---|---|---|
| 0.1 | Belief correction (N6) | **Not an error.** Cancels exactly to `k₀ + 4b*`. Withdrawn. |
| 0.2 | Wage-equation `Age` units (P8) | **Code correct.** Stata re-indexes age 26 → model period 1. |
| 0.3 | `2 ×` on the parental wage (P7a) | **Intentional.** The regression is on the mean; `2 ×` is household earnings. This is why data moments are **per parent**, not per household. |
| 0.4 | Retirement (C3) | **Removed.** `child_lifecycle.jl` is canonical. |
| 0.5 | ε timing (N1) | **ε observed before the transfer**, `E_ε` outermost — see below. |
| 0.5b | `ā^P` / `δ_P` (N12) | **`δ_P = c_floor = 0.01`.** |
| 0.5c | `z₀` at separation (C6) | **Drawn from the stationary distribution.** |
| 0.6 | Child horizon `T` | **51** — ages 18–68 inclusive. |
| 0.7 | Wage shock process (C5) | **Keep the stationary AR(1)** as a documented approximation. |
| 0.8 | φ normalization (P7b) | **Drop the normalization claim.** `φ₂` is a scale, not a share. |
| 0.9 | College length | **Four years**, ages 18–21, work at 22. Code is right; the paper display was off by one. |
| — | N5, N7 | **Deliberate modelling choices, not errors.** |
| — | C2 (psychic-cost exponent) | **Out of scope** by instruction. |

Later decisions, each dated in the code beside the value: `phi_1 = lambda_1 = 1`
normalisation (2026-08-30); `psi_terminal = 0` (2026-08-30 — the parent values the child's
HC through altruism `omega`, not a separate terminal bonus); `beta_0 = 0.98` (2026-08-28);
`R_1 = 0` (asserted, never estimated); own study only in the technology with school time
exogenous (2026-09-09); the HC shock (2026-09-11, **preliminary, not through the advisor**).

### 0.5 — ε timing, in full

```
E_{ε₀} [ max_{d,tr} E_{z₀} [ W_d(tr; ε₀, z₀) ] ]
```

Nested in that order. `ε₀` is observed at the half period and `z₀` is not, so enrolment and
the transfer condition on `ε₀` but not on realised `z₀`. It is **not**
`max_{d,tr} E_{ε₀,z₀}[W_d]`, which would select the transfer before the shock is seen.

---

## Current defaults (constructor values, not the paper's table)

`PARENT_DEFAULTS` in `parent_family.jl` and `CHILD_DEFAULTS` in `child_lifecycle.jl` are
the single source of truth for the two blocks; nothing should hardcode a parameter that
lives there. Since 2026-09-12 both carry the sixteen-parameter fit `2026-09-11_182836_exp16b`
at full precision ([`SMM.md`](SMM.md), Part 3), so `run_all.jl`, the notebook and the SMM's
incumbent all start from the same fitted model. Build the child with
`ConSavLaborCollege_AR1(; Na = 30, Nk = 30, Nt = 5, CHILD_DEFAULTS...)`.

**Parent** — `Parent_child_interaction_age_specific_AR1`

| Param | Value | Meaning |
|---|---|---|
| `T` | 17 | periods = child ages 1–17 (no age-0 period) |
| `rho`, `eta` | 1.5, 2.0 | CRRA; leisure curvature |
| `phi_1`, `phi_2`, `phi_3` | **1.0** (normalised), 0.1963, 1.5337 | weights on consumption / parental leisure / child HC |
| `lambda_1`, `lambda_2` | **1.0** (normalised), 13.862 | child's weights on leisure / HC |
| `mu_0`, `mu_1` | 1.0, −0.04 | welfare weight after `T_CHILD_VOICE` |
| `R_0`, `R_1` | 48.34, **0** (fixed) | HC technology TFP, `R_t = R_0 + R_1(t−1)` |
| `sigma_1_0/1` | −0.889, −0.093 | elasticity to parental time (logs: `σ_jt = exp(σ_j0 + σ_j1(t−1))`) |
| `sigma_2_0/1` | −3.692, −0.098 | elasticity to money |
| `sigma_3_0/1` | −0.90, 0.0 | elasticity to own HC, `exp(−0.9) = 0.41`, flat; `≥ 1` is explosive |
| `sigma_4_0/1` | −6.099, 0.130 | elasticity to the child's own study; zero before `T_CHILD_VOICE` |
| `sigma_eta`, `Neta` | 0.0315, 5 | HC shock SD (fitted; 0 = deterministic); Gauss–Hermite nodes |
| `tau`, `tax_lambda`, `y` | 0.18, 0.82, 0.6 | progressivity, tax level, non-labour income (`λ(wh)^(1−τ)`) |
| `r`, `beta_0` | 0.03, 0.98 | interest rate; discount factor |
| `a_max`, `Na` | 100.0, 30 | asset grid (was 50: simulated assets reached 281.5 with 0.43% off-grid; 0.10% at 100) |
| `Nk` | 2 | `BothCollege ∈ {0, 1}`, drawn from `Bernoulli(0.3)` |
| `hc_min`, `hc_focus`, `hc_max`, `Nhc` | 50, 700, 1500, 30 | HC grid in W-score units, focused on `[50, 700]` with a sparse tail (`create_focused_grid`) |
| `Np`, `p_ar1`, `sigma_p` | 5, 0.9, 0.1 | AR(1) wage shock, Rouwenhorst (matches the sd and autocorrelation exactly; Tauchen overstated the sd by 21%). 7 → 5 → 3 leaves every moment flat to the third digit |
| `simN`, `seed` | 5000, 1234 | simulation size; every RNG is seeded from `seed` |
| `β0 … β_age_capital` | 2.799, 0.308, 0.023, −4.3e-4, 0.017, −4.3e-4 | the Stata wage regression |
| `school_time` | `default_school_time(T)` | ~0.32–0.35 of the week from age 6 |

**Child** — `ConSavLaborCollege_AR1`

| Param | Value | Meaning |
|---|---|---|
| `T`, `t_college` | 51, 4 | horizon (ages 18–68); college years |
| `beta`, `rho`, `eta`, `phi` | 0.97, 1.0 (1.5 in `CHILD_DEFAULTS`), 2.0, 18.0 | discount; CRRA; inverse Frisch; labour disutility scale |
| `r`, `y`, `tau`, `tax_lambda` | 0.03, 0.6, 0.18, 0.82 | as the parent |
| `c_floor`, `delta_P` | 0.01, 0.01 | consumption floor (= optimizer bound); min retained parental asset |
| `a_max`, `Na` | 100.0, 30 | the **child's** asset grid — must cover the parent's terminal assets plus 51 periods of accumulation |
| `ap_min`, `ap_max`, `Nap` | `delta_P`, `a_max`, `Na` | **parental** asset grid, separate since N13. Indexes the transfer arrays and the terminal-value spline; starts at `delta_P` and carries an exact node at the college threshold `a_req[1] + delta_P`, so there is no dead band |
| `k_max`, `Nk` | 1500, 30 | the child's HC (`θ`) grid = the parent's `hc_grid` |
| `w`, `lnw0`, `beta_E`, `alpha_theta`, `alpha_thetaE`, `gamma1`, `gamma1E`, `gamma2`, `gamma2E`, `m_theta` | 12.5, `log w − 0.4144`, −0.294, 0.654, 0.322, 0.0234, 0.0318, −1.99e-4, −3.14e-4, 7.3486 | the wage equation — [`WAGE_PROCESS.md`](WAGE_PROCESS.md) |
| `kappa_0`, `kappa_theta`, `kappa_ParEd`, `m_psychic` | −0.357, −3.619, −0.108, 6.2634 | psychic cost of college, centred at `m_psychic` (`CHILD_DEFAULTS`, fitted) |
| `college_cost` | 1.2 | annual cost, model units |
| `psi_terminal`, `kappa_terminal`, `omega` | **0.0**, 8.787 (fitted), 0.5 (0.3 in `CHILD_DEFAULTS`) | parent's terminal weight on child HC (off by instruction) / on own retained assets / altruism |
| `mu` | 0.5 | the parent's weight in the college decision; the family coefficient on the child's value is `(1 − mu) + mu·omega` |
| `Np`, `p_ar1`, `sigma_p` | 5, 0.95, 0.2 | AR(1) wage shock, Rouwenhorst |
| `Nt`, `sigma_eps` | 5, 1.142 (fitted) | Gauss–Hermite nodes and SD of the taste shock `ε₀` |

**Grid caps by instruction**: assets and HC ≤ 30 nodes, shock discretisations ≤ 5 (`Np`,
`Nt`, `Neta`) — all converged at these sizes. The child's `Na`/`Nk` at 30 rather than 50
costs ~7pp on the college share and nothing else: it is a threshold choice, so its
location tracks grid resolution. `run_all.jl` uses child `30×30×5`, parent `30×2×30`,
`simN = 5000`; `--quick` uses `20×20×5`, `10×2×10`, 500.

Where the numbers come from: parent preferences, HC technology and the child's psychic
cost are **estimated** ([`SMM.md`](SMM.md)); the child wage process is calibrated from
Daruich & Fernández (2023) and Colas ([`WAGE_PROCESS.md`](WAGE_PROCESS.md)); grid sizes,
solver settings and numerical guards are measured in [`ERRORS.md`](ERRORS.md).

---

## Running it

### Files

| File | What it is |
|---|---|
| `code/run_all.jl` | **One reproducible end-to-end run.** Solve, simulate, diagnose, tables, PDF. |
| `code/transfer_CRRA_wage.ipynb` | Interactive driver: counterfactuals and figures. |
| `code/src/parent_family.jl` | **Parent problem.** Struct, constructor, backward-induction solver, objectives, constraints, simulators. |
| `code/src/child_lifecycle.jl` | **Child lifecycle — canonical.** No retirement, progressive tax. |
| `code/src/diagnostics.jl` | Accuracy checks: Bellman residuals, domains, monotonicity, gradients. |
| `code/src/tables.jl` | LaTeX tables (`threeparttable`) and the PDF build. |
| `code/src/paths.jl` | **Every path in the project.** Nothing else hard-codes a folder name. |
| `code/src/manifest.jl` | Run provenance: `write_manifest(dir; params...)`. |
| `code/src/tiktak.jl`, `code/smm/` | Estimation — [`SMM.md`](SMM.md). |
| `model.txt` | LaTeX model specification from the paper. |
| `../Project.toml`, `Manifest.toml` | Dependencies, pinned to the verified set (Julia 1.11). |
| `../output/figures/`, `../output/tables/` | tracked in git; `../output/data/` and `../output/smm_runs/` are not. |

Superseded modules (`child_lifecycle_ret.jl`, `child_lifecycle_ar1.jl`) are gone or
archived; do not fix them.

### Setup and the interactive session

```bash
./tools/setup-git-filters.sh                              # once per clone
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Prefer the persistent `julia` MCP session over `julia script.jl` for diagnostics (see
[`CLAUDE.md`](../CLAUDE.md)); load source with `includet` so `Revise` tracks edits.
Load order matters: `child_lifecycle.jl` **before** `parent_family.jl`, because the parent
file names `ConSavLaborCollege_AR1` in its type signatures.

```julia
include("src/paths.jl"); include("src/manifest.jl"); include("src/diagnostics.jl")
include("src/child_lifecycle.jl"); include("src/parent_family.jl")
child = ConSavLaborCollege_AR1(; Na=30, Nk=30, Nt=5, simN=5000, CHILD_DEFAULTS...)
solve_model_work!(child); solve_model_college!(child)
optimal_transfer_work!(child); optimal_transfer_college!(child)
V = terminal_value_spline(child; s = 10.0)           # one surface per BothCollege state
m = Parent_child_interaction_age_specific_AR1(Na=30, Nk=2, Nhc=30, simN=5000)
m.V_child_interp = V
solve_model!(m); simulate_model!(m)
```

In the notebook the same order is cells 3 → 4 → 6 → everything after; the notebook
additionally solves ~20 belief-specific parent models for the subjective-expectations
experiment, which is the expensive part.

> ⚠️ **Before trusting a run.** `check_simulation` counts non-finite states,
> `check_solver_domain` measures the *solution* leaving the grid (which forward simulation
> cannot see), and `check_feasibility_mask` checks the NaN pattern against both theoretical
> masks. One caveat remains: **P5**, the continuation interpolation on the child side moves
> optimal labour supply by up to 0.11–0.17 at some states, and refining the grid does not
> shrink it. See [`ERRORS.md`](ERRORS.md).

### One reproducible run

```bash
cd code && julia --project=.. run_all.jl            # production grids, ~half a minute
cd code && julia --project=.. run_all.jl --quick    # smoke test, ~20 s
```

Solves the child lifecycle and the parent problem, simulates, runs every accuracy
diagnostic, writes all LaTeX tables to `output/tables/`, and compiles them into
`output/reports/all_tables.pdf`. Reproducible by construction: every RNG is seeded from
one `SEED`; `Project.toml`/`Manifest.toml` pin the package set; each table emits a
`.meta.toml` with the git commit, timestamp and parameters; and the PDF wrapper is
generated from whatever `.tex` files are on disk, so it can never go stale.

### LaTeX tables

`code/src/tables.jl` emits `threeparttable` + `booktabs` tables in the same format as
`Redistribution_and_Human_Capital/{Tables,outcomes}` — `\toprule\toprule` … `\midrule`
… `\bottomrule\bottomrule`, `[H]` placement, `\tnote{}` footnotes. Each file is
`\input`-able straight into the paper; the preamble needs `booktabs`, `threeparttable`
and `float`.

| writer | produces |
|---|---|
| `table_college_work(path_choice, name)` | counts and shares, like `base_college_work_choice.tex` |
| `table_outcomes(models, labels, name)` | end-of-family outcomes, like `resource_summary.tex` |
| `table_belief_groups(...)` | per-belief-group means, like `hetero_table.tex` |
| `table_diagnostics(pairs, name)` | numerical diagnostics |
| `write_table(name; ...)` | the generic builder for anything else |
| `build_tables_pdf()` | every `.tex` in `output/tables/` into one PDF |

### Writing output

Never build a path by hand. `paths.jl` gives you:

| Call | Returns | Creates dir? |
|---|---|---|
| `figdir("Baseline")` | `output/figures/Baseline` | no |
| `figpath("Baseline")` | same | yes |
| `tabpath()`, `datapath()`, `reportpath()` | the matching `output/` subfolder | yes |
| `unique_path(dir, "name")` | `name.pdf`, else `name_2.pdf`, … | yes |
| `sanitize(title)` | a safe filename stem from a plot title | — |

Record what produced a set of results:

```julia
write_manifest(figpath("Parameters"); experiment = "sigma counterfactuals",
                                      mu_1 = -0.04, rho = 1.5, Na = 30, simN = 5000)
```

---

## Numerical guards worth knowing about

Both were added after the parent solve died on NaN iterates; the derivations are in
[`ERRORS.md`](ERRORS.md), "Numerical guards in the parent solver".

- `LEISURE_FLOOR = 1e-2`. The child's leisure `1 − τ_p − i_c − school` is a *nonlinear*
  constraint, not a box bound, so SLSQP evaluates points that violate it — the only
  quantity it can drive non-positive. Below the floor the log is **linearised**, not
  flattened: value and slope both match at the floor, so the derivative stays bounded by
  `1/L` instead of cliffing. It is `1e-2` rather than `1e-4` because the parent's own
  leisure is CRRA with curvature `η`, where the bounded derivative is `L^(−η)`. Verified
  inactive at the optimum.
- `TIME_FLOOR = 1e-3` on every time share (`τ_p`, `i_c`, `h_p`) in every parent
  optimisation, with the warm start from `t+1` clamped into the current box.
- `snap_parent` corrects only float-sized violations (`tol = 1e-10`), by design: a
  genuinely out-of-bounds value passes through rather than being silently rewritten.
- `solve_model!` **throws** below a 95% converged share rather than printing, because the
  notebook wraps counterfactuals in `@suppress_output`.
- Dierckx `Spline2D` clamps the value outside its range but keeps returning the boundary
  derivative; `eval_child_value` clamps both together, or SLSQP's line search breaks and
  surfaces as `ROUNDOFF_LIMITED`.

## Known issues

The full audit is in [`ERRORS.md`](ERRORS.md), with severity, measurement and what was
done. Open at the time of writing (2026-09-12), in the order they matter:

1. 🟠 **P13** — the HC technology had no idiosyncratic shock, so the child block could not
   fit completion by ability and by parental education jointly. `sigma_eta` is the
   proposed fix, implemented and estimated as a **preliminary** experiment; it is a
   specification change and awaits the advisor.
2. 🟡 **P7c** — `kappa_ParEd` is targeted on *either*-parent college while the model means
   *both*. Open by instruction.
3. 🟡 **P5** — the continuation interpolation moves policies on the **child** side (the
   parent side is fixed). Re-solving against a cubic spline instead of `Gridded(Linear())`
   moves optimal labour supply by up to 0.11–0.17 of the time endowment, and quadrupling
   the grid does not shrink it. The Bellman residual is blind to this because it
   re-evaluates the stored policy rather than re-optimising.
4. 🟡 **P7b** — the `BothCollege` share is hardcoded at `Bernoulli(0.3)` and needs an
   empirical source from the estimation sample.
5. 🟡 **P10**, 🟠 **P11** — the leisure fix and the HC recalibration each exposed a
   calibration tension (the `φ₂` level; a declining `τ_p` against the college margin) that
   was left for the estimation to resolve; both stay listed as open until the SMM is final.
6. ⚪ **G3** — `create_focused_grid` builds a non-monotone grid when the range is under 3.0.
7. ⏸️ **C2**, **C8** — deferred out of scope by instruction.

---

## Change log (layout only; no model logic)

**2026-08-02** — code to `code/`, model spec and notes to `docs/`, figures to
`output/figures/`. Parent model extracted from the notebook into `parent_family.jl`
verbatim; superseded notebooks to `archive/` (pre-extraction notebook at
`archive/Combined Models/Full model/transfer_CRRA_wage_ORIGINAL.ipynb`). `paths.jl` replaced
all 30 hard-coded `joinpath(@__DIR__, "plots", …)` calls and fixed the `"Plots"`/`"plots"`
inconsistency; `plot_family_counterfactuals` no longer creates a dated directory on
`save=false`. `manifest.jl` added. Notebook outputs stripped on commit via
`tools/nbstrip.py`; LFS scoped to `archive/**/*.ipynb`. `Manifest.toml` committed.
