# Model ↔ Code Map

Maps the model in [`model.txt`](model.txt) to the code that implements it, and records the
specification decisions that are frozen.

- **P** = [`code/src/parent_family.jl`](../code/src/parent_family.jl)
- **C** = [`code/src/child_lifecycle.jl`](../code/src/child_lifecycle.jl)

References are by **section**, not line number. Both files are organised
primitives → constraints → objectives → solver → simulation, with banner comments, so a
section name stays valid when code moves. (This file previously carried line numbers "as of
2026-08-02" and pointed at `child_lifecycle_ret.jl`, which no longer exists.)

---

## 1. Childhood and adolescence (`t = 1..17`, child ages 1–17)

| Model | Equation | Code |
|---|---|---|
| Parent utility | `U_p = φ₁c^(1−ρ)/(1−ρ) + φ₂l_p^(1−η)/(1−η) + φ₃ ln HC`, `l_p = 1 − h_p − τ_p` | `util_parent` — P, *Primitives* |
| Family utility | `U_f = φ₁c^(1−ρ)/(1−ρ) + φ₂l_p^(1−η)/(1−η) + α̃₁ ln l_c + α̃₂ ln HC` | `util_total` — P, *Primitives* |
| `α̃₁ = (1−μ̃)λ₁`, `α̃₂ = μ̃φ₃ + (1−μ̃)λ₂` | | inline in `util_total` |
| Parental leisure | CRRA with curvature `η`, linearised below `LEISURE_FLOOR` | `crra_leisure` / `d_crra_leisure` — P, *Primitives* |
| Child leisure | stays logarithmic, `l_c = 1 − τ_p − i_c` | `log_leisure` = `crra_leisure(·, 1.0)` |
| Welfare weight | `μ̃_t = 1` for `t < T_CHILD_VOICE`; declining after | `mu_vector`, keyed off `T_CHILD_VOICE = 6` |
| HC technology | `HC' = R·τ_p^σ₁·e_p^σ₂·HC^σ₃·i_c^σ₄`, `σ_j,t = exp(σ_j0 + σ_j1(t−1))` | `HC_technology_full` (t ≥ 6) / `HC_technology_parentonly` (t ≤ 5) — P, *Primitives* |
| Budget | `a' = (1+r)a + y + λ(wh)^(1−τ) − c_p − e_p`, `a' ≥ 0` | `asset_constraint_full` (5 controls) / `asset_constraint_parentonly` (4) — P, *Constraints* |
| Wage | `ln w = β₀ + β_E·BothCollege + β_age·t + β_age²·t² + interactions`, `× 2` | `wage_func` — P, *Primitives* |
| Terminal value | the child's own lifecycle at age 18 | `eval_child_value` — P, *Interpolation*; built by `terminal_value_spline` — C |

**`T_CHILD_VOICE = 6`.** Parent-only periods are `t = 1..5`; the child's own study time
enters the technology and the child bargains from `t = 6`. Six things key off this constant
rather than repeating the literal.

## 2. The half period at 18

| Model | Code |
|---|---|
| `E_{ε₀}[ max_{d,tr} E_{z₀}[ W_d(tr; ε₀, z₀) ] ]` | `optimal_transfer_work!` / `optimal_transfer_college!` — C, *Transfer stage* |
| Parent keeps at least `δ_P` | `compute_min_assets`, `delta_P` — C, *College feasibility* |
| Handoff | `parent.sim_hc[:, T+1] → child.sim_k_init`, `parent.sim_k[:, 1] → child.sim_bc_init` |

## 3. The child's lifecycle (`T = 51`, ages 18–68)

| Model | Code |
|---|---|
| Wage | `ln w = lnw₀ + β_E·E + (α_θ + α_θE·E)(ln θ − m_θ) + (γ₁ + γ₁E·E)age + (γ₂ + γ₂E·E)age²` | `wage_func` — C, *Primitives* |
| Progressive tax | `λ(wh)^(1−τ)` | `after_tax_income` — C, *Primitives* |
| Psychic cost of college | `κ₀ + κ_θ ln θ + κ_ParEd·BothCollege` | `pared_value_offset` — C, *Primitives* |
| College vs work | four college years (18–21), work from 22 | `solve_model_college!` / `solve_model_work!` — C, *Solver* |

A graduate's working life is solved with `E = 1` into the **college** arrays, so
post-graduation policies are read from `sol_*_grad`, not `sol_*_work`.

---

## Frozen specification decisions

Taken **2026-08-05** so that no work was done against a specification later discarded.
Implement against these. (Merged here from the former `SPEC_DECISIONS.md`.)

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

### 0.5 — ε timing, in full

```
E_{ε₀} [ max_{d,tr} E_{z₀} [ W_d(tr; ε₀, z₀) ] ]
```

Nested in that order. `ε₀` is observed at the half period and `z₀` is not, so enrolment and
the transfer condition on `ε₀` but not on realised `z₀`. It is **not**
`max_{d,tr} E_{ε₀,z₀}[W_d]`, which would select the transfer before the shock is seen.

---

## Where the numbers come from

| | |
|---|---|
| Parent preference and HC-technology parameters | estimated — [`SMM.md`](SMM.md) |
| Child wage and psychic cost | calibrated from Daruich & Fernández (2023) and Colas — [`WAGE_PROCESS.md`](WAGE_PROCESS.md) |
| Grid sizes, solver settings, numerical guards | [`ERRORS.md`](ERRORS.md) |

`PARENT_DEFAULTS` in `parent_family.jl` is the single source of truth for the parent block;
nothing should hardcode a parameter that lives there.
