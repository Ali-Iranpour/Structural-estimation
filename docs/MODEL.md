# Model ↔ Code Map

Maps each equation in [`model.txt`](model.txt) to the code that implements it.
Line numbers are as of 2026-08-02.

- **P** = `code/src/parent_family.jl`
- **C** = `code/src/child_lifecycle_ret.jl`

⚠️ marks a known departure from the model — see "Known issues" in [`GUIDE.md`](GUIDE.md).

---

## 1. Childhood and adolescence (`t = 1..17`)

| Model | Equation | Code |
|---|---|---|
| Parent utility | `U_p = φ₁c^(1-ρ)/(1-ρ) − φ₂h^(1+η)/(1+η) + φ₃ln HC` | `util_parent` — P:704 |
| Family utility | `U_f = φ₁c^(1-ρ)/(1-ρ) − φ₂h^(1+η)/(1+η) + α̃₁ln l_c + α̃₂ln HC` | `util_total` — P:688 |
| `α̃₁ = (1−μ̃)λ₁`, `α̃₂ = μ̃φ₃ + (1−μ̃)λ₂` | | inline in `util_total` |
| Welfare weight | `μ̃_t = 1` (t<`T_CHILD_VOICE`); `1 − μ₁(t−(`T_CHILD_VOICE`−1))` otherwise | `mu_vector` in constructor, keyed off `T_CHILD_VOICE = 7` |
| | | ⚠️ code uses `μ_0 + μ_1(t−5)` with `μ_1 = −0.04`, i.e. the opposite sign convention |
| Budget constraint | `a' = (1+r)a + y − T(wh) + b − c_p − e_p`, `a' ≥ 0` | `asset_constraint_full` — P:759 (5 controls)<br>`asset_constraint_parentonly` — P:796 (4 controls) |
| Progressive tax | `T(·)` via HSV/Benabou | `λ(wh)^(1−τ)` inline in the objectives; marginal `λ(1−τ)(wh)^(−τ)w` |
| Parent time | `h_p + τ_p + l_p = 1` | `constraint_min_leisure_full` — P:835<br>`constraint_min_leisure_parentonly` — P:846 |
| Child time | `τ_c + τ_p + l_c = 1` | `constraint_child_time` — P:857 |
| HC production (t≤7) | `ln HC' = ln R + σ₁ln τ_p + σ₂ln e_p + σ₃ln HC` | `HC_technology_parentonly` — P:732 |
| HC production (t>7) | `… + σ₄ln τ_c` | `HC_technology_full` — P:721 |
| | `σ_jt` entered as logs: `σ_jt = exp(σ_j0 + σ_j1(t−1))` | `sigma_*_vector` in constructor — P:180 |
| Wage equation | `ln w = β₀ + β₁BC + β₂Age + β₃Age² + β₄(BC×Age) + β₅(BC×Age²) + z` | `wage_func` — P:747 |
| | | ⚠️ `Age` ← model period `t`; `2×` multiplier not in the model |
| AR(1) shock | `z_t = ρz_{t−1} + ε` | `tauchen(Np, p_ar1, sigma_p, 0, 3)` in constructor |
| | | ⚠️ `model.txt` says random walk (ρ=1); code uses ρ=0.9, `Np=3` |

### Value functions

| Stage | Model | Code |
|---|---|---|
| Childhood `t ≤ 5` | max over `c_p, e_p, h_p, τ_p` | `solve_model!` loop `(T_CHILD_VOICE-1):-1:1` → `obj_work_period_parentonly` |
| Adolescence `5 < t < 17` | max over `c_p, e_p, h_p, τ_p, τ_c` | `solve_model!` loop `(T-1):-1:T_CHILD_VOICE` → `obj_work_period_full` |
| Terminal `t = 17` | `U_f + βE[V^CD_{T_L−1}]` | `solve_model!` terminal block → `obj_last_period_full` — P:537 |
| | `V^CD` enters as | `model.V_child_interp` (a `Dierckx.Spline2D` built in the notebook) |

⚠️ `obj_work_period_full` (`grad[4]`) and `obj_work_period_parentonly` (`grad[3]`) carry a
spurious `dV_dk_sum` term. `k` is the fixed `BothCollege` indicator, so `∂k'/∂h_p = 0`.
`obj_last_period_full` is correct.

---

## 2. Transfer and college decision (`t = T_L = 18`)

| Model | Equation | Code |
|---|---|---|
| Parent terminal value | `V^P = ψ ln HC + κ ln a_term + ω V^C` | `terminal_value` — C:865 (returns `ψ ln HC + κ ln a_term`; `ω` folded into `coef`) |
| Child terminal value | `V^C = max{V^E(ε₀), V^W}` | ⚠️ **not implemented as written** — see below |
| Family objective | `(1−θ)V^C + θV^P` | `coef = (1−mu) + mu*omega`; `f = coef*V_child + mu*V_parent`<br>`obj_transfer_work` — C:793, `obj_transfer_college` — C:829 |
| | θ ≡ `model.mu` | note the weights sum to `1 + θω`, not 1 — faithful to `model.txt` |
| Transfer choice, ε known | `max_{0≤tr≤a} E_z[…]` | `optimal_transfer_college!` — C:745 (ε-specific)<br>`optimal_transfer_work!` — C:697 |
| Transfer choice, ε integrated | `max_tr E_{ε,z}[…]` | `optimal_transfer_exp_college!` — C:1055 |

⚠️ **The discrete max is taken in the wrong place.** The notebook builds the terminal value
as `v_max = safe_maximum.(sol_exp_v_college, sol_tr_v_work)` = `max(E_ε[V^E], V^W)`, but the
model requires `E_ε[max(V^E(ε), V^W)]`. `simulate_model_family!` (C:888) implements the
correct rule agent-by-agent, so solve and simulate currently disagree.

---

## 3. Adulthood (`t = 18..68`)

### Work path

| Model | Equation | Code |
|---|---|---|
| Utility | `U^W = c^(1−ρ)/(1−ρ) − φh^(1+η)/(1+η)` | `util_work` — C:469 |
| Budget | `a' = (1+r)a + T(wh) − c + b` | `obj_work_period` — C:346; `asset_constraint_work` — C:422 |
| Progressive tax | `λ(wh)^(1−τ)` | `after_tax_income` — C:156; derivative `d_after_tax_dh` — C:162 |
| Wage | `w = w₀(1 + α·HC)·z` | `wage_func` — C:150 |
| HC | `HC' = HC + h` | `k_next = capital + h` in `obj_work_period` |
| Bellman | `V^W = max_{c,h}{U^W + βE[V^W']}` | `solve_model_work!` — C:169 |

### College path

| Model | Equation | Code |
|---|---|---|
| Utility | `U^E = c^(1−ρ)/(1−ρ) − κ_X` | `util_college` — C:479 |
| Psychic cost | `κ_X = κ/(HC+1)²` | ⚠️ **C:485 uses `(k+1)^4`** |
| Budget | `a' = (1+r)a − c − c_college + b` | `obj_college_period_general` — C:389; `asset_constraint_college` — C:436 |
| HC (homogeneous) | `HC' = HC + h^E` | `k_next = capital + college_boost` |
| HC (perceived, belief `b_m`) | `H̃C' = H̃C + b_m` | `k_next = k + belief_values[m]` — P:1160 |
| Graduation correction | `HC' = H̃C + b* + (T_E−1)(b* − b_m)` | `k + college_boost_true + 3*(…)` — P:1160 ✓ `3 = T_E − 1` |
| Taste shock at entry | `+ ε₀` only at `t = 18` | `(t==1 ? ε : 0.0)` in `obj_college_period_general` ✓ |
| Bellman | `V^E`, continuation → `V^W` after `t_college` | `solve_model_college!` — C:257 |

---

## 4. In the code but not in the model

| Code | Location | Note |
|---|---|---|
| **Retirement** | `t_retire = 42`, pension = 0.5 × after-tax notional earnings | `solve_model_work!` — C:169; `util_retire` — C:461; `pension_amount` — C:539. `model.txt`: "The model has no retirement stage." |
| `a_next ≤ a_max` | `asset_constraint_max` — P:778, P:814 | numerical device, no counterpart in the model |
| `WAGE_SCALING_FACTOR = 0.584` | P:327, C:144 | undocumented units normalization |
| `2 ×` in parent wage | P:747 | makes it the household total, not the mean |
| `−1e8` penalty branches | `util_total` P:688, `HC_technology_*` P:721/732 | objective/gradient become inconsistent when triggered |

## 5. In the model but not in the code

| Model | Note |
|---|---|
| `b_t` as distinct from `y` | code has a single scalar `model.y` |
| Random walk (`ρ = 1`) | code uses `ρ = 0.9` (parent) / `0.95` (child) |
| `(φ₁,φ₂,φ₃)` normalized to sum to 1 | code: `1.0 + 20.0 + 0.03 = 21.03` |
| `V^E` case at `t = 22` | code implements 4 college years (18–21), entering work at 22 — matches the prose, not the displayed value function |
