# Known Errors

Audit of the model code against [`model.txt`](model.txt). Line numbers are current as of
**2026-08-02** and were re-verified against the files after the reorganization.

**Nothing in this list has been fixed.** This is the to-do list.

| Severity | Meaning |
|---|---|
| 🔴 **Critical** | Invalidates published results. Fix before using any output. |
| 🟠 **High** | Materially biases results, or hides failure so you cannot tell whether they are biased. |
| 🟡 **Medium** | Affects interpretation, robustness, or reproducibility. |
| ⚪ **Low** | Cosmetic, maintainability, or latent. |

### Which files are live

| File | Status |
|---|---|
| `code/src/parent_family.jl` | **LIVE** — parent problem |
| `code/src/child_lifecycle_ret.jl` | **LIVE** — the child module the notebook includes |
| `code/transfer_CRRA_wage.ipynb` | **LIVE** — driver |
| `code/src/child_lifecycle_ar1.jl` | **NOT INCLUDED** — kept for reference. Its issues are latent unless you switch to it. |

### Where each error lives

`parent_family.jl` was extracted from the notebook, so the `P` errors exist in **exactly
one place**. Verified: zero copies of `dV_dk_sum`, `asset_constraint_max`, `util_total`,
`solve_model!`, `simulate_model!` etc. remain in `code/transfer_CRRA_wage.ipynb`.

The notebook is still **fully affected at runtime** — it includes `parent_family.jl` and
calls its solvers — but there is now one site to fix per bug, not two that can drift.

`archive/Combined Models/Full model/transfer_CRRA_wage_ORIGINAL.ipynb` still contains all
of them inline. That is the frozen pre-extraction copy; do not fix it.

Errors that genuinely live in the notebook are the `N` and `M` items: parameter values
passed at construction, the terminal-value construction, experiment design, and labels.

### Summary

| # | Issue | File | Severity |
|---|---|---|---|
| P1 | Spurious `∂V/∂k` in the labor-supply gradient | parent_family | 🔴 |
| N1 | College choice taken outside the ε expectation (11 sites) | notebook | 🔴 |
| P2 | Unseeded RNG — counterfactuals lack common random numbers | parent_family + notebook | 🔴 |
| C1 | `-Inf` sentinel reachable under the enforced constraint | both child modules | 🟠 |
| C2 | Psychic cost uses `^4`, model says `^2` | both child modules | 🟠 |
| C3 | Retirement exists in code, not in the model | child_lifecycle_ret | 🟠 |
| N2 | 65 × `@suppress_output` discards all convergence diagnostics | notebook | 🟠 |
| P3 | Simulated states never clamped; artificial `a ≤ a_max` in the solve | parent_family | 🟠 |
| C4 | Asymmetric transfer optimization across the discrete choice | both child modules | 🟠 |
| C5 | Shock discretization too coarse for the assumed persistence | all | 🟠 |
| P4 | Objective/gradient inconsistent in `-1e8` penalty branches | parent_family | 🟠 |
| P5 | Piecewise-linear continuation value under gradient-based SLSQP | all | 🟠 |
| N3 | CEV formula assumes homotheticity the value function lacks | notebook | 🟠 |
| N4 | θ-experiment baseline uses a different ω than its treatment arms | notebook | 🟠 |
| N5 | `psi_terminal_belief_bin` computed and never used | notebook | 🟠 |
| N6 | Belief correction can drive human capital negative | child modules + notebook | 🟠 |
| N7 | Res-vs-Exp arms asymmetric (child y=1.08, parent y=1.2) | notebook | 🟡 |
| N8 | Model/label order swapped in one figure | notebook | 🟡 |
| N9 | Tax counterfactual labels do not match the τ values used | notebook | 🟡 |
| P6 | NaN guard unreachable; poisons the backward init chain | parent_family | 🟡 |
| P7 | φ weights not normalized; `BothCollege` share hardcoded | parent_family | 🟡 |
| P8 | Verify `Age` units in the wage equation | parent_family | 🟡 |
| M1 | Tables are never written to disk | notebook | 🟡 |
| C6 | `stationary_dist` used in solve, median state in simulation | both child modules | 🟡 |
| C7 | `findfirst` can return `nothing` | both child modules | ⚪ |
| C8 | Duplicate `discrete_draw`; unused `Nt` dimension | child_lifecycle_ar1 | ⚪ |
| X1 | No accuracy diagnostics anywhere | all | 🟠 |

---

# `code/src/parent_family.jl`

## 🔴 P1 — Spurious `∂V/∂k` in the labor-supply gradient

**Lines 636 and 678.** `k` is the fixed `BothCollege` indicator, so `k_next = capital`
(lines 611, 658) and `∂k_next/∂h_p = 0`. But both gradients still add `dV_dk_sum`:

```julia
# line 636, obj_work_period_full        (t = 16..8)
grad[4] = dutil_dh_p + model.beta_vector[t] * (dV_da_sum * marginal + dV_dk_sum)
# line 678, obj_work_period_parentonly  (t = 7..1)
grad[3] = -model.phi_2_vector[t] * (h_p ^ model.eta) +
          model.beta_vector[t] * (marginal * dV_da_sum + dV_dk_sum)
```

`k_grid = range(0.0, 1.0, length=2)`, so the k-direction gradient equals
`V(BothCollege=1) − V(BothCollege=0)` — the entire lifetime value gap between education
types, a large positive constant. It is added to the hours FOC at every grid point,
driving `h_p` to its upper bound for all `t ≤ 16`.

`obj_last_period_full` (line 630s, `grad[4] = dutil_dh_p + β*(dV_da*marginal)`) is
**correct**, which confirms this is an incomplete edit from when `k` was parental human
capital, not a deliberate choice.

**Fix.** Delete `dV_dk_sum` from both gradients. Two lines. Every parental labor-supply
and consumption policy for `t ≤ 16` changes.

## 🔴 P2 — Unseeded RNG (with the notebook)

**Line 941** (`simulate_model!`) and **line 1091** (`simulate_model_hetero!`):

```julia
next_state = sample(1:model.Np, Weights(model.p_transition[current_state, :]))
```

Global RNG. Plus `rand(Beta(α, β), simN)` in the notebook at **cells 40, 77, 78, 79
(line 9)**. Every counterfactual arm therefore draws different AR(1) shock paths, so each
plotted difference is (treatment effect + Monte Carlo noise over 5,000 agents × 17
periods), and results change on every re-run.

The child modules do this correctly via the stored `draws_uniform_p`. The `seed::Int=1234`
constructor kwarg is **never used** — lines 288–291 hardcode `MersenneTwister(1234/5678/9012)`,
and `rng_p = MersenneTwister(3456)` is created and never referenced.

**Fix.** Pre-draw a seeded `draws_uniform_p::Matrix{Float64}` in the constructor and use
the existing `discrete_draw`. Seed the Beta draw.

## 🟠 P3 — Simulated states never clamped; artificial upper asset bound

**Line 988** (and 1119) writes `sim_a[i, t+1]` with no floor at `1e-6` and no cap at
`a_max`, unlike the child simulation which does `max(a_next, a_min)`
(child_lifecycle_ar1.jl:567, 1000).

Separately, `asset_constraint_max` (**line 778**) and `asset_constraint_max_parentonly`
(**line 814**) impose `a_next ≤ a_max = 50`. **This constraint is not in the model.** It is
a numerical device to keep iterates on the grid, and with
`sim_a_init ~ LogNormal(0.296, 1.402)` (**line 290**; mean 3.59, p99 ≈ 35) it binds for the
right tail — exactly the households whose transfer behaviour the paper is about.

## 🟠 P4 — Objective and gradient disagree in the penalty branches

`util_total` returns a flat `-1e8` at **line 692** when `c ≤ 0 || i_c ≤ 0 || leisure_c ≤ 0`,
and `util_parent` at **line 706**. But `obj_work_period_full` computes
`term_leisure_c = (1-μ)λ₁/leisure_c * (-1)` from the smooth formula regardless — with
negative `leisure_c` that is a wrong-signed gradient attached to a constant objective.

`HC_technology_full` (**line 723**) and `HC_technology_parentonly` (**line 734**) return
`-1e8` as a *human-capital level*, which is then multiplied by `σ₁/t_p` in the gradient.

SLSQP assumes `∇f` is the gradient of `f`. Where they disagree the line search accepts
steps that worsen the objective. **Fix:** keep the domain feasible via bounds and
constraints; never return a penalty from a smooth objective.

## 🟠 P5 — Piecewise-linear continuation value under gradient-based SLSQP

`create_interp` (**line 873**) uses `Gridded(Linear())`, so `V_{t+1}` is C⁰ but not C¹ and
`Interpolations.gradient` is piecewise-constant with jumps at every knot. SLSQP builds a
BFGS quadratic model from that. Same in both child modules
(`create_interpolator`, ar1:427 / ret:500).

Also **line 875**: `interp_vec = Vector{Any}` forces dynamic dispatch on `interp[j_p]`
inside the innermost objective — a large avoidable cost.

**Fix.** `Cubic(Line(OnGrid()))` or a shape-preserving spline; type the vector concretely.

## 🟡 P6 — NaN guard unreachable, and poisons the init chain

**Line 387:** the check sits in an `elseif` after `rt == "converged"` and `rt == "maxeval"`:

```julia
if rt == "converged" ... elseif rt == "maxeval" ... elseif any(isnan, x_opt) ...
```

NLopt routinely returns `:FTOL_REACHED` with a NaN iterate, in which case the guard never
fires. There is **no NaN check at all** for `t < T`, where `init = sol_c[t+1, …]` — so one
NaN propagates backward through the initial-guess chain and corrupts every earlier period.

**Line 517:** `other_dict[ret] = …` is only populated in the `t ≤ 7` loop, so
`print_period_stats` reports "Other: 0.0%" unconditionally for the full-model periods even
when SLSQP returns `:ROUNDOFF_LIMITED` or `:FAILURE`.

**Line 377/444/506:** `xtol_rel!(opt, 1e-4)` only; `ftol_rel` is never set for the parent.

## 🟡 P7 — Parameters that do not match the spec

- **Line 195:** `phi_2_0 = 20.0`, with `phi_1_0 = 1.0` and `phi_3_0 = 0.03` — sum 21.03.
  `model.txt` says `(φ₁,φ₂,φ₃)` are "normalized to sum to one". Under CRRA that
  normalization is not a free rescaling.
- **Line 291:** `Bernoulli(0.3)` — the `BothCollege` share is hardcoded and should be
  sourced to the estimation sample.
- **Line 756:** `return 2 * exp(log_wage) * …` — the `2 ×` makes it the household total,
  contradicting "wages are defined as the mean across the two parents".
- **Line 207:** `Np = 3` for `ρ = 0.9`. See C5.

## 🟡 P8 — Verify `Age` units in the wage equation

**Line 751:** `model.β_age * t` uses the model period `t ∈ 1..17`. This is correct **only
if** the Stata `Age` variable was normalized (e.g. `age − 25`).

Evidence favours normalized: the implied peak `β₂/(2|β₃|) = 0.0230108/0.0008638 ≈ 26.6`
reads as actual age **51.6** under that assumption (a textbook wage peak), versus **26.6**
under raw age (implausible). Confirm against the regression script — if raw, the parental
wage profile **flips sign** over the horizon (+27% vs −10%).

---

# `code/src/child_lifecycle_ret.jl` — LIVE

## 🟠 C1 — `-Inf` infeasibility sentinel is reachable

`compute_min_assets` (**line 513**) uses a hard-coded `c_min = 0.3` (**line 515**) to mark
college states infeasible, writing `-Inf` at **line 277**. But `asset_constraint_college`
(**line 436**) enforces only `a_next ≥ a_min = 0.01`.

At the current calibration `a_min_t = [3.354, 2.555, 1.732, 0.884]`. From the smallest
feasible `t=3` state (`a = 1.732`), the reachable set is `a_next ∈ [0.010, 1.173]` — which
straddles `a_min_t[4] = 0.884`. That region is then linearly interpolated by
`create_interpolator` (**line 500**), and `-Inf` propagates (interior points → `-Inf`;
`0 × -Inf` → `NaN`). SLSQP given NaN does not error — it returns the last iterate silently.

Also `sol_tr_v_work` (**line 715**) and `sol_tr_v_college` (**line 759**) are filled with
`-Inf`, then interpolated in `simulate_model_family!`.

**Fix.** Enforce `a_next ≥ a_min_t[t+1]` so no feasible point evaluates the infeasible
region, and drop the `-Inf` fill. Never linearly interpolate an array containing `-Inf`.

## 🟠 C2 — Psychic cost uses the wrong power

**Line 485:** `psychic_cost = model.kappa / (k + 1.0)^4`. `model.txt` specifies
`κ_X = κ/(HC+1)²`.

At `HC = 1, κ = 5`: model gives 1.25, code gives 0.31 — 4× too small and decaying twice as
fast in `HC`. This term generates the human-capital gradient in college enrolment, i.e.
the link from childhood investment to the college margin. One-character fix.

## 🟠 C3 — Retirement exists in the code but not in the model

**Line 83:** `t_retire::Int = 42`; **line 69:** `T::Int = 52`. `solve_model_work!`
(**line 169**) has a retirement block (**lines 174–220**) paying
`0.5 × after_tax_income(w_pre, h_avg)` (**lines 180, 194**), with `util_retire`
(**line 461**) and `pension_amount` (**line 539**).

`model.txt` §Environment: *"The model has no retirement stage and ends as the child becomes
68 years old."* Also `T = 52` from age 18 gives a terminal age of **69**, not 68.

A pension floor raises both `V^W` and `V^E` and shifts the college margin and the optimal
transfer. Either add retirement to the paper or include `child_lifecycle_ar1.jl` instead.

## 🟠 C4 — Asymmetric optimization across the discrete choice

| | work | college |
|---|---|---|
| initial guess | `tr_hi * 0.5` (**line 735**) | `tr_hi * 0.99` (**line 779**) |
| tolerance | `ftol_rel = 1e-8` (**line 731**) | `ftol_rel = 1e-6` (**line 775**) |

The college/work decision is made by comparing `sol_tr_v_college` against
`sol_tr_v_work` — two **locally** optimized values from different starting points at
different tolerances. With `κ log(a−tr)` against an interpolated `V^child(tr)`,
non-concavity is likely, so the discrete choice can flip for purely numerical reasons.

**Fix.** It is a 1-D problem — use golden-section or a grid search, or multistart both
branches identically.

## 🟠 C5 — Shock discretization too coarse

**Line 76:** `p_ar1 = 0.95, sigma_p = 0.2, Np = 5`; **line 98:** `tauchen(Np, p_ar1, sigma_p, 0.0, 3)`.
Parent: `Np = 3` for `ρ = 0.9` (parent_family.jl:207).

Tauchen with 5 states cannot represent `ρ = 0.95`; 3 states for `ρ = 0.9` is worse. Also
`p_grid = exp.(mc.state_values)` gives `E[exp(z)] = exp(σ_z²/2) ≠ 1` — a systematic upward
wage drift. `model.txt` says the shock is a **random walk** (ρ = 1), which Tauchen cannot
discretize at all (infinite unconditional variance).

**Fix.** Rouwenhorst with `N ≥ 7`; normalize `p_grid = exp.(z .- σ_z²/2)`; reconcile ρ with
the paper.

## 🟡 C6 — Stationary distribution in solve, median state in simulation

`optimal_transfer_work!` / `_college!` (**lines 699, 747**) take the expectation over the
AR(1) state using `stationary_dist(p_transition)` (**line 874**), but the simulation starts
every agent at the median state (**line 123**,
`sim_p_init_idx = fill(ceil(Int, Np/2), simN)`). The transfer policy is optimal for a
distribution the simulated child is not drawn from.

`stationary_dist` itself (**line 874**) uses `eigen(P')` + `argmax(real(vals))`, which is
fragile; solving `(I − P')π = 0` with a normalization is more robust.

## ⚪ C7 — `findfirst` can return `nothing`

**Lines 596, 945:** `eps_indices = [findfirst(w -> w ≥ rand(rng), cum_weights) …]`.
`t_weight` sums to 1 only up to floating-point error; if `cum_weights[end] = 1 − ε`, a
draw above it yields `nothing`, which then indexes an array.

---

# `code/src/child_lifecycle_ar1.jl` — NOT INCLUDED (latent)

Same lineage as `child_lifecycle_ret.jl`, without retirement and with a **flat** tax. Kept
for reference; the notebook does not include it. Everything below is latent unless you
switch to it.

**Note:** despite lacking retirement, this module is *closer* to `model.txt` on that one
point (C3 does not apply to it). If you adopt it, the flat tax (below) becomes the blocker
instead.

| Issue | Line(s) | Severity | Note |
|---|---|---|---|
| **Flat proportional tax, not progressive** — `wage_func` returns `(1 - model.tau) * base_wage * p_shock * 0.584`; no `after_tax_income`/`d_after_tax_dh` at all | **424** | 🟠 | `model.txt` specifies progressive `T(·)`. `child_lifecycle_ret.jl` has the HSV form; this file never got it. |
| **Psychic cost `^4`** (C2) | **417** | 🟠 | Model says `(HC+1)²`. Line 418 keeps a commented-out `κ·log(k)` alternative. |
| **`-Inf` sentinel reachable** (C1) | **227**, 439, 607, 651 | 🟠 | `c_min = 0.3` at 439 vs `a_min` enforced at 387. |
| **Asymmetric transfer optimization** (C4) | **627** (`0.5`) vs **671**/**807** (`0.99`) | 🟠 | Tolerances match here (all `1e-8`, lines 623/667/803), so only the initial guess is asymmetric. |
| **Coarse shocks** (C5) | **73** (`p_ar1=0.90, Np=5`), **92** | 🟠 | Plus `E[exp(z)] ≠ 1`. |
| **Piecewise-linear continuation value** (P5) | **427** | 🟠 | `Gridded(Linear())` + `Line()` extrapolation. |
| **Stationary vs median state** (C6) | **117**, 765 | 🟡 | `stationary_dist` at 765 uses `eigen(P')`. |
| **Magic `0.584` hardcoded, not a named constant** | **424**, 549, 982, 1126 | 🟡 | `child_lifecycle_ret.jl` uses `WAGE_SCALING_FACTOR` (ret:144). Here it is a bare literal in four places, and the sim divides it back out. |
| **Belief correction can go negative** (N6) | **1138** | 🟠 | `k + b* + 3(b* − b_m)`. The `3` is correctly `T_E − 1`, but with `b_m` up to 4.875 and `b* = 2.0` this gives `k − 6.6`. |
| **`findfirst` → `nothing`** (C7) | **500**, 919, 1081 | ⚪ | |
| **Duplicate `discrete_draw`** | **452** and **859** | ⚪ | Identical definitions in one file. |
| **`Nt` dimension replicated for shock-free arrays** | **97** | ⚪ | `sol_shape = (T, Na, Nk, Np, Nt)`; the work path never sees ε, so 10× the memory. At `t=1` the college optimizer also re-solves `Nt` times for the same `c` (ε is additive and cannot shift the argmax). |
| **Defaults differ from the live module** | **66** (`T=50`, `rho=1.0`), **75** (`Nt=11`) | ⚪ | `rho=1.0` selects the log branch of `util_work` (401). The notebook always passes `rho=1.5`. |

---

# `code/transfer_CRRA_wage.ipynb`

Notebook references are **cell N, line M** (line within that cell).

## 🔴 N1 — College choice taken outside the ε expectation

**11 sites:** cell 10 L1 · cell 38 L51 · cell 40 L72 · cell 53 L21 · cell 61 L21 ·
cell 69 L20 · cell 70 L21 · cell 71 L21 · cell 77 L72 · and 2 more in cells 78–79.

```julia
v_max = safe_maximum.(child_model.sol_exp_v_college, child_model.sol_tr_v_work)
```

`sol_exp_v_college` comes from `optimal_transfer_exp_college!` (ret:1055), which has
**already integrated ε** (`weight = π_p[ip] * t_weight[it]`). So this computes

  `max{ E_ε[V^E], V^W }`  instead of  `E_ε[ max{V^E(ε), V^W} ]`.

`model.txt` puts the max **inside**: `V^C = max{V^E(ε₀), V^W}`, with the transfer chosen
after uncertainty resolves. By Jensen this understates the option value of college and
biases take-up downward, most where the two alternatives are close — the margin that
identifies your parameters.

**And the code contradicts itself:** `simulate_model_family!` (ret:888) compares
`sol_tr_v_college_interp[it][ip]` against work **per agent, for that agent's ε** — the
correct rule. The backward induction was solved against a terminal value the simulated
child never faces.

**Fix.** The ε-specific building block already exists (`sol_tr_v_college[:,:,ip,it]`):

```julia
v_max = sum(t_weight[it] .* max.(sol_tr_v_college[:,:,:,it],
                                 sol_tr_v_work[:,:,:,1]) for it in 1:Nt)
```

and drop `optimal_transfer_exp_college!` from the terminal-value path — it implements a
*commitment* timing (parent picks `tr` before ε) that the prose rules out.

## 🟠 N2 — Convergence diagnostics discarded

**65 occurrences of `@suppress_output`**, including every counterfactual solve: cell 25
L10/L15, cell 28 L12–L239 (24 solves), cell 38 L25/L46/L70, cell 40 L46/L67, and more.

The macro redirects stdout to `/dev/null`, which swallows `print_period_stats`. For 30+
models you have no idea what share of 30×2×30×3 grid points × 17 periods converged.
Combined with P6 (`other_dict` never populated outside the `t ≤ 7` loop), non-converged
NLopt returns are accepted silently and written into the policy arrays.

**Fix.** Return the stats instead of printing; assert a minimum converged share and error
out below it.

## 🟠 N3 — CEV formula assumes homotheticity the value function lacks

**Cell 30, lines 44–52:**

```julia
delta_v_high = avg_welfare_tax_high / avg_welfare_baseline
lambda_high  = delta_v_high^(1 / (1 - model_baseline.rho)) - 1
```

`λ = (V_cf/V_base)^{1/(1−ρ)} − 1` is valid only if `V` is homogeneous of degree `(1−ρ)` in
consumption. The parent's value function is **not**: it contains `α̃₂ log HC`,
`α̃₁ log l_c`, `−φ₂h^{1+η}/(1+η)`, and a terminal value `ψ log HC + κ log a_term + ω V^C`.
Scaling consumption does not scale those terms, so the exponent does not recover a
consumption-equivalent.

Direction happens to come out right (both `V` are negative under `ρ = 1.5`, so an
improvement gives a ratio < 1 and `λ > 0`), but the magnitude has no interpretation.

Also **dead code**: `beta = 0.96`, `phi1 = 1`, `T = 17`, `sum_disc` (lines 39–43) are
computed and never used — leftovers from the log-utility version.

## 🟠 N4 — θ-experiment baseline uses a different ω than its arms

| cell | model | `omega` | `mu` (θ) |
|---|---|---|---|
| 69 | baseline | **0.35** | 0.5 (default) |
| 70 | "Decision by Parent" | 0.30 | 0.65 |
| 71 | "Decision by Child" | 0.30 | 0.40 |

The baseline differs from both treatments in the altruism weight as well as the bargaining
weight, so the college-decision counterfactual confounds ω with θ. Set ω identically
across all three arms.

Related, and unchanged from the earlier audit: because
`coef = (1−θ) + θω`, the weight on `V^C` at θ = 0.65, ω = 0.3 is `0.35 + 0.195 = 0.545`,
still comparable to the parent's 0.65. "Decision by Parent" overstates what the experiment
does — even at θ = 1 the child keeps weight ω.

## 🟠 N5 — `psi_terminal_belief_bin` computed and never used

**Cells 40, 77, 78, 79 — line 39:**

```julia
psi_terminal_belief_bin = psi_from_belief_linear.(college_boost_belief_bin)
```

The belief-specific child models are then constructed with `psi_terminal=1.0` hardcoded
(cell 40 L67 and the parallel cells). If the paper claims beliefs shift the terminal
human-capital weight, **that channel is silently off**.

**Fix.** Pass `psi_terminal = psi_terminal_belief_bin[m]`.

## 🟠 N6 — Belief correction can drive human capital negative

`k_next = k + college_boost_true + 3 * (college_boost_true - belief_values[m])`
(ar1:1138; same form in `simulate_model_family_hetero!`, parent_family.jl:1160).

The `3` is correctly `T_E − 1` per `model.txt`. But `college_boost_belief_bin` spans
`[0.125, 4.875]` around a truth of 2.0, so the top bin gives
`k + 2 + 3(2 − 4.875) = k − 6.6` → negative human capital, then `Line()`-extrapolated
policies below the grid and `w = w₀(1 + α·HC)` with negative `HC`.

**Fix.** Floor at `k_min`, or bound the belief grid so the correction cannot go negative.

## 🟡 N7 — Res-vs-Exp arms asymmetric

- Child model **cell 53 L2**: `y = 1.08`
- Parent model **cell 54 L16**: `y = 1.2`
- Low arm **cell 61 L21 / cell 55**: `y = 0.27` for both

Baseline `y = 0.6`. So the "high" arm is ×1.8 for the child but ×2.0 for the parent, while
the "low" arm is ×0.45 for both. Almost certainly unintended, and it breaks the symmetry
of the positive/negative comparison.

## 🟡 N8 — Model/label order swapped in one figure

**Cell 73, lines 13–17:**

```julia
plot_family_counterfactuals(
    [model_only_child, model_baseline],
    ["Baseline", "Decision by Child"];
```

`models[1] = model_only_child` receives `labels[1] = "Baseline"` — and `models[1]` is also
drawn solid/black as the reference series. Straight swap.

*Checked and NOT a problem:* the μ̃ arms in cells 25/27/32/34 are correctly labelled. The
arms are now polar cases (`mu_1=0` → μ̃ ≡ 1 "Parent only"; `mu_0=mu_1=0` → μ̃ = 0 after age
7 "Child only"), and both the "Low/High μ_t" and "Parent only/Child only" labellings match.

## 🟡 N9 — Tax counterfactual labels do not match the τ values used

**Cell 30, lines 35-37 and 48/53** print tax rates that are not the ones the models were
built with:

| | actual τ | printed |
|---|---|---|
| baseline (cell 54) | **0.18** (constructor default, parent_family.jl:183) | `τ=0.25` ❌ |
| `model_tax_benefit_high` (cell 28 L232) | **0.25** | `τ=0.35` ❌ |
| `model_tax_benefit_low` (cell 28 L237) | **0.10** | `τ=0.10` ✓ |

Stale labels carried over from `transfer_model_AR1.ipynb`, where the baseline was τ=0.25
and the high arm τ=0.35. In this notebook the experiment is **0.18 → 0.25**, a much
smaller change than the printed output claims. Any CEV quoted from this cell is attached
to the wrong tax rates. (The CEV itself is separately invalid — see N3.)

The `y` labels are correct: `y = 0.6 / 1.0 / 0.2` × 10³ = 6000 / 10000 / 2000.

Also **cell 28 line 217**: `phi_3_0 = 0.1  # Increased from 1.0` — the default is 0.03,
not 1.0.

## 🟡 M1 — Tables are never written to disk

The notebook contains **zero** `CSV.write`, `writedlm`, `tabpath`, or `tabdir` calls.
`output/tables/` is empty because nothing writes to it. The two results tables —
`belief_df` (cell 47) and the `@sprintf` belief summary (cell 43) — exist **only as
notebook output**, which is now stripped on commit.

Before the reorganization those tables lived in the committed notebook. They no longer do.
See "Output paths" below.

## ⚪ Other

- **`Spline2D(…; s=10.0)`** (cell 10 L11, and each `V_child_interp*`): a hand-tuned
  smoothing spline is applied to the child's value function before it becomes the parent's
  terminal condition. The residual budget is absolute, so the effective smoothing differs
  across counterfactual arms whose value functions differ in scale (e.g. `y = 0.27` vs
  `y = 1.2`). Cell 11 plots `s ∈ {0, 0.1, 1, 10}` and the largest was chosen.
- **`plot_simulation_results(…; save_dir::AbstractString = nothing)`** — the annotation
  rejects the default; latent `MethodError`, never fires only because every call site
  passes `save_dir`.
- **Execution-order sensitivity** is much reduced now that the model lives in `src/`, but
  `model_baseline`, `child_model`, and `V_child_interp1` are still reassigned across cells
  (e.g. cell 69 rebuilds `V_child_interp1`). Figures still depend on run order.

---

# 🟠 X1 — No accuracy diagnostics (all files)

There is no Euler-equation residual, no solve on a refined grid, no monotonicity or
concavity check on `V`, and no assertion that simulated states stay inside the grids.

For a structural paper this is the largest single gap. A single "are all simulated states
inside the solution domain?" assertion would have caught C1 and P3 immediately.

---

# Output paths — status

Verified after the 2026-08-02 reorganization.

## Figures ✅ correct

All 7 `savefig` call sites resolve under `output/figures/`, via `figdir(...)` from
`code/src/paths.jl`:

| cell | destination |
|---|---|
| 15 | `output/figures/Baseline/` |
| 18 | `output/figures/Baseline/PolicyFunctions/<stamp>/` |
| 19 | `output/figures/Baseline/<stamp>/` |
| 23 | `output/figures/Adulthood/<stamp>/` |
| 31 | `output/figures/Parameters/<stamp>/` |
| 33 | `output/figures/Parameters/terminal_assets/` |
| 48 | `output/figures/Adulthood/<stamp>/` |

All 30 former `joinpath(@__DIR__, "plots", …)` calls were replaced; **0** hardcoded path
literals remain. The `"Plots"` vs `"plots"` capitalization inconsistency is gone. All 7
sites create their directory **inside** the `if save` branch, so `save=false` no longer
leaves empty timestamped folders.

Note cell 23 and 48 write to `output/figures/Adulthood/`, which did not exist under the old
`plots/` tree — the old code wrote there too, so this is pre-existing, but those runs used
`save=false`. Consider `figdir("Baseline", "Adulthood")` for consistency with the rest.

## Tables ❌ nothing is written

`output/tables/` is empty and no code writes to it (M1 above). To fix, in the cell that
builds `belief_df`:

```julia
using CSV
CSV.write(joinpath(tabpath(), "belief_summary.csv"), belief_df)
```

`tabpath()` creates and returns `output/tables/`. This matters more now that notebook
outputs are stripped on commit.

## Data ⚠️ not used

`output/data/` exists and is git-ignored, but nothing writes to it. Solved models are not
cached, so any re-run re-solves from scratch (hours). `datapath()` is available.

## Provenance ⚠️ available, not wired in

`write_manifest` (`code/src/manifest.jl`) is loaded but **called nowhere** — there are 0
call sites. Until it is called, no figure records the commit or parameters that produced
it. Add one call per experiment batch:

```julia
write_manifest(figpath("Parameters"); experiment = "sigma counterfactuals",
                                      mu_1 = -0.04, rho = 1.5, Na = 30, simN = 5000)
```

---

## Suggested order

1. **P1** — two lines; every parental policy for `t ≤ 16` is currently wrong.
2. **P2** — until counterfactuals share random numbers you cannot tell whether any other
   fix changed anything.
3. **N1** — the ε-timing fix; changes every reported number.
4. **C1, P3, X1** — the domain and feasibility bugs, plus the assertion that would have
   caught them.
5. **N2, P4, P6** — optimizer hygiene: whether the reported policies are optima at all.
6. **C2, C3, N4, N5, N6, N7, N8** — specification and experiment-design corrections.
7. **C5, P5** — accuracy and speed, which then make X1 affordable.
8. **M1** + provenance — persist tables and wire in `write_manifest`.
