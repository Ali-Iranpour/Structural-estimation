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
| `code/src/child_lifecycle.jl` | **LIVE, canonical** — the child module the notebook includes. All child-side fixes go here. |
| `code/transfer_CRRA_wage.ipynb` | **LIVE** — driver |
| `code/src/child_lifecycle_ret.jl` | **SUPERSEDED** — reference only, kept until equivalence/integration tests pass. Do not fix. |
| `code/src/child_lifecycle_ar1.jl` | **SUPERSEDED** — reference only. Do not fix. |

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
| N10 | Heterogeneous high-resource arms solve work at `y=0.6`, college at `y=1.08` | notebook | 🔴 |
| N11 | Notebook cannot run top-to-bottom; two results tables read stale globals | notebook | 🔴 |
| ~~C1~~ | ~~`-Inf` sentinel → 30% NaN~~ — **FIXED** in `child_lifecycle.jl` | — | ✅ |
| C9 | Child simulation never clamps assets (live module only) | child_lifecycle_ret | 🟠 |
| ~~C10~~ | ~~Return codes never inspected~~ — **FIXED**, `check_nlopt!` errors | — | ✅ |
| C11 | Transfer/simulation extrapolation — **partly fixed** (domains aligned; `Line()` vs `Flat()` remains) | child_lifecycle | 🟡 |
| C2 | Psychic cost uses `^4`, model says `^2` — **OUT OF SCOPE by instruction** | child_lifecycle | ⏸️ |
| ~~C3~~ | ~~Retirement not in the model~~ — **FIXED**; notebook switched | — | ✅ |
| N2 | 65 × `@suppress_output` discards all convergence diagnostics | notebook | 🟠 |
| P3 | Simulated states never clamped; artificial `a ≤ a_max` in the solve | parent_family | 🟠 |
| ~~C4~~ | ~~Asymmetric transfer optimization~~ — **FIXED**, branch-symmetric `maximize_1d` | — | ✅ |
| ~~C5~~ | ~~Shock discretization~~ — **RESOLVED: documented approximation** (Phase 0.7) | — | ✅ |
| P4 | Objective/gradient inconsistent in `-1e8` penalty branches | parent_family | 🟠 |
| P5 | Piecewise-linear continuation value under gradient-based SLSQP | all | 🟠 |
| N3 | CEV formula assumes homotheticity the value function lacks | notebook | 🟠 |
| N4 | θ-experiment baseline uses a different ω than its treatment arms | notebook | 🟠 |
| ~~N5~~ | ~~`psi_terminal_belief_bin` unused~~ — **NOT AN ERROR, deliberate choice** | — | ⚪ |
| ~~N6~~ | ~~Belief correction can drive HC negative~~ — **WITHDRAWN, was wrong** | — | ⚪ |
| ~~N7~~ | ~~Res-vs-Exp arms asymmetric~~ — **NOT AN ERROR, deliberate choice** | — | ⚪ |
| N8 | Model/label order swapped in one figure | notebook | 🟡 |
| N9 | Tax counterfactual labels do not match the τ values used | notebook | 🟡 |
| ~~N12~~ | ~~`ā^P` placeholder~~ — **FIXED**: `delta_P = c_floor = 0.01` (Phase 0.5b) | — | ✅ |
| P6 | NaN guard unreachable; poisons the backward init chain | parent_family | 🟡 |
| P7 | φ weights not normalized; `BothCollege` share hardcoded (`2×` **resolved**) | parent_family | 🟡 |
| ~~P8~~ | ~~Verify `Age` units~~ — **RESOLVED, code correct** | — | ⚪ |
| M1 | Tables are never written to disk | notebook | 🟡 |
| C6 | `stationary_dist` in solve vs median state in simulation — **required by the N1 timing** | both child modules | 🟠 |
| C7 | `findfirst` can return `nothing` | both child modules | ⚪ |
| C8 | Duplicate `discrete_draw`; unused `Nt` dimension | child_lifecycle_ar1 | ⚪ |
| C12 | `sim_a[:, T+1]` never written — child terminal assets are all NaN | both child modules | ⚪ |
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

Since `k_grid = range(0.0, 1.0, length=2)`, the k-direction gradient is the level gap
`V(BothCollege=1) − V(BothCollege=0)`. Stated plainly, the gradient tells the optimizer
that **working more makes you college-educated**: with `h_p ∈ [0,1]`, it credits full-time
work with the entire lifetime value of converting a non-college household into a college one.

### Measured effect (verified 2026-08-02)

Gradient decomposition of `∂f/∂h_p` at `t=12`:

| state (a, BC, HC) | `∂U/∂h` | `β·∂V/∂a·wage` | **`β·∂V/∂k` (bug)** | total (buggy) | total (correct) |
|---|---|---|---|---|---|
| (1.31, 0, 1.53) | −3.851 | +2.483 | **+2.2** | +0.8 | −1.369 |
| (13.45, 0, 1.53) | −3.273 | +1.933 | **+1.5** | +0.2 | −1.340 |
| (42.06, 1, 1.53) | −1.682 | +1.312 | **+0.7** | +0.3 | −0.370 |

The bug term is the *same order of magnitude* as the legitimate terms, not overwhelming —
but it **flips the sign of the total**, so the optimizer is told "work more" where the
truth is "work less".

A/B solve, identical model, only these two lines changed:

| policy | buggy | fixed | bias |
|---|---|---|---|
| `h_p` labor supply | 0.3577 | 0.2894 | **+23.6%** |
| `c_p` consumption | 3.7180 | 4.3741 | **−15.0%** |
| `e_p` education spending | 0.3722 | 0.2280 | **+63.3%** |
| `τ_p` parental care time | 0.1542 | 0.3895 | **−60.4%** |

**The worst damage is not labor supply.** `e_p` and `τ_p` are the two human-capital
investment inputs (`σ₂ log e_p`, `σ₁ log τ_p`). The bug tilts the entire
time-versus-money investment margin — the mechanism the paper is about. Plausible chain,
measured but not itself verified: the parent works more, has less time for the child, and
substitutes purchased inputs for parental time.

Bias by period, mean `h_p`:

```
  t     buggy    fixed     diff
   1    0.3443   0.2396   +0.1047
   7    0.3482   0.2622   +0.0860
  12    0.3522   0.2978   +0.0544
  16    0.3929   0.3684   +0.0245
  17    0.4180   0.4180   +0.0000   <- terminal period identical
```

`t=17` is **exactly identical** because `obj_last_period_full` never had the term — its
`grad[4] = dutil_dh_p + β*(dV_da*marginal)` is already correct. One function being right
while the other two are wrong confirms this is an incomplete edit from when `k` was
parental human capital, not a deliberate choice.

The bias **grows going backwards** (+0.02 at `t=16` → +0.10 at `t=1`): each period's wrong
policy produces a wrong `V_t`, which becomes the next iteration's continuation value, so
the error compounds through the backward induction. The childhood periods are worst hit.

*Caveat:* measured on a coarse grid (`Na=10, Nhc=10`) with a stand-in terminal value
`1·log(HC) + 5·log(a)` shaped like the real one, because a full solve takes hours.
Direction and rough magnitude should hold; exact percentages will differ on the real
30×2×30×3 grid. Re-run the comparison after fixing.

An earlier version of this document claimed the bug drives `h_p` to its upper bound. It
does not — `h_p` stays interior (max 0.618 both with and without the bug). The effect is a
shifted interior optimum.

**Fix.** Delete `+ dV_dk_sum` from both gradients. The accumulator itself can stay, though
removing it saves a gradient call per shock state.

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

Strictly this is a **variance and reproducibility** problem, not a bias one — unseeded
draws do not shift `E[·]`. It stays Critical because every result in the paper is a
*difference between arms*, and without common random numbers you cannot tell whether a
plotted gap is a treatment effect or a reshuffle; nor can any number be reproduced.

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

**Lines 387-389:** when the guard *does* fire it sets `x_opt = init` but **does not
recompute `minf`**. Line 396 then stores `model.sol_v[...] = -minf` — the value from the
NaN solution — alongside a policy taken from `init`. Policy and value are inconsistent at
every such point.

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
- **Line 756:** `return 2 * exp(log_wage) * …` — **RESOLVED, intentional.** The Stata
  regression is on `Wage_Mean = (Wage_Mother + Wage_Father)/2` (wage2_styled.do:107), so
  `2 × mean` is household earnings when both parents supply `h_p` hours. Documentation
  fix only: `model.txt` should state that `h_p` is hours *per parent* and that household
  labor income is `2 · w̄ · h_p`.
- **Line 207:** `Np = 3` for `ρ = 0.9`. See C5.

## ⚪ ~~P8~~ — RESOLVED: the `Age` units are correct

Verified against `data_cleaning/Initial_Distributions/Code/wage2_styled.do`:

```stata
global Age_S_model 26 ;                          /* line 73  */
replace Age = Age - $Age_S_model + 1 ;           /* line 132 */
reg Log_Wage Both_College Age Age_2 Age_2_Edu Age_Edu, vce(robust) ;   /* line 320 */
```

`Age` in the regression **is** re-indexed so biological age 26 → model period 1. Therefore
`model.β_age * t` with `t ∈ 1..17` (parent_family.jl:751) is exactly right, and the
regression at line 320 matches `wage_func` term for term (constant + 5 regressors).

The implied peak `β₂/(2|β₃|) ≈ 26.6` is in model periods, i.e. **biological age 51.6** — a
textbook wage peak, confirming the normalization.

**No code change.** Documentation only: `model.txt` eq. (wage_log) writes `Age_{it}`
unqualified and should state that it is model time, `= biological age − 25`.


---

# `code/src/child_lifecycle_ret.jl` — LIVE

## 🔴 C1 — `-Inf` infeasibility sentinel is reachable → 30% NaN downstream

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

### Measured contamination (verified 2026-08-02)

Solved on `Na=20, Nk=20, Nt=6` with the notebook's parameters, then counted:

| array | % NaN | % Inf |
|---|---|---|
| `sol_v_college` | 0.00 | 1.35 |
| `sol_c_college` | 1.35 | 0.00 |
| `sol_tr_college` | **5.00** | 0.00 |
| `sol_tr_v_college` | **5.00** | 25.00 |
| `sol_exp_college` | **30.00** | 0.00 |
| `sol_exp_v_college` | **30.00** | 0.00 |
| all `*_work` arrays | 0.00 | 0.00 |

`sol_exp_v_college` is **exactly the array the notebook feeds into `V_child_interp`**
(`v_max = safe_maximum.(sol_exp_v_college, sol_tr_v_work)`), and it is **30% NaN**.

Root cause: `optimal_transfer_exp_college!` guards on `assets ≤ 1e-3`
(child_lifecycle_ret.jl:1069) instead of `assets ≤ a_min_t[1] = 3.354` as
`optimal_transfer_college!` does (line 757). For assets between those bounds it *does*
attempt the optimization, and the objective evaluates `V1_college` over the `-Inf` region,
so NLopt returns NaN — which is then stored unchecked (see C10).

**`safe_maximum` masks this.** Where `sol_exp_v_college` is NaN it silently returns the
*work* value, so for ~30% of the `(a, HC)` grid the parent's terminal value is the work
value regardless of whether college would be better. That substitution happens for
numerical reasons, not economic ones, and produces no warning. This is why C1 is Critical,
not High: it is not a latent risk, it is contaminating the live terminal value.

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

## 🟠 C9 — Child simulation never clamps assets (live module only)

**child_lifecycle_ret.jl:672** (`simulate_model_child!`) and **:1029**
(`simulate_model_family!`):

```julia
sim_a[i, t+1] = a_next        # no floor, no cap
```

The **not-used** `child_lifecycle_ar1.jl` does clamp, at :567 and :1000:

```julia
sim_a[i, t+1] = max(a_next, a_min)
```

So the live module *lost* a guard the older one has. Combined with `Flat()` policy
extrapolation, simulated children can leave `[a_min, a_max] = [0.01, 50]` and then run on
constant boundary policies for the rest of the lifecycle. The adult Bellman has no upper
bound on `a_next` either, so from `a = a_max` any feasible saving choice immediately exits
the grid.

`sim_k` is never clamped in either module: `k_next = k + h` over 52 periods, and the
belief correction at N6 can drive it negative.

**Fix.** Clamp to the solution domain in both simulators, and assert that the share of
clamped observations is negligible — if it is not, the grid is too small.

## 🟠 C10 — Child solvers never inspect NLopt return codes

`ret` is captured at **7 sites** — child_lifecycle_ret.jl:214, 246, 296, 314, 736, 780,
1089 — and **never examined**. There is no `result_type_name`, no convergence tally, no
finiteness check. Whatever NLopt last evaluated is written straight into the solution
arrays.

The parent solver at least counts return codes (however imperfectly, see P6); the child
solver does not look at all. The measured NaN shares under C1 are the direct consequence.

**Fix.** Check `ret` and `isfinite(minf)` at every site; count and report; error out if the
converged share falls below a threshold.

## 🟠 C11 — Transfer solve and simulation use different extrapolation

The transfer optimization builds its child-value interpolants with **`Line()`**
(child_lifecycle_ret.jl:704, 751, 1060) and searches `tr` down to `1e-6`
(work, :729) or `1e-12` (college, :773, :1082) — far below `a_min = 0.01`. So the value
that *selects* the transfer is a linear extrapolation off the bottom of the grid.

Every simulation interpolant uses **`Flat()`** instead (:558–:938). The policy actually
executed at a given transfer is therefore not the policy whose value justified that
transfer.

**Fix.** Use one extrapolation convention throughout, and bound `tr` below by `a_min` so
the transfer problem never evaluates outside the child's solution domain.

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

## ✅ ~~C5~~ — RESOLVED: documented approximation

**Phase 0.7 decision (2026-08-05): keep the stationary AR(1); document it.**

`wage2_styled.do` §3 estimates `u_t = eps_t (transitory) + sum iota (permanent random
walk)`, with `σ_ε = 0.1335`, `σ_ι = 0.1893`, initial-shock variance `0.2357`. The model
implements a single stationary AR(1) (`ρ = 0.95`, `σ_p = 0.2`, Tauchen) — no transitory
component, persistent part stationary rather than a unit root. `σ_p ≈ σ_ι`, so it
approximates the permanent component and drops the transitory one.

This is now a stated approximation rather than a defect. The paper must say so — the
current text claims a random walk, which is neither implemented nor Tauchen-discretizable.
Exact wording in [`SPEC_DECISIONS.md`](SPEC_DECISIONS.md).

### Superseded detail — Shock discretization too coarse

**Line 76:** `p_ar1 = 0.95, sigma_p = 0.2, Np = 5`; **line 98:** `tauchen(Np, p_ar1, sigma_p, 0.0, 3)`.
Parent: `Np = 3` for `ρ = 0.9` (parent_family.jl:207).

Tauchen with 5 states cannot represent `ρ = 0.95`; 3 states for `ρ = 0.9` is worse. Also
`p_grid = exp.(mc.state_values)` gives `E[exp(z)] = exp(σ_z²/2) ≠ 1` — a systematic upward
wage drift. `model.txt` says the shock is a **random walk** (ρ = 1), which Tauchen cannot
discretize at all (infinite unconditional variance).

**Fix.** Rouwenhorst with `N ≥ 7`; normalize `p_grid = exp.(z .- σ_z²/2)`; reconcile ρ with
the paper.

## 🟠 C6 — Stationary distribution in solve, median state in simulation

`optimal_transfer_work!` / `_college!` (**lines 699, 747**) take the expectation over the
AR(1) state using `stationary_dist(p_transition)` (**line 874**), but the simulation starts
every agent at the median state (**line 123**,
`sim_p_init_idx = fill(ceil(Int, Np/2), simN)`). The transfer policy is optimal for a
distribution the simulated child is not drawn from.

`stationary_dist` itself (**line 874**) uses `eigen(P')` + `argmax(real(vals))`, which is
fragile; solving `(I − P')π = 0` with a normalization is more robust.

**Raised to High by the N1 timing decision.** Under "enrolment and transfer depend on `ε₀`
but not realized `z₀`", the transfer is optimal *against a distribution of `z₀`*. That is
only coherent if the simulation draws `z₀` from that same distribution. Either draw the
initial child shock from `stationary_dist`, or — if `z₀` is meant to be observed at
separation — make the transfer policy condition on it, which changes the timing again.
Pick one; the current combination is neither.

## ⚪ C12 — `sim_a[:, T+1]` is never written

Both simulators guard the transition on `if t < T`, so `sim_a` is filled for columns
`1..T` while the array is allocated with `T+1` columns. The final column stays `NaN`
(measured: exactly `1/(T+1)` of the array). The child consumes everything at `T`, so the
economically correct value is `0`, but `sim_a[:, end]` currently returns `NaN` to anything
that reads it.

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

## 🔴 N1 — Information timing: ε integrated in the wrong place

### The convention (DECIDED 2026-08-05)

The family observes the preference shock `ε₀` at the separation half-period, but the
child's initial wage shock `z₀` is still unresolved. Enrolment and the transfer may
therefore condition on `ε₀` but **not** on realized `z₀`:

```
Half period T_L        V^CD(a,HC,ε₀,m) =  max      E_{z₀}[ (1-θ)V₁^d + θV^{P,d} ]
                                        d∈{E,W}
                                        0≤tr≤a-ā^P

Last parent period     continuation   =  E_{ε₀}[ V^CD(a',HC',ε₀,m) ]
t = 17
```

so the object the parent's `t=17` problem needs is

```
        E_{ε₀} [  max_{d,tr}  E_{z₀} [ W_d(tr; ε₀, z₀) ] ]
```

**The nesting is the whole point.** It is *not* `max_{d,tr} E_{ε₀,z₀}[W_d]`, which would
select the transfer before the preference shock is observed.

Expanding with the parental specification (`V^P = ψ log HC + κ log a_term + ω V^C`), and
noting that `V^{P,d}` depends on `z₀` only through the altruism term:

```
V^CD = max_{d,tr} { [(1-θ) + θω]·E_{z₀}[V₁^d] + θψ log HC + θκ log(a-tr) }
```

which matches `coef = (1-mu) + mu*omega` in `obj_transfer_work` / `obj_transfer_college`,
with `terminal_value` supplying the other two terms.

### What the code does instead

**11 sites:** cell 10 L1 · cell 38 L51 · cell 40 L72 · cell 53 L21 · cell 61 L21 ·
cell 69 L20 · cell 70 L21 · cell 71 L21 · cell 77 L72 · and 2 more in cells 78–79.

```julia
v_max = safe_maximum.(child_model.sol_exp_v_college, child_model.sol_tr_v_work)
```

`sol_exp_v_college` comes from `optimal_transfer_exp_college!` (ret:1055), which weights by
`π_p[ip] * t_weight[it]` — it integrates ε **inside** the `max` over `tr`.

**The defect is asymmetric.** Only the college branch is wrong:

| branch | code computes | should be |
|---|---|---|
| work | `max_tr E_{z₀}[W_W]` | same ✅ — `W_W` has no ε |
| college | `max_tr E_{ε,z₀}[W_E]` ❌ | `max_tr E_{z₀}[W_E \| ε]` |
| across `d` | `max` applied to the ε-averaged college value ❌ | `E_ε` applied after the `max` |

Two separate errors compound: the transfer is chosen under commitment, and the discrete
`max` sits on the wrong side of `E_ε`. By Jensen the second understates the option value of
college, most where the alternatives are close — the margin that identifies the parameters.

A consequence: `values_grid = v_max[:, :, p_mid, t_mid]` is a single surface with **no ε
dimension**, so the parent never sees an ε-varying continuation to take an expectation over.

**And the code contradicts itself:** `simulate_model_family!` (ret:888) compares
`sol_tr_v_college_interp[it][ip]` against work **per agent, for that agent's ε** — the
correct rule under the adopted convention. The backward induction was solved against a
terminal value the simulated child never faces.

### Fix

The ε-specific building block already exists — `optimal_transfer_college!` loops over `it`
and stores `sol_tr_v_college[ia,ik,:,it] = max_tr E_{z₀}[W_E | ε_it]`, which is exactly the
inner object. Nothing new is needed:

```julia
v_max = sum(t_weight[it] .* max.(sol_tr_v_college[:,:,:,it],
                                 sol_tr_v_work[:,:,:,1]) for it in 1:Nt)
```

`max.(·,·)` is the `max_d`; `Σ_it t_weight[it]·(·)` is `E_{ε₀}`. Then **delete
`optimal_transfer_exp_college!`** — it implements the rejected commitment timing and is the
source of the 30% NaN in C1 (one deletion closes two findings).

The work branch needs no change. The forward simulator needs no change.

### Consistency requirements this convention imposes

- **C6 is no longer optional.** "Depends on `ε₀` but not realized `z₀`" only holds if the
  simulation draws `z₀` from the same distribution the transfer integrated over. The solve
  uses `stationary_dist`; the simulation puts every child at the median state.
- **N12 (`ā^P`) must be given a real value** — the transfer bound is part of the stated
  problem, not a numerical guard.

## 🟡 N12 — `ā^P`, the minimum retained parental asset, is a spec addition set to `1e-9`

The adopted half-period problem constrains `0 ≤ tr ≤ a − ā^P`. `model.txt` currently writes
`0 ≤ tr ≤ a_{T_L}`, i.e. `ā^P = 0`, which makes `κ log(a_term)` unbounded below as
`tr → a`.

The code effectively sets `ā^P = 1e-9`:

```julia
tr_hi = assets - 1e-9        # child_lifecycle_ret.jl:712, :756
lower_bounds!(opt, [1e-6])   # work,    :729
lower_bounds!(opt, [1e-12])  # college, :773, :1082
```

so `κ log(a_term)` can reach `9 · log(1e-9) ≈ -186`. That is a numerical guard standing in
for an economic object.

**Fix.** Choose `ā^P` on economic grounds (the parent's late-life consumption floor —
which is what the `κ_term` footnote already describes), state it in `model.txt`, and use it
as the bound. Note the interaction with **C4**: the college branch currently initializes at
`0.99·tr_hi`, i.e. hard against this boundary, while the work branch starts at `0.5·tr_hi`.

## 🔴 N10 — Heterogeneous high-resource arms solve the work path at the wrong `y`

**Cells 77 and 79.** `base_child` is built with **no `y`**, so it takes the default
`y = 0.6`:

```julia
base_child = ConSavLaborCollege_AR1(Na=50, Nk=50, Nt=10, sigma_eps=0.5, rho=1.5,
    psi_terminal=1.0, kappa_terminal=5.0, omega=0.3, a_max=50.0, w=20.0)   # y = 0.6
solve_model_work!(base_child); optimal_transfer_work!(base_child)
```

Each belief-specific `child_model` is then built with **`y = 1.08`** — but its entire work
solution is assigned from the `y = 0.6` object:

```julia
child_model = ConSavLaborCollege_AR1(..., college_boost=..., y=1.08)
child_model.sol_c_work    = base_child.sol_c_work       # solved at y = 0.6
child_model.sol_h_work    = base_child.sol_h_work
child_model.sol_v_work    = base_child.sol_v_work
child_model.sol_tr_v_work = base_child.sol_tr_v_work
child_model.sol_tr_work   = base_child.sol_tr_work
```

Consequences within a single agent's problem:

- **College path:** `y = 1.08` for the 4 college years, then `y = 0.6` for ages 22-68,
  because `solve_model_college!` copies the work arrays for `t > t_college`.
- **Work path:** `y = 0.6` throughout.
- The discrete comparison `sol_tr_v_college` (partly `y = 1.08`) against `sol_tr_v_work`
  (`y = 0.6`) gives college a resource bonus the work alternative never receives, so
  enrolment is **mechanically tilted toward college**.

Cells 38, 40 and 78 are **not** affected — there `child_model` takes the default `y`, so it
matches `base_child`. The bug is specific to the two cells where `y = 1.08` was added to
`child_model` and not to `base_child`.

This also contradicts the non-heterogeneous resource experiment (cell 53), which correctly
re-solves the whole child problem — work path included — at the raised `y`.

**Fix.** Pass the same `y` to `base_child`, or drop the copying and solve the work path per
arm. Note the copy is by reference, not `deepcopy`: the belief models share one array
object. Harmless today because nothing writes to them afterwards, but fragile.

## 🔴 N11 — The notebook cannot run top-to-bottom

Static check of first-assignment vs first-use across cells in order:

| variable | first assigned | first used | |
|---|---|---|---|
| `batch_dir` | cell 27 | **cell 26** | used one cell early |
| `model_parent_hetro` | **never** | cell 43 | |
| `final_assets` | **never** | cells 43, 50 | |
| `final_hc` | **never** | cells 43, 50 | |

Cell 42 assigns `final_assets_het`, `final_hc_het`, `belief_type_het`. Cells 43 and 50 read
`final_assets`, `final_hc`, `model_parent_hetro` — a rename that was never propagated.
Those names exist nowhere in this notebook; they are leftovers from
`transfer_model_AR1.ipynb`.

On a clean kernel these three cells raise `UndefVarError`. They only ever succeeded because
the notebook was run out of order with stale globals in scope — which means **the belief
summary table (cell 43) and the belief DataFrame (cell 50) were computed from variables
belonging to a different model run.** Both are results tables.

**Fix.** Rename the uses to `final_assets_het` / `final_hc_het` / the correct model object,
move `batch_dir` above cell 26, then run the notebook on a fresh kernel top to bottom and
confirm it completes. Until it does, no stored output can be trusted.

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

## ⚪ ~~N5~~ — NOT AN ERROR: deliberate modelling choice

**Confirmed by the author (2026-08-05) as intended, not a defect.** Beliefs shift the
perceived college boost only; the terminal human-capital weight `ψ_term` is deliberately
held common across belief types. Retained here so the choice is on the record rather than
looking like an oversight to a future reader.


**Cells 40, 77, 78, 79 — line 39:**

```julia
psi_terminal_belief_bin = psi_from_belief_linear.(college_boost_belief_bin)
```

The belief-specific child models are then constructed with `psi_terminal=1.0` hardcoded
(cell 40 L67 and the parallel cells). If the paper claims beliefs shift the terminal
human-capital weight, **that channel is silently off**.

**Fix.** Pass `psi_terminal = psi_terminal_belief_bin[m]`.

## ⚪ ~~N6~~ — WITHDRAWN (the belief correction is correct)

**This finding was wrong and is retracted.** The graduation correction cancels exactly:

```
k_1..k_3 :  k₀ + b_m,  k₀ + 2b_m,  k₀ + 3b_m        (t < t_college)
k_4      :  k₃ + b* + 3(b* − b_m)
         =  k₀ + 3b_m + b* + 3b* − 3b_m
         =  k₀ + 4b*                                  independent of b_m
```

Verified numerically for `b_m ∈ {0.125, 1.0, 2.0, 3.5, 4.875}` with `b* = 2.0`: every
belief lands on `k₀ + 8.0`, and no intermediate value is negative. The original claim
applied the correction to `k₀` instead of to `k_{t_college}`, which already carries three
accumulated `b_m` increments.

This matches `model.txt` eq. (5), `HC = H̃C_{t_c} + b* + (T_E−1)(b* − b_m)`, and the code's
`3` is correctly `T_E − 1`.

**One residual ⚪ item:** the multiplier is hardcoded as `3` rather than `t_college - 1`
(child_lifecycle_ar1.jl:1138, parent_family.jl:1160). Correct at `t_college = 4`; silently
wrong if the college length ever changes.


## ⚪ ~~N7~~ — NOT AN ERROR: deliberate modelling choice

**Confirmed by the author (2026-08-05) as intended, not a defect.** The child's `y = 1.08`
and the parent's `y = 1.2` in the high-resource arm are chosen separately on purpose. Worth
stating explicitly in the paper so the asymmetry reads as a design choice.


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

## Fixed on branch `fix/remove-retirement` (2026-08-05)

`code/src/child_lifecycle.jl` is now the canonical child module and the notebook includes
it. `_ret.jl` and `_ar1.jl` are reference-only — **all child-side fixes go to the canonical
module**.

| item | what changed |
|---|---|
| **C3** | Retirement removed. `simulate_model_family_hetero!` in `parent_family.jl` also stripped of `t_retire`/`pension_amount`/`interp_c_retire` — it was a hard blocker, since it `@unpack`ed a field the new module does not have. |
| **C1** | College feasibility rebuilt. `compute_min_assets` now uses `c_floor`, **the same floor the optimizer enforces** (was `c_min = 0.3` against a `0.01` bound), and returns `a_req[1..t_college+1]` with `a_req[t_college+1] = a_min`. `asset_constraint_college` now imposes `a_{t+1} ≥ a_req[t+1]`, so a student who enters can always finish. |
| **C1 / item 5** | **No `-Inf` is ever stored.** Infeasible cells hold `NaN` ("not computed"); every college interpolant — value, policy, and transfer — is built over the feasible slice of the asset grid via `first_feasible_a` / `first_feasible_parent_a`, so `NaN` never enters an interpolation. `-Inf` is applied only at the discrete comparison, in both simulators. Verified: **0% `Inf` in every stored array.** |
| **C10** | `check_nlopt!` added at every solver site: rejects `:FAILURE`/`:INVALID_ARGS`/`:OUT_OF_MEMORY`/`:FORCED_STOP` and any non-finite `minf` or iterate. Numerical failure now **errors**; economic infeasibility is a separate, silent mask. |
| **C4 / item 8** | `maximize_1d` — coarse grid sweep plus two local refinements, derivative-free, **identical grid and tolerances in both branches**. Replaces SLSQP, which was started at `0.5·tr_hi` / `1e-8` for work and `0.99·tr_hi` / `1e-6` for college. |
| **N12 / item 4** | Transfer domains are now explicit: work `tr ∈ [0, max(0, a − δ_P)]` (never infeasible; collapses to `{0}` when the parent holds less than `δ_P`), college `tr ∈ [a_req[1], a − δ_P]`. **When `a − δ_P < a_req[1]` the college optimizer is not called** — the branch is declared infeasible. `δ_P` is a named field (`delta_P = 0.05`) instead of an implicit `1e-9`. |

### Verified (Na=20, Nk=20, Nt=6)

```
array                  %NaN    %Inf    note
sol_v_work             0.00    0.00
sol_c_work             0.00    0.00
sol_tr_v_work          0.00    0.00    work branch never infeasible
sol_v_college          1.08    0.00    NaN = infeasible
sol_tr_v_college      20.00    0.00    NaN = infeasible (4 of 20 asset points)
sol_exp_v_college     20.00    0.00
ANY Inf anywhere?     false
```

`a_req = [2.2763, 1.7346, 1.1766, 0.6019, 0.01]`; minimum parental assets for college
`= 2.3263`; 16 of 20 grid points feasible. Work transfers hit exactly `0` at 66.2% of
states — previously unreachable, since the lower bound was `1e-6`. College transfers are
bounded below by `a_req[1]` as intended. The solve completes without `check_nlopt!`
firing, and `simulate_model_family_hetero!` runs end to end.

### Still open

- **`δ_P = 0.05` is a placeholder.** N12 asks for an economic value; this is not one.
- **The child asset grid still starts at `a_min = 0.01`, not `0`.** The work branch can now
  choose `tr = 0`, but the child's own grid does not contain it, so the first child period
  is evaluated by extrapolation. Including `0` in `a_grid` is the clean fix.
- **C11 partly stands:** the transfer solve still extrapolates `Line()` while the
  simulators use `Flat()`. The *domains* now agree; the extrapolation convention does not.
- **Family college share is 0%** in the standalone test. This predates these changes — it
  was already 0% immediately after the retirement removal, before items 3/4/5/8 — and the
  notebook overwrites the initial conditions from the parent solve, so the standalone
  number is not diagnostic. Worth re-checking after a real run.

---

## Fix roadmap

Each phase makes the next one verifiable. **Do not re-run counterfactuals for the paper
until Phase 3 is complete.**

---

### Phase 0 — Freeze the specification  *(decisions, not code)*

Every item below determines *which* code is worth repairing. Settle them first, record the
answer in `model.txt`, then write code once.

| # | Decision | Status |
|---|---|---|
| 0.1 | **N6 — withdrawn.** The belief correction cancels to `k₀ + 4b*`. | ✅ closed |
| 0.2 | **P8 — resolved.** Stata `Age` is re-indexed to model time; `β_age * t` is correct. | ✅ closed |
| 0.3 | **P7 (wage half) — resolved.** `2 × mean parental wage` = household earnings. | ✅ closed |
| 0.4 | **C3 — retirement removed.** `child_lifecycle.jl` canonical; notebook switched. | ✅ done |
| 0.5 | **N1 — ε observed before the transfer; `E_ε` outermost.** | ✅ decided → 3.3 |
| 0.5b | **N12 — `δ_P = c_floor = 0.01`.** | ✅ done |
| 0.5c | **C6 — `z₀` drawn from the stationary distribution.** | ✅ decided → 3.3b |
| 0.6 | **Child horizon `T = 51`** (ages 18–68 inclusive). | ✅ done |
| 0.7 | **C5 — keep the stationary AR(1) as a documented approximation.** | ✅ decided |
| 0.8 | **P7b — drop the φ normalization claim** (`φ₂` is a scale, not a share). | ✅ decided |
| 0.9 | **College length: four years**, ages 18–21. Code right, paper display off by one. | ✅ decided |
| — | **N5, N7 — deliberate choices, not errors.** | ✅ closed |
| — | **C2 — out of scope by instruction.** | ⏸️ deferred |

All Phase 0 decisions are frozen in [`SPEC_DECISIONS.md`](SPEC_DECISIONS.md), which also
lists the eight `model.txt` edits they imply. Those are paper prose and are left for the
author to apply.

#### 0.4 — Retirement: build from `child_lifecycle_ret.jl`, do not start from `ar1`

`model.txt` says there is no retirement stage, so retirement comes out. But the *starting
point* matters, and the intuitive choice is the expensive one:

| feature | `_ret.jl` | `_ar1.jl` |
|---|---|---|
| `after_tax_income` (progressive tax) | 8 refs | **0** |
| `d_after_tax_dh` (its derivative) | 3 refs | **0** |
| `WAGE_SCALING_FACTOR` named const | 6 refs | **0** (four bare `0.584` literals) |
| `tax_lambda` field | 5 refs | **0** |
| `sim_a` sized `T+1` | ✅ | ❌ (`T`) |
| `max(a_next, a_min)` clamp | **0** | 2 sites |
| retirement block | ~61 lines | none |

Starting from `_ar1.jl` means re-implementing the entire progressive-tax system across ~22
reference sites — and `_ar1.jl` still has the flat `(1-τ)` baked into `wage_func:424`, which
is a *specification* error, not just a missing feature.

**Status: done, not switched.** `code/src/child_lifecycle.jl` exists on branch
`fix/remove-retirement` (1136 → 1056 lines). Verified: solves with 0% NaN/Inf in every work
array, matching `_ret.jl`'s contamination profile exactly; both simulators run; the new
`obj_last_period` gradient agrees with finite differences to ~1e-10. `T` default 51 (see
0.6). **The notebook still includes `child_lifecycle_ret.jl`** — switching it changes every
result, so it is a deliberate step, not a side effect.

**Method used:** copy `child_lifecycle_ret.jl` → `child_lifecycle.jl`, delete the ~61
retirement lines (`t_retire`, `h_avg`, `util_retire`, `pension_amount`,
`asset_constraint_retire`, `create_interpolator_retire`, `value_of_retirement`, and the two
retirement branches in each simulator), and port the two `max(a_next, a_min)` lines from
`_ar1.jl`. Then retire both old modules to `archive/`.

Net: delete ~61 lines + add 2, versus port ~22 sites and fix a tax specification.

#### 0.5 — ε timing: DECIDED

**Convention:** `ε₀` observed at the half period, `z₀` not. Enrolment and transfer may
condition on `ε₀` but not on realized `z₀`. The parent's `t=17` continuation is

```
E_{ε₀} [ max_{d,tr} E_{z₀} [ W_d(tr; ε₀, z₀) ] ]
```

with the expectations **nested in that order**. Full statement, the asymmetry in the
current code, and the fix: see **N1**.

Implications recorded as separate items:

- **0.5b / N12** — the bound `0 ≤ tr ≤ a − ā^P` introduces `ā^P`, which `model.txt` does not
  have and the code sets to `1e-9`. Give it an economic value.
- **0.5c / C6** — the convention *requires* the simulation to draw `z₀` from the same
  distribution the transfer integrated over. Currently the solve uses `stationary_dist`
  and the simulation uses the median state. Raised to High.

**Also update `model.txt`:** replace the displayed `V^{CD}_{T_L-1} = max_tr E_{ε₀,z}[·]`
with `V^{CD}_{T_L-1} = E_{ε₀}[V^{CD}_{T_L}(·,ε₀)]`, and let eq. (T_L−1) take `E_z` only —
the current pair double-counts the ε expectation.

#### 0.6 — Child horizon (nobody has flagged this)

`model.txt`: the child is followed from 18 to death at 68. Ages 18…68 inclusive is
**51 periods**.

| module | `T` | implied terminal age |
|---|---|---|
| `child_lifecycle_ret.jl` | 52 | **69** ❌ |
| `child_lifecycle_ar1.jl` | 50 | **67** ❌ |

Neither matches. Decide `T = 51` (or restate the paper) before solving anything, because
`T` shifts every backward-induction value.

#### 0.7 — Shock process

`model.txt` says "we assume these shocks follow a random walk: `z_t = ρz_{t-1} + ε`".
A random walk means `ρ = 1`. The code uses `ρ = 0.9` (parent) and `0.95` (child), and
Tauchen **cannot** discretize a unit root (infinite unconditional variance).

Either estimate and report `ρ`, or commit to a unit root and change the discretization.
Note `wage2_styled.do` identifies shock variances from *second* differences (lines 55+),
which is the standard permanent-transitory decomposition — that suggests a unit-root
permanent component plus a transitory one, i.e. the code's single stationary AR(1) may be
the wrong object entirely. **Check what the do-file actually estimates before choosing.**

#### 0.8 — φ normalization

`model.txt` says `(φ₁,φ₂,φ₃)` are "strictly positive and normalized to sum to one". The
code has `1.0 + 20.0 + 0.03 = 21.03`. Under CRRA this is not a free rescaling — it
interacts with `ρ` and with the scale of the terminal value. Either renormalize and
re-anchor, or drop the sentence.

#### 0.9 — College length

The prose says four years, entering the labor market at 22. The `V^E` display has a case at
`t = 22` whose continuation is `V^W_{23}` — five years. The code implements four
(`t_college = 4`, ages 18–21). **Fix the paper's third case to `t = 21`.**

#### Also confirm

The excerpt you circulated has `V^P = ψ log HC − κ log a_term − ω V^C` and
`(1−θ)V^C − θV^P`. `docs/model.txt` has **`+`** in all three places. Confirm your working
draft has not picked up sign errors — as written, those minuses invert the entire objective.

**Exit criterion:** `model.txt` updated, decisions recorded, and `docs/MODEL.md` re-mapped
to the chosen specification.

---

### Phase 1 — Make every run deterministic and observable  *(hours)*

| # | Issue | Action |
|---|---|---|
| 1.1 | **N11** | Fix `batch_dir` ordering and the `final_assets`/`final_hc`/`model_parent_hetro` → `*_het` renames. Verify on a **small grid** first, not a production solve. |
| 1.2 | **P2** | Seed everything and establish common random numbers across arms: initial conditions, parent shocks, child shocks, taste shocks, belief draws. |
| 1.3 | **C10 + P6** | At every optimization site check return code, `isfinite(minf)`, `all(isfinite, xopt)`, constraint violation, distance from bounds. Never keep a stale `minf` after replacing `x_opt`. |
| 1.4 | **N2** | Return structured diagnostics instead of printing; stop suppressing. |
| 1.5 | **X1 (minimal)** | NaN/Inf counts per array; Bellman constraint violations; simulated states outside grids; boundary shares; finite-difference gradient agreement. |

**Exit criterion:** a deterministic small-grid run that either completes cleanly or fails
loudly at the exact state responsible.

---

### Phase 2 — Repair the child/transfer numerical domain  *(days)*

| # | Issue | Action |
|---|---|---|
| 2.1 | **C1** | Remove the `-Inf` contamination. Also resolve the deeper inconsistency: `c_min = 0.3` versus the optimizer's `c ≥ 0.01` excludes economically feasible college states. |
| 2.2 | **C11** | Transfer solve and simulation must evaluate the same value function over the same asset domain. Either include zero in the grid or impose one justified lower bound. |
| 2.3 | **C4** | Same bounded optimizer, bounds, tolerances and search strategy for both alternatives. It is one-dimensional — golden-section or grid-plus-refinement is safer than SLSQP. |
| 2.4 | **C9** | Impose consistent domains for `a_next` **and** `k_next`; expand grids where boundary shares are material. Clamp only floating-point-sized violations — clipping economically meaningful states silently rewrites the transition law. |
| 2.5 | **P3** | Same treatment on the parent side. Do not simultaneously cap `a_next ≤ a_max` in the solver and let the simulation exceed it. |

**Exit criterion:** no non-finite values; no material clipping; negligible upper-grid
binding; transfer values and simulated policies agree state by state.

---

### Phase 3 — Correct the central economic equations  *(days)*

| # | Issue | Note |
|---|---|---|
| 3.1 | **P1** | Two lines. Could be done earlier, but its full-model effect is only measurable once the terminal value is clean. |
| 3.2 | **C2** | `(HC+1)^2`, not `^4`. |
| 3.3 | **N1** | Implement the Phase-0.5 timing. Concretely: rebuild `v_max` as `Σ_it t_weight[it]·max.(sol_tr_v_college[:,:,:,it], sol_tr_v_work[:,:,:,1])` at all 11 sites, and **delete `optimal_transfer_exp_college!`** — which also closes the 30% NaN in C1. Work branch and forward simulator need no change. |
| 3.3b | **C6** | Draw the initial child shock from `stationary_dist` (or condition the transfer on an observed `z₀`). Required for N1's timing to be coherent, so it lands with it. |
| 3.3c | **N12** | Apply the chosen `ā^P` as the transfer bound in both branches. Pairs with C4, which currently starts the college search hard against that boundary. |
| 3.4 | **N10** | Re-solve the work path at the treatment `y` in cells 77/79. Copying baseline work policies into a high-`y` child model is invalid. |
| 3.5 | **P4** | Remove the `-1e8` penalty branches; objective and gradient must always describe the same function. |
| 3.6 | **C3** | Switch the notebook to `child_lifecycle.jl`; archive `_ret.jl` and `_ar1.jl`. The module is already built and verified. |

**Exit criterion:** baseline policies satisfy feasibility and gradient tests, and the
backward and forward discrete-choice rules agree state by state.

---

### Phase 4 — Numerical accuracy  *(weeks)*

| # | Issue | Note |
|---|---|---|
| 4.1 | **C5** | Compare Tauchen vs Rouwenhorst at N = 7, 9, 11. Match moments and persistence; do not pick by convention. |
| 4.2 | — | *(C6 moved to 3.3b — it is now a consistency requirement of the N1 timing, not an accuracy refinement.)* |
| 4.3 | **P5** | Do **not** blindly swap linear for unrestricted cubic. Compare: linear + derivative-free optimizer; shape-preserving (Schumaker); smooth + monotonicity checks. |
| 4.4 | **X1 (full)** | Bellman residuals, grid-refinement comparisons, simulation-domain coverage, monotonicity/concavity, Monte Carlo standard errors, enrolment/transfer stability across grids. |

**Exit criterion:** key moments and policy functions stable under grid and shock refinement.

---

### Phase 5 — Counterfactual design and reporting  *(days)*

| # | Issue | Note |
|---|---|---|
| 5.1 | **N4** | Equalize ω across the θ arms. |
| 5.2 | **N5** | Passing `psi_terminal_belief_bin` to the college models is **not sufficient** while the work-transfer value is copied from a shared `base_child`. Recompute the belief-specific work transfer value too. Interacts with N10. |
| 5.3 | **N7** | Make the resource arms symmetric, or state why they differ. |
| 5.4 | — | Other experiment-definition defects: "High φ₂" *reduces* φ₂ from 20 to 15; the σ₄ slope experiment also moves its intercept; `R_1_baseline = 0.05` differs from the constructor's `0.06`. |
| 5.5 | **N3** | Replace or remove the invalid CEV. |
| 5.6 | **N8, N9** | Fix labels **after** treatment definitions are final. |
| 5.7 | **M1** | Write tables to `output/tables/`; wire in `write_manifest` (seeds, commit, parameters). |

---

### Dependency notes

- **Phase 0 before everything.** 0.4 and 0.5 decide which Bellman problem you are repairing.
- **N11 before any fix can be validated** — a notebook that cannot run in order cannot verify anything.
- **P2 before any counterfactual re-run**, or you cannot attribute a change to a fix.
- **C1 before N1.** N1 rewrites how the terminal value is assembled; doing that while 30% of the input is NaN means you cannot tell whether it worked. Note the fixes overlap: deleting `optimal_transfer_exp_college!` removes the array that carries the 30% NaN, so N1 partly *is* the C1 fix on the transfer side.
- **N1, C6 and N12 land together.** The timing convention is only coherent if `z₀` is drawn from the distribution the transfer integrated over (C6) and the transfer bound is the stated one (N12). Fixing N1 alone leaves the model internally inconsistent in a different way.
- **X1 (minimal) early.** It converts most of Phases 2-3 from eyeballing into checkable assertions.
- **N5 after N10** — they share the `base_child` copying defect.
