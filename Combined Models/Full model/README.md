# Full Model — Parent-Child Lifecycle Model

The current model. A family (two parents, one child) is solved over `t = 1..17`
(child ages 0–17); at age 18 the family chooses college vs. work and the parents
transfer assets; the child is then solved to age 68.

The specification is in [`model.txt`](model.txt). The mapping from each equation to the
code that implements it is in [`MODEL.md`](MODEL.md).

---

## Files

| File | What it is |
|---|---|
| `transfer_CRRA_wage.ipynb` | **Driver notebook.** Solves, simulates, runs all counterfactuals, writes figures. |
| `src/parent_family.jl` | **Parent problem.** Struct, constructor, backward-induction solver, objectives, constraints, simulators. Extracted from the notebook — edit the model *here*. |
| `ConSavLabor_college_ret.jl` | **Child lifecycle, with retirement.** This is the module the notebook includes. |
| `ConSavLabor_college_AR1.jl` | Child lifecycle, *no* retirement. Kept for reference; **not** currently included by the notebook. |
| `model.txt` | LaTeX model specification from the paper. |
| `MODEL.md` | Equation ↔ code map. |
| `Project.toml` | Pinned dependencies (verified on Julia 1.11.3). |
| `plots/` | Figure output. Git-ignored. |

---

## Running it

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Open `transfer_CRRA_wage.ipynb` with the IJulia (Julia 1.11) kernel and run cells in
order. Load order matters:

1. Cell 3 — package imports
2. Cell 4 — `include("ConSavLabor_college_ret.jl")`, then construct and solve the child model
3. Cell 6 — `include("src/parent_family.jl")` — **must come after step 2**, because
   `parent_family.jl` names `ConSavLaborCollege_AR1` in its type signatures
4. Everything after — parent solve, simulation, counterfactuals, plots

A full run solves ~20 belief-specific parent models on a 30×2×30×3 grid with 5-variable
NLopt problems at each point, plus the child lifecycle. Budget hours, not minutes.

To work with the model outside the notebook:

```julia
include("ConSavLabor_college_ret.jl")
include("src/parent_family.jl")
m = Parent_child_interaction_age_specific_AR1(Na=30, Nk=2, Nhc=30, simN=5000)
m.V_child_interp = V_child_interp1   # built from the child solve
solve_model!(m); simulate_model!(m)
```

---

## Structure of the model

**Parent problem** (`src/parent_family.jl`), states `(a, k, HC, z)`:

- `a` — household assets
- `k` — **`BothCollege` indicator ∈ {0,1}**, a fixed household type. *Not* parental human
  capital; it does not accumulate (`k_next = capital`). It enters only through the wage
  equation. The grid is `Nk = 2`.
- `HC` — child's cognitive skill
- `z` — AR(1) wage shock (Tauchen, `Np = 3`)

Three regimes: `t = 1..7` parents decide alone (4 controls); `t = 8..16` the child bargains
and study time is added (5 controls); `t = 17` terminal, continuation is the college/transfer
value `V_child_interp`.

**Child problem** (`ConSavLabor_college_ret.jl`), `T = 52` periods from age 18:

- college path, `t_college = 4` years, then the work path
- work path with progressive tax and learning-by-doing `HC_{t+1} = HC_t + h_t`
- retirement from `t_retire = 42`, pension = 0.5 × after-tax notional earnings

**Wage equations.** Parents use the estimated profile
`ln w = β₀ + β_BC·BothCollege + β_age·t + β_age2·t² + interactions + z`.
The child uses `w = w₀(1 + α·HC)·z`. Both are taxed as `λ(w·h)^(1−τ)` (HSV/Benabou).

---

## Key parameters (code defaults, not the paper's table)

**Parent** — `Parent_child_interaction_age_specific_AR1`

| Param | Value | Meaning |
|---|---|---|
| `T` | 17 | periods (child ages 0–17) |
| `rho`, `eta` | 1.5, 2.0 | CRRA; inverse Frisch |
| `phi_1_0, phi_2_0, phi_3_0` | 1.0, 20.0, 0.03 | consumption / labor disutility / child HC weights |
| `lambda_1_0, lambda_2_0` | 0.7, 0.3 | child's leisure / HC weights |
| `mu_0, mu_1` | 1.0, −0.04 | welfare weight `μ̃_t = 1` for `t ≤ 7`, then `μ_0 + μ_1(t−7)` |
| `tau`, `tax_lambda` | 0.18, 0.82 | progressive tax `λ(wh)^(1−τ)` |
| `r`, `beta_0` | 0.03, 0.96 | interest rate; discount factor |
| `R_0, R_1` | 2.0, 0.06 | HC technology TFP, `R_t = R_0 + R_1(t−1)` |
| `sigma_{1..4}_0/1` | see constructor | elasticities, entered as **logs**: `σ_jt = exp(σ_j0 + σ_j1·(t−1))` |
| `a_max`, `Na` | 50.0, 30 | asset grid |
| `Nk` | 2 | BothCollege ∈ {0,1} |
| `hc_max`, `Nhc` | 6.0, 30 | child HC grid |
| `Np`, `p_ar1`, `sigma_p` | 3, 0.9, 0.1 | AR(1) wage shock |
| β wage coefficients | see constructor | from the Stata wage regression |

**Child** — `ConSavLaborCollege_AR1` (in `ConSavLabor_college_ret.jl`)

| Param | Value | Meaning |
|---|---|---|
| `T`, `t_college`, `t_retire` | 52, 4, 42 | horizon; college years; retirement period |
| `rho`, `eta`, `phi` | 1.5, 2.0, 18.0 | CRRA; inverse Frisch; labor disutility scale |
| `kappa` | 5.0 | psychic cost of college |
| `college_cost`, `college_boost` | 1.2, 2.0 | annual cost; annual HC increment `b*` |
| `psi_terminal`, `kappa_terminal`, `omega` | 1.0, 10.0, 0.5 | parent's terminal weights on child HC / own assets / altruism |
| `mu` | 0.5 | **θ**, parent's weight in the college decision |
| `Np`, `p_ar1`, `sigma_p` | 5, 0.95, 0.2 | AR(1) wage shock |
| `Nt`, `sigma_eps` | 11 (10 passed), 0.5 | Gauss-Hermite nodes for the taste shock ε₀ |

Note the notebook overrides several of these at construction (`Na=50, Nk=50, Nt=10, rho=1.5`).

---

## Known issues

Open findings from the 2026-08 code audit. **None have been fixed** — this section is the
to-do list, ordered by severity. Line numbers refer to the current files.

### Critical

1. **Spurious `∂V/∂k` in the labor-supply FOC.** Since `k` is now the fixed `BothCollege`
   indicator, `∂k_next/∂h_p = 0`, but `obj_work_period_full` (`grad[4]`) and
   `obj_work_period_parentonly` (`grad[3]`) still add `dV_dk_sum`. With `k_grid = {0,1}`
   that term equals the entire lifetime value gap between education types — a large
   positive constant added to the hours gradient. It drives `h_p` to its upper bound for
   `t ≤ 16`. `obj_last_period_full` is correct, which confirms it's an incomplete edit.
   *Fix: delete `dV_dk_sum` from those two gradients.*

2. **College choice taken outside the ε expectation.** The model has
   `V^C = max{V^E(ε₀), V^W}` with the max *inside*, and the transfer chosen after
   uncertainty resolves. The code computes `max(E_ε[V^E], V^W)` via
   `v_max = safe_maximum.(sol_exp_v_college, sol_tr_v_work)` — 9 occurrences in the
   notebook. By Jensen this understates the option value of college and biases take-up
   downward, most at the margin. `simulate_model_family!` does it *correctly*, so the
   solve and the simulation currently use different decision rules.
   *Fix: `sum(t_weight[it] .* max.(sol_tr_v_college[:,:,:,it], sol_tr_v_work[:,:,:,1]) for it in 1:Nt)`,
   and drop `optimal_transfer_exp_college!` from the terminal-value path.*

3. **Unseeded RNG in the parent simulation.** `sample(1:Np, Weights(...))` and
   `rand(Beta(α,β), simN)` use the global RNG, so every counterfactual arm draws
   different shocks. Comparisons confound the parameter change with Monte Carlo noise,
   and results change on every re-run. The child model does this correctly via
   `draws_uniform_p`. *Fix: pre-draw seeded uniforms in the parent constructor.*

### High

4. **`-Inf` sentinel is reachable.** `compute_min_assets` uses `c_min = 0.3` to mark
   infeasible college states as `-Inf`, but `asset_constraint_college` only enforces
   `a_next ≥ a_min = 0.01`. At the current calibration `a_min_t = [3.354, 2.555, 1.732, 0.884]`,
   and the reachable set from feasible points straddles the `-Inf` region, which is then
   linearly interpolated. *Fix: enforce `a_next ≥ a_min_t[t+1]`.*

5. **Psychic cost uses the wrong power.** `ConSavLabor_college_ret.jl:485` has
   `kappa/(k+1)^4`; the model says `κ/(HC+1)²`.

6. **Retirement is in the code but not in the model.** `t_retire = 42` with a pension at a
   0.5 replacement rate. `model.txt` says "The model has no retirement stage and ends as
   the child becomes 68". Also `T = 52` gives a terminal age of 69, not 68.

7. **Convergence diagnostics discarded.** 65 uses of `@suppress_output` swallow
   `print_period_stats`, so the converged/maxeval share is unknown for every counterfactual.
   `other_dict` is only populated in the `t ≤ 7` loop, so "Other: 0.0%" is printed
   unconditionally elsewhere. Non-converged NLopt returns are accepted silently.

8. **Simulated states are never clamped** to the solver's own constraints, and
   `asset_constraint_max` (`a_next ≤ a_max`) is a numerical device with no counterpart in
   the model — it binds for the right tail of the LogNormal(0.296, 1.402) initial assets.

9. **Asymmetric transfer optimization.** Work starts at `tr_hi*0.5` with `ftol_rel=1e-8`;
   college starts at `tr_hi*0.99` with `ftol_rel=1e-6`. The discrete college/work
   comparison is between two differently-initialized local optima.

10. **Shock discretization too coarse.** Parent `Np=3` for `ρ=0.9`; child `Np=5` for
    `ρ=0.95`. Use Rouwenhorst with `N ≥ 7`. Also `E[exp(z)] = exp(σ_z²/2) ≠ 1`, giving a
    small systematic wage drift.

11. **Piecewise-linear continuation value** (`Gridded(Linear())`) under gradient-based
    SLSQP: `∇V` is discontinuous at every knot. Cubic or shape-preserving interpolation
    would cut iteration counts substantially.

### Medium

12. `(φ₁,φ₂,φ₃) = (1.0, 20.0, 0.03)` sum to 21.03; `model.txt` says they are "normalized to
    sum to one".
13. `BothCollege` share hardcoded as `Bernoulli(0.3)` — should be sourced to the estimation sample.
14. The `2 ×` multiplier in the parent `wage_func` contradicts "wages are defined as the
    mean across the two parents" (it makes it the household total).
15. **Verify age units.** `β_age * t` uses `t ∈ 1..17`. Correct only if the Stata `Age` was
    normalized (e.g. `age − 25`). The implied peak `β₂/(2|β₃|) ≈ 26.6` reads as actual age
    51.6 under that assumption (plausible) or 26.6 raw (implausible) — so probably right,
    but confirm against the regression script and state it in the paper.
16. σ₄ uses `(t−7)` while σ₁–σ₃ use `(t−1)`, so `σ_{4,0}` is the elasticity at `t=7` and
    `σ_{1,0}` at `t=1` — not comparable in a table. No returns-to-scale restriction on
    `Σ_j σ_jt` (≈0.60 at baseline, unguarded in counterfactuals).
17. Counterfactual labels: check the `μ_t` slope arms (low slope ⇒ *high* `μ̃_t`) and the
    model/label order in the college-decision figure.
18. No accuracy diagnostics anywhere — no Euler residuals, no grid-refinement check, no
    assertion that simulated states stay inside the grids.

---

## Change log

- **2026-08-02** — Repository reorganized. Parent model extracted from the notebook into
  `src/parent_family.jl` (no logic changed); superseded notebooks moved to `archive/`;
  `Project.toml`, `MODEL.md` added; both READMEs rewritten. Pre-extraction notebook
  preserved at `archive/Combined Models/Full model/transfer_CRRA_wage_ORIGINAL.ipynb`.
