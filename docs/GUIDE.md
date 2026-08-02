# Guide — Running the Model

The current model. A family (two parents, one child) is solved over `t = 1..17`
(child ages 0–17); at age 18 the family chooses college vs. work and the parents
transfer assets; the child is then solved to age 68.

The specification is in [`model.txt`](model.txt). The mapping from each equation to the
code that implements it is in [`MODEL.md`](MODEL.md).

---

## Files

| File | What it is |
|---|---|
| `code/transfer_CRRA_wage.ipynb` | **Driver notebook.** Solves, simulates, runs all counterfactuals, writes figures. |
| `code/src/parent_family.jl` | **Parent problem.** Struct, constructor, backward-induction solver, objectives, constraints, simulators. Extracted from the notebook — edit the model *here*. |
| `code/src/child_lifecycle_ret.jl` | **Child lifecycle, with retirement.** This is the module the notebook includes. |
| `code/src/child_lifecycle_ar1.jl` | Child lifecycle, *no* retirement. Kept for reference; **not** currently included by the notebook. |
| `code/src/paths.jl` | **Every path in the project.** Nothing else hard-codes a folder name. |
| `code/src/manifest.jl` | Run provenance: `write_manifest(dir; params...)`. |
| `model.txt` | LaTeX model specification from the paper. |
| `MODEL.md` | Equation ↔ code map. |
| `../Project.toml` | Dependencies, pinned to the verified set (Julia 1.11.3). |
| `../output/figures/` | Figure output — **tracked in git**. |
| `../output/data/` | Solved models / simulation dumps — git-ignored. |

---

## Running it

```bash
./tools/setup-git-filters.sh                              # once per clone
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Open `code/transfer_CRRA_wage.ipynb` with the IJulia (Julia 1.11) kernel and run cells in
order. Load order matters:

1. Cell 3 — package imports
2. Cell 4 — `paths.jl`, `manifest.jl`, `child_lifecycle_ret.jl`, then construct and
   solve the child model
3. Cell 6 — `include("src/parent_family.jl")` — **must come after step 2**, because
   `parent_family.jl` names `ConSavLaborCollege_AR1` in its type signatures
4. Everything after — parent solve, simulation, counterfactuals, plots

A full run solves ~20 belief-specific parent models on a 30×2×30×3 grid with 5-variable
NLopt problems at each point, plus the child lifecycle. Budget hours, not minutes.

To work with the model outside the notebook (from `code/`):

```julia
include("src/paths.jl")
include("src/manifest.jl")
include("src/child_lifecycle_ret.jl")
include("src/parent_family.jl")
m = Parent_child_interaction_age_specific_AR1(Na=30, Nk=2, Nhc=30, simN=5000)
m.V_child_interp = V_child_interp1   # built from the child solve
solve_model!(m); simulate_model!(m)
```

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

## Structure of the model

**Parent problem** (`code/src/parent_family.jl`), states `(a, k, HC, z)`:

- `a` — household assets
- `k` — **`BothCollege` indicator ∈ {0,1}**, a fixed household type. *Not* parental human
  capital; it does not accumulate (`k_next = capital`). It enters only through the wage
  equation. The grid is `Nk = 2`.
- `HC` — child's cognitive skill
- `z` — AR(1) wage shock (Tauchen, `Np = 3`)

Three regimes: `t = 1..7` parents decide alone (4 controls); `t = 8..16` the child bargains
and study time is added (5 controls); `t = 17` terminal, continuation is the college/transfer
value `V_child_interp`.

**Child problem** (`code/src/child_lifecycle_ret.jl`), `T = 52` periods from age 18:

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

**Child** — `ConSavLaborCollege_AR1` (in `code/src/child_lifecycle_ret.jl`)

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

The full audit — every error, its severity, and the exact file and line — is in
[`ERRORS.md`](ERRORS.md). **Nothing in it has been fixed.**

The three that most affect results:

1. 🔴 **Spurious `∂V/∂k` in the labor-supply gradient** (`parent_family.jl:636, 678`).
   `k` is the fixed `BothCollege` indicator, so `∂k'/∂h_p = 0`, but both gradients still
   add `dV_dk_sum` — the whole lifetime value gap between education types. Drives `h_p`
   to its bound for all `t ≤ 16`.
2. 🔴 **College choice taken outside the ε expectation** (11 sites in the notebook).
   `max(E_ε[V^E], V^W)` instead of `E_ε[max(V^E(ε), V^W)]`. Understates the option value
   of college, and disagrees with what `simulate_model_family!` actually does.
3. 🔴 **Unseeded RNG in the parent simulation** (`parent_family.jl:941, 1091`).
   Counterfactual arms do not share random numbers.

---

## Change log

**2026-08-02b — `code/` `docs/` `output/` layout.** No model logic changed.

- Fixed a regression introduced earlier the same day: removing the unconditional
  `mkpath` from `plot_family_counterfactuals` left it with no directory creation at
  all, so `save=true` would have failed. `mkpath` restored inside the `if save`
  branch. Audited all 7 `savefig` sites; the other 6 were already correct.
- Added `docs/ERRORS.md` — the full audit with severity, file and line.

- Code to `code/`, model spec and notes to `docs/`, figures to `output/figures/`
  (structure preserved; the 20 empty timestamped `Parameters/` folders dropped).
- Added `code/src/paths.jl`. Replaced all 30 hard-coded `joinpath(@__DIR__, "plots", …)`
  calls with `figdir(…)`, which also fixed the `"Plots"`/`"plots"` capitalization
  inconsistency.
- Fixed the bug that created those empty folders: `plot_family_counterfactuals` called
  `mkpath(save_dir)` before checking `if save`, so every `save=false` call still made a
  dated directory.
- Added `code/src/manifest.jl` (`write_manifest`) for run provenance.
- Renamed `ConSavLabor_college_ret.jl` → `code/src/child_lifecycle_ret.jl`,
  `ConSavLabor_college_AR1.jl` → `code/src/child_lifecycle_ar1.jl`.
- `output/figures/` and `output/tables/` are now tracked in git; `output/data/` ignored.
- Notebook outputs stripped on commit via `tools/nbstrip.py` (55.8 MB → 0.17 MB).
  LFS scoped to `archive/**/*.ipynb` so archived notebooks keep working.
- `Manifest.toml` committed, pinned to the verified version set.

**2026-08-02a — Reorganize around the current model.** Parent model extracted from the
notebook into `parent_family.jl` (verbatim, no logic changed); superseded notebooks moved
to `archive/`; `Project.toml` and `MODEL.md` added; READMEs rewritten. Pre-extraction
notebook preserved at `archive/Combined Models/Full model/transfer_CRRA_wage_ORIGINAL.ipynb`.
