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
| `code/run_all.jl` | **One reproducible end-to-end run.** Solve, simulate, diagnose, tables, PDF. |
| `code/transfer_CRRA_wage.ipynb` | Interactive driver: counterfactuals and figures. |
| `code/src/parent_family.jl` | **Parent problem.** Struct, constructor, backward-induction solver, objectives, constraints, simulators. Extracted from the notebook — edit the model *here*. |
| `code/src/child_lifecycle.jl` | **Child lifecycle — CANONICAL.** No retirement, progressive tax. All child-side fixes go here. |
| `code/src/child_lifecycle_ret.jl` | Superseded (had retirement). Reference only — do not fix. |
| `code/src/child_lifecycle_ar1.jl` | Superseded (flat tax). Reference only — do not fix. |
| `code/src/diagnostics.jl` | Accuracy checks: Bellman residuals, domains, monotonicity, gradients. |
| `code/src/tables.jl` | LaTeX tables (`threeparttable`) and the PDF build. |
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
2. Cell 4 — `paths.jl`, `manifest.jl`, `diagnostics.jl`, `child_lifecycle.jl`, then
   construct and solve the child model
3. Cell 6 — `include("src/parent_family.jl")` — **must come after step 2**, because
   `parent_family.jl` names `ConSavLaborCollege_AR1` in its type signatures
4. Everything after — parent solve, simulation, counterfactuals, plots

`run_all.jl` at production grids (child 50×50×10, parent 30×2×30, `simN` 5000) takes
about **half a minute**. The notebook additionally solves ~20 belief-specific parent
models for the subjective-expectations experiment, which is the expensive part.

To work with the model outside the notebook (from `code/`):

```julia
include("src/paths.jl")
include("src/manifest.jl")
include("src/diagnostics.jl")
include("src/child_lifecycle.jl")
include("src/parent_family.jl")
m = Parent_child_interaction_age_specific_AR1(Na=30, Nk=2, Nhc=30, simN=5000)
m.V_child_interp = V_child_interp1   # built from the child solve
solve_model!(m); simulate_model!(m)
```

> ⚠️ **Before trusting a run.** The diagnostics no longer understate the problem —
> `check_simulation` counts non-finite states, `check_solver_domain` measures the
> *solution* leaving the grid (which forward simulation cannot see), and
> `check_feasibility_mask` checks the NaN pattern against both theoretical masks. One
> caveat remains: **P5**, the continuation interpolation moves optimal labor supply by up
> to 0.11–0.17 at some states, and refining the grid does not shrink it. See
> [`ERRORS.md`](ERRORS.md).

### One reproducible run

```bash
cd code && julia --project=.. run_all.jl            # production grids
cd code && julia --project=.. run_all.jl --quick    # smoke test, ~20 s
```

Solves the child lifecycle and the parent problem, simulates, runs every accuracy
diagnostic, writes all LaTeX tables to `output/tables/`, and compiles them into
`output/reports/all_tables.pdf`.

Reproducible by construction: every RNG is seeded from one `SEED` constant;
`Project.toml`/`Manifest.toml` pin the package set; each table emits a `.meta.toml` with
the git commit, timestamp and parameters; and the PDF wrapper is generated from whatever
`.tex` files are on disk, so it can never go stale against the tables.

### LaTeX tables

`code/src/tables.jl` emits `threeparttable` + `booktabs` tables in the same format as
`Redistribution_and_Human_Capital/{Tables,outcomes}` — `\toprule\toprule` … `\midrule`
… `\bottomrule\bottomrule`, `[H]` placement, `\tnote{}` footnotes. Each file is
`\input`-able straight into the paper.

| writer | produces |
|---|---|
| `table_college_work(path_choice, name)` | counts and shares, like `base_college_work_choice.tex` |
| `table_outcomes(models, labels, name)` | end-of-family outcomes, like `resource_summary.tex` |
| `table_belief_groups(...)` | per-belief-group means, like `hetero_table.tex` |
| `table_diagnostics(pairs, name)` | numerical diagnostics |
| `write_table(name; ...)` | the generic builder for anything else |
| `build_tables_pdf()` | every `.tex` in `output/tables/` into one PDF |

The paper's preamble needs `booktabs`, `threeparttable` and `float`.

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
- `z` — AR(1) wage shock (Tauchen, `Np = 3`). See ERRORS.md Phase 4: Rouwenhorst matches
  the target moments exactly where Tauchen overstates the sd by 31% — an open decision.

Three regimes: `t = 1..6` parents decide alone (4 controls); `t = 7..16` the child bargains
and study time is added (5 controls); `t = 17` terminal, continuation is the college/transfer
value `V_child_interp`.

**Child problem** (`code/src/child_lifecycle.jl`), `T = 51` periods, ages 18–68:

- college path, `t_college = 4` years (ages 18–21), then the work path
- work path with progressive tax and learning-by-doing `HC_{t+1} = HC_t + h_t`
- **no retirement** — removed in Phase 0.4 to match `model.txt`
- terminal period: works and consumes everything, no bequest

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
| `mu_0, mu_1` | 1.0, −0.04 | welfare weight `μ̃_t = 1` for `t ≤ 6`, then `μ_0 + μ_1(t−6)`. The boundary is `T_CHILD_VOICE` in `parent_family.jl` — one constant, six call sites |
| `tau`, `tax_lambda` | 0.18, 0.82 | progressive tax `λ(wh)^(1−τ)` |
| `r`, `beta_0` | 0.03, 0.96 | interest rate; discount factor |
| `R_0, R_1` | 2.0, 0.06 | HC technology TFP, `R_t = R_0 + R_1(t−1)` |
| `sigma_{1..4}_0/1` | see constructor | elasticities, entered as **logs**: `σ_jt = exp(σ_j0 + σ_j1·(t−1))` |
| `a_max`, `Na` | 50.0, 30 | parent's asset grid |
| `Nk` | 2 | BothCollege ∈ {0,1} |
| `hc_max`, `Nhc` | 6.0, 30 | child HC grid |
| `Np`, `p_ar1`, `sigma_p` | 3, 0.9, 0.1 | AR(1) wage shock |
| β wage coefficients | see constructor | from the Stata wage regression |

**Child** — `ConSavLaborCollege_AR1` (in `code/src/child_lifecycle.jl`)

| Param | Value | Meaning |
|---|---|---|
| `T`, `t_college` | 51, 4 | horizon (ages 18–68); college years |
| `c_floor`, `delta_P` | 0.01, 0.01 | consumption floor (= optimizer bound); min retained parental asset |
| `a_max`, `Na` | 100.0, 30 | **child's** asset grid. 100, not 50: it must cover the parent's terminal assets plus 51 periods of the child's own accumulation |
| `ap_min`, `ap_max`, `Nap` | `delta_P`, `a_max`, `Na` | **parental** asset grid, separate since N13. Indexes the transfer arrays and the terminal-value spline. Starts at `delta_P` so the parent can always retain its floor; carries an exact node at the college threshold `a_req[1] + delta_P`, so there is no dead band |
| `rho`, `eta`, `phi` | 1.5, 2.0, 18.0 | CRRA; inverse Frisch; labor disutility scale |
| `kappa` | 5.0 | psychic cost of college |
| `college_cost`, `college_boost` | 1.2, 2.0 | annual cost; annual HC increment `b*` |
| `psi_terminal`, `kappa_terminal`, `omega` | 1.0, 10.0, 0.5 | parent's terminal weights on child HC / own assets / altruism |
| `mu` | 0.5 | **θ**, parent's weight in the college decision |
| `Np`, `p_ar1`, `sigma_p` | 5, 0.95, 0.2 | AR(1) wage shock |
| `Nt`, `sigma_eps` | 11 (10 passed), 0.5 | Gauss-Hermite nodes for the taste shock ε₀ |

Note the notebook overrides several of these at construction
(`Na=50, Nk=50, Nt=10, rho=1.5, a_max=100.0`).

---

## Known issues

The full audit is in [`ERRORS.md`](ERRORS.md): **4 open** (1 high, 1 medium, 2 deferred
by instruction), plus a Resolved log of what has been fixed.

1. 🟠 **P5** — the continuation interpolation moves policies. Re-solving the same states
   against an interpolating cubic spline instead of `Gridded(Linear())` moves optimal
   labor supply by up to 0.11–0.17 of the time endowment, and quadrupling the grid does
   not shrink it. The Bellman residual is blind to this: it sits at 5.6e-13 on every grid
   because it re-evaluates the stored policy rather than re-optimizing. This is a decision
   about the interpolation scheme, not a bug to patch — ERRORS.md lays out the options.
2. 🟡 **P7b** — the `BothCollege` share is hardcoded at `Bernoulli(0.3)` and still needs
   an empirical source from the estimation sample.
3. ⏸️ **C2**, **C8** — deferred out of scope by instruction.

Two numerical guards are worth knowing about when reading the parent solver, both added
after the solve died on NaN iterates:

- `LEISURE_FLOOR = 1e-4`. The child's leisure `1 − τ_p − i_c` is a *nonlinear* constraint,
  so SLSQP evaluates points that violate it. Below the floor `log` is **linearized**, not
  flattened: value and slope both match at the floor, so the derivative is bounded by
  `1/L` instead of cliffing by it. Verified inactive at the optimum — the minimum child
  leisure over 54,000 solved states is 0.465.
- Both backward-induction loops floor `t_p` and `h_p` at `1e-4` (the parent-only loop used
  `1e-6`) and clamp the warm start from `t+1` into the current box.

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
