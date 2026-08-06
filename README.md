# Structural Estimation — Parent-Child Lifecycle Model

Julia code for a structural lifecycle model of intergenerational human capital
investment and college choice. A family (two parents, one child) is followed from
the child's birth to age 18; the child is then followed to age 68.

---

## Where to start

| I want to… | Go to |
|---|---|
| Run the model | [`docs/GUIDE.md`](docs/GUIDE.md) |
| Read the model as written in the paper | [`docs/model.txt`](docs/model.txt) |
| Find which code implements which equation | [`docs/MODEL.md`](docs/MODEL.md) |
| Know what's currently broken | [`docs/ERRORS.md`](docs/ERRORS.md) — every error, severity, file and line |
| Understand why the spec says what it says | [`docs/SPEC_DECISIONS.md`](docs/SPEC_DECISIONS.md) |
| Find an old version | [`archive/NOTES.md`](archive/NOTES.md) |

---

## Layout

```
.
├── code/
│   ├── run_all.jl                  ONE reproducible end-to-end run (--quick to smoke test)
│   ├── transfer_CRRA_wage.ipynb    interactive driver: counterfactuals, figures
│   └── src/
│       ├── paths.jl                every path in the project — nothing else hard-codes one
│       ├── manifest.jl             run provenance (git SHA, versions, parameters)
│       ├── diagnostics.jl          accuracy checks: Bellman residuals, domains, gradients
│       ├── tables.jl               LaTeX tables (threeparttable) + PDF build
│       ├── parent_family.jl        parent problem: struct, solver, simulators
│       ├── child_lifecycle.jl      child lifecycle — CANONICAL, no retirement
│       ├── child_lifecycle_ret.jl  superseded, reference only
│       └── child_lifecycle_ar1.jl  superseded, reference only
│
├── docs/
│   ├── model.txt                   LaTeX model specification from the paper
│   ├── MODEL.md                    equation ↔ code map
│   ├── GUIDE.md                    how to run, parameters
│   ├── ERRORS.md                   full audit: severity, file, line
│   ├── SLSQP_algorithm.md          methodology note
│   └── Flat_policy_function.md     methodology note
│
├── output/
│   ├── figures/                    81 PDFs — tracked in git
│   ├── tables/                     .tex + .meta.toml provenance — tracked in git
│   ├── reports/                    all_tables.pdf — tracked in git
│   └── data/                       solved models, simulation dumps — git-ignored
│
├── tools/
│   ├── nbstrip.py                  strips notebook outputs on commit
│   └── setup-git-filters.sh        run once per clone
│
├── archive/                        superseded work — see archive/NOTES.md
├── Project.toml / Manifest.toml    dependencies, pinned to a verified set
└── README.md
```

---

## Model in one paragraph

Two parents and one child interact over `t = 1..17` (child ages 0–17). Parents choose
consumption, labor supply, education expenditure, and time with the child; from age 7
the child bargains over their own study time and leisure under an age-varying welfare
weight. The child's cognitive skill follows a Cobb-Douglas production function in
parental time, education spending, lagged skill, and (after age 7) the child's own
study time. At age 18 the family jointly chooses college vs. work and the parents
transfer assets, which become the child's initial wealth. The child then solves a
consumption-saving-labor problem to age 68. Parental wages follow an estimated profile
in age and whether both parents hold a college degree, plus an AR(1) shock; labor
income is taxed progressively.

Full specification: [`docs/model.txt`](docs/model.txt).

---

## Setup

```bash
git clone <repo> && cd Structural-estimation
./tools/setup-git-filters.sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Then either run everything non-interactively:

```bash
cd code && julia --project=.. run_all.jl
```

or open `code/transfer_CRRA_wage.ipynb` with the IJulia (Julia 1.11) kernel and run cells
in order. See [`docs/GUIDE.md`](docs/GUIDE.md) for load order, runtime, and caveats.

---

## Conventions

**Paths.** Never hard-code a folder name. `code/src/paths.jl` defines every location;
use `figdir("Baseline")`, `tabpath()`, `datapath()`. The `*path` variants create the
directory and return it; the `*dir` variants are pure. This is what lets the repository
be reorganized without breaking a single `savefig`.

**Provenance.** When a run produces figures or tables worth keeping, record what made
them:

```julia
write_manifest(figpath("Parameters"); experiment = "sigma counterfactuals",
                                      mu_1 = -0.04, rho = 1.5, Na = 30, simN = 5000)
```

This writes `run_manifest.toml` with the timestamp, git commit (suffixed `-dirty` if the
tree was modified), Julia and package versions, and the parameters you pass. It is the
link from a figure in the paper back to the code that produced it.

**Notebook outputs are stripped on commit.** Your working copy keeps its results; the
committed version does not (55.8 MB → 0.17 MB). This is why `output/figures/` is tracked
— figures, not notebook cells, are the durable record of results. Run
`./tools/setup-git-filters.sh` once per clone or the filter will not be active.

**Dependencies are pinned exactly** in `Project.toml`. Unpinned, Pkg resolves ForwardDiff
to 1.x and NLopt to 1.2.x, neither of which has been tested here. Relax one pin at a time
and re-run before committing a new Manifest.

---

## Known limitations

[`docs/ERRORS.md`](docs/ERRORS.md) carries the full list — every finding with its severity,
file and line — followed by the improvement backlog and an ordered work plan.

**4 findings are open: 1 high, 1 medium, 2 deferred by instruction.** The one that matters
before trusting any output:

**P5 — the continuation interpolation moves policies, and this is now measured.** The
solver's `Gridded(Linear())` continuation is C0 but not C1. Re-solving the same states
against an interpolating cubic spline instead moves optimal labor supply by up to **0.11 to
0.17** of the unit time endowment, and **quadrupling the grid from 20×20 to 80×80 does not
shrink it**. The Bellman residual is blind to this — it sits at 5.6e-13 on every grid,
because it re-evaluates the stored policy rather than re-optimizing.

This is not a bug to patch: it means the numerical solution is not pinned down at those
states, and the fix is a decision about the interpolation scheme. ERRORS.md lays out the
three options.

The other two: **P7b**, the `BothCollege` share is hardcoded at `Bernoulli(0.3)` and still
needs an empirical source; **C2** and **C8** are deferred out of scope by instruction.

A green `run_all.jl` now means the solution is internally consistent, on-domain, and
feasibility-masked correctly — it does not settle P5.
