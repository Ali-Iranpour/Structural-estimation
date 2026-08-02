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
| Find an old version | [`archive/NOTES.md`](archive/NOTES.md) |

---

## Layout

```
.
├── code/
│   ├── transfer_CRRA_wage.ipynb    driver: solve, simulate, counterfactuals, figures
│   └── src/
│       ├── paths.jl                every path in the project — nothing else hard-codes one
│       ├── manifest.jl             run provenance (git SHA, versions, parameters)
│       ├── parent_family.jl        parent problem: struct, solver, simulators
│       ├── child_lifecycle_ret.jl  child lifecycle WITH retirement (the one used)
│       └── child_lifecycle_ar1.jl  child lifecycle, no retirement (not included)
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
│   ├── tables/                     tracked in git
│   ├── reports/                    tracked in git
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

Then open `code/transfer_CRRA_wage.ipynb` with the IJulia (Julia 1.11) kernel and run
cells in order. See [`docs/GUIDE.md`](docs/GUIDE.md) for load order, runtime, and caveats.

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

[`docs/ERRORS.md`](docs/ERRORS.md) carries the full list, ordered by severity. The
three that most affect results: a spurious `∂V/∂k` term in the labor-supply gradient, the
college choice being taken outside the taste-shock expectation, and unseeded RNG in the
parent simulation (so counterfactual arms do not share random numbers). None are fixed.
