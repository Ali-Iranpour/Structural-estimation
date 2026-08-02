# Structural Estimation — Parent-Child Lifecycle Model

Julia code for a structural lifecycle model of intergenerational human capital
investment and college choice. A family (two parents, one child) is followed from
the child's birth to age 18; the child is then followed to age 68.

**The current model lives in [`Combined Models/Full model/`](Combined%20Models/Full%20model/).**
Everything in `archive/` is superseded and kept only for reference.

---

## Where to start

| I want to… | Go to |
|---|---|
| Run the model | [`Combined Models/Full model/README.md`](Combined%20Models/Full%20model/README.md) |
| Read the model as written in the paper | [`Combined Models/Full model/model.txt`](Combined%20Models/Full%20model/model.txt) |
| Find which code implements which equation | [`Combined Models/Full model/MODEL.md`](Combined%20Models/Full%20model/MODEL.md) |
| Know what's currently broken | "Known issues" in the Full model README |
| Find an old version | [`archive/NOTES.md`](archive/NOTES.md) |

---

## Layout

```
.
├── Combined Models/Full model/     ← CURRENT MODEL
│   ├── transfer_CRRA_wage.ipynb        driver notebook (solve, simulate, counterfactuals)
│   ├── src/parent_family.jl            parent problem: struct, solver, simulators
│   ├── ConSavLabor_college_ret.jl      child lifecycle — WITH retirement (the one used)
│   ├── ConSavLabor_college_AR1.jl      child lifecycle — no retirement (not included)
│   ├── model.txt                       LaTeX model spec from the paper
│   ├── MODEL.md                        equation ↔ code map
│   ├── README.md                       how to run, parameters, known issues
│   ├── Project.toml                    pinned dependencies
│   └── plots/                          figure output (git-ignored)
│
├── docs/                           methodology notes
│   ├── SLSQP_algorithm.md
│   └── Flat_policy_function.md
│
└── archive/                        superseded work — see archive/NOTES.md
    ├── Combined Models/{Archive, Child & Parent part, Full model}/
    ├── ConSavLabor/  ConSavLabor_college/  Family Model/  test codes/
    └── Thesis_code.ipynb
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

Full specification: [`model.txt`](Combined%20Models/Full%20model/model.txt).

---

## Setup

```bash
cd "Combined Models/Full model" && julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Then open `transfer_CRRA_wage.ipynb` with the IJulia (Julia 1.11) kernel and run cells
in order. See the Full model README for runtime expectations and caveats.

---

## Notes on this repository

- **`plots/` is git-ignored.** Figures are not under version control — regenerate them by
  running the notebook, or drop the ignore rule if you want them tracked.
- **Git history is large (~1.3 GB).** Notebooks are committed with their output cells and
  several are 30–58 MB. Consider `nbstripout` before further commits; otherwise history
  grows by roughly the notebook size on every save.
- **Git LFS is applied inconsistently.** `.gitattributes` declares `*.ipynb filter=lfs`,
  but only some notebooks are actually stored as LFS pointers. Worth making uniform.
