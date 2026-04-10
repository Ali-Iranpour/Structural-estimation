# Structural Estimation — Lifecycle Family Model

This repository contains the Julia/Jupyter code for a structural lifecycle model of intergenerational human capital investment and college decisions. The project builds from a simple consumption-saving model up to a full parent-child dynamic model with AR(1) wage shocks, progressive taxation, and retirement.

---

## Repository Structure

```
├── ConSavLabor/                               # Core consumption-saving and labor models
│   ├── consumption_saving.ipynb               # Basic T-period consumption-saving model
│   ├── ConSavLabor.ipynb                      # Labor supply with endogenous human capital
│   ├── ConSavLabor_AR1.ipynb                  # Labor model with AR(1) persistent wage shocks
│   └── ConSavLabor_stochastic.ipynb           # Labor model with i.i.d. wage shocks
│
├── ConSavLabor_college/                       # College choice and belief heterogeneity models
│   ├── ConSavLabor_college.ipynb              # College vs. work model with lifecycle decisions
│   ├── ConSavLabor_college_AR1.ipynb          # College model with AR(1) persistent wage shocks
│   ├── ConSavLabor_college_SE.ipynb           # Model with heterogeneous beliefs about college returns
│   └── ConSavLabor_college_retire.ipynb       # College model extended with retirement phase
│
├── Family Model/                              # Family decision-making and child HC investment
│   ├── family.ipynb                           # Dynamic family model with education and care decisions
│   └── parent_child_model.ipynb               # Dynamic family model with parent-child interaction
│
├── Combined Models/                           # Combined parent-child lifecycle models
│   ├── Full model/                            # Full integrated model (see folder README)
│   │   ├── transfer_model.ipynb               # Base T-period parent-child model (log utility)
│   │   ├── transfer_model_AR1.ipynb           # Parent-child model with AR(1) wage shocks
│   │   ├── transfer_CRRA.ipynb                # Parent-child model with CRRA utility + retirement
│   │   ├── transfer_CRRA_wage.ipynb           # CRRA model with stochastic wages
│   │   ├── ConSavLabor_college_AR1.jl         # Julia module: child's AR(1) college/work lifecycle
│   │   ├── ConSavLabor_college_ret.jl         # Julia module: child's lifecycle with retirement
│   │   └── plots/                             # All generated output figures (PDFs)
│   │
│   ├── Child & Parent part/                   # Intermediate combined models
│   │   ├── Family_with_asset.ipynb            # Parent-child model with regime switch at t=7
│   │   ├── family_only_HC.ipynb               # Parent-child model with HC investment only
│   │   └── modeified_family_with_asset.ipynb  # Modified asset-holding version
│   │
│   └── Archive/                               # Earlier/experimental versions
│       ├── transfer_model.ipynb               # Archived base transfer model
│       ├── Parent_OLG.ipynb                   # Overlapping-generations parent model
│       ├── ConSavLabor_college.jl             # Archived Julia module
│       ├── consavlabor.jl                     # Archived Julia module
│       └── issue.md                           # Notes on modeling issues
│
├── test codes /                               # Scratch notebooks and experiments
│   ├── test.ipynb
│   ├── new.ipynb
│   ├── familyyyyyy.ipynb
│   ├── transfer_model.ipynb
│   └── ConSavLabor_college_AR1.jl
│
├── docs/                                      # Documentation
│   ├── SLSQP_algorithm.md                     # SLSQP algorithm math and pseudocode
│   └── Flat_policy_function.md                # Notes on flat policy function issues
│
├── Thesis_code.ipynb                          # Top-level thesis notebook
├── README.md                                  # This file
└── .gitignore
```

---

## Model Descriptions

### `ConSavLabor/` — Core Consumption-Saving and Labor Models

**`consumption_saving.ipynb`**

Implements a basic **T-period consumption-saving model** using backward induction and simulation. Includes visualizations of policy and value functions, along with counterfactual analysis for different income and wealth scenarios.

**`ConSavLabor.ipynb`**

Implements a dynamic **consumption-saving-labor model** with endogenous human capital accumulation. Solves using backward induction, simulates individual behavior, and includes counterfactual analysis for taxes, wages, preferences, and initial wealth.

**`ConSavLabor_AR1.ipynb`**

Extends the consumption-saving-labor model with a **persistent AR(1) wage shock**. Uses Tauchen discretization for the AR(1) process and Gauss-Hermite quadrature for integration.

**`ConSavLabor_stochastic.ipynb`**

Solves and simulates a finite-horizon $T$-period consumption-saving model with endogenous labor supply and **i.i.d. transitory wage shocks**. At each period $t$, the agent chooses consumption $c_t$ and labor $\ell_t$ to maximize lifetime utility under a stochastic budget constraint.

---

### `ConSavLabor_college/` — College Choice and Belief Heterogeneity

**`ConSavLabor_college.ipynb`**

Implements a dynamic model of **college and labor supply decisions** over the life cycle. The agent chooses between college or the labor market at age 18, then makes optimal decisions over consumption, saving, labor supply, and human capital accumulation.

**`ConSavLabor_college_AR1.ipynb`**

Extends the college model with a **persistent AR(1) wage process**. Solves for separate college and work policy functions and simulates life-cycle outcomes under wage uncertainty.

**`ConSavLabor_college_SE.ipynb`**

Extends the college model to allow **heterogeneous subjective beliefs** about college returns. Solves for optimal policies under each belief type and simulates outcomes to study how misperceptions affect college attendance and lifetime earnings.

**`ConSavLabor_college_retire.ipynb`**

Extends the college model to include a **mandatory retirement phase**. After the working lifecycle, the agent transitions to retirement where income comes from savings and a pension.

---

### `Family Model/` — Family Decision-Making

**`family.ipynb`**

Implements a **T-period dynamic family model** with endogenous investment in child human capital. Each period, the family chooses consumption, labor supply, child care time, and education expenditure to maximize lifetime utility.

**`parent_child_model.ipynb`**

Implements a **T-period dynamic family model** where a parent and child jointly decide on parental consumption, labor supply, child care time, child's study time, and education expenditure via a cooperative interaction weighted by the child's bargaining parameter.

---

### `Combined Models/Full model/` — Full Integrated Parent-Child Lifecycle Model

See [Combined Models/Full model/README.md](Combined%20Models/Full%20model/README.md) for detailed documentation.

The full model combines the parent-child family decision structure with the child's full college/work lifecycle, including AR(1) shocks, CRRA utility, progressive taxation, and retirement. Four notebook variants explore different utility specifications and wage processes.

---

### `Combined Models/Child & Parent part/` — Intermediate Combined Models

**`Family_with_asset.ipynb`**

Combines parent and child decisions in a single model with a **regime switch at $t=7$** (adolescence to early adulthood). Tracks assets, human capital, and care decisions across both regimes.

**`family_only_HC.ipynb`**

Simplified version focusing on the **human capital accumulation channel** without full asset dynamics.

**`modeified_family_with_asset.ipynb`**

Modified version with alternative asset-grid specifications and boundary conditions.

---

## Optimization Library

This project uses NLopt's SLSQP solver for interior optimization at each grid point. For details about the algorithm, including pseudocode and math, see [docs/SLSQP_algorithm.md](docs/SLSQP_algorithm.md). Notes on flat policy function issues are in [docs/Flat_policy_function.md](docs/Flat_policy_function.md).

---

## Dependencies

All notebooks are written in **Julia** and run in Jupyter via the IJulia kernel. Key packages:

| Package | Purpose |
|---|---|
| `NLopt` | Nonlinear optimization (SLSQP) |
| `Interpolations` / `Dierckx` | Grid interpolation |
| `QuantEcon` | Tauchen discretization, utility routines |
| `FastGaussQuadrature` | Gauss-Hermite quadrature for shock integration |
| `Plots` / `StatsPlots` | Visualization |
| `Parameters` | Struct keyword constructors |
| `Base.Threads` | Parallelization over grid points |

---

## Author

Ali Iranpour — TeIAS Thesis, 2025–2026
