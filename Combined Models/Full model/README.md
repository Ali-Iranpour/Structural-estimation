# Full Model — Integrated Parent-Child Lifecycle Model

This folder contains the complete integrated model that combines a **T-period parent-child family decision problem** with the **child's full lifecycle** (college choice, working years, and optional retirement). The parent observes the child's human capital and decides on transfers, care time, and education expenditure, while the child solves a separate lifecycle problem that feeds back into the parent's terminal value.

---

## Folder Structure

```
Full model/
├── transfer_model.ipynb            # Base model: log utility, i.i.d. shocks
├── transfer_model_AR1.ipynb        # AR(1) wage shocks variant
├── transfer_CRRA.ipynb             # CRRA utility + retirement phase
├── transfer_CRRA_wage.ipynb        # CRRA utility + stochastic wages
│
├── ConSavLabor_college_AR1.jl      # Julia module: child's AR(1) college/work lifecycle
├── ConSavLabor_college_ret.jl      # Julia module: child's lifecycle with retirement
│
└── plots/
    ├── Baseline/                   # Baseline simulation plots
    │   └── PolicyFunctions/        # Policy function plots
    ├── Parameters/                 # Counterfactual plots by parameter (timestamped runs)
    │   └── terminal_assets/        # Terminal asset distribution counterfactuals
    ├── Res_vs_Exp/                 # Residual vs. expectation counterfactuals
    │   ├── positive/               # High welfare / high expectation scenarios
    │   │   └── Adulthood/
    │   └── negative/               # Low welfare / low expectation scenarios
    │       └── Adulthood/
    ├── SE/                         # Subjective expectation counterfactuals
    ├── Slide/                      # Presentation-quality plots
    │   └── 2025-10-14_084102/
    └── college decision/           # College vs. work counterfactuals
        └── Adulthood/
```

---

## Model Overview

The full model has two nested components:

### 1. Child's Lifecycle (Julia modules)

The child solves a standard **dynamic college-or-work problem** with:
- Binary college/work choice at period $t=0$
- Endogenous human capital accumulation $k_{t+1} = f(k_t, \ell_t, e_t)$
- Consumption-saving with budget constraint
- Persistent AR(1) or i.i.d. wage shocks
- Optional retirement phase (in `ConSavLabor_college_ret.jl`)

The child's lifecycle is solved first (backward induction) and its value function is passed as the **terminal condition** to the parent's problem.

### 2. Parent's Problem (notebooks)

The parent solves a $T=18$-period dynamic optimization over:

| Decision Variable | Description |
|---|---|
| $c_t$ | Parental consumption |
| $\ell_t$ | Parental labor supply |
| $s_t$ | Child's study time (chosen/allocated by parent) |
| $e_t$ | Education expenditure |
| $m_t$ | Parental care time |

**State variables:** assets $a_t$, parental human capital $k_t$, child human capital $HC_t$

**At the terminal period** $T$ (separation/college entry):
- Parent transfers assets to the child
- Child's continuation value enters the parent's terminal value with weight $\omega$

---

## Notebook Variants

### `transfer_model.ipynb` — Base Model
- **Utility:** Log utility for parent
- **Wage shocks:** i.i.d. transitory shocks (Gauss-Hermite quadrature)
- **Child module:** `ConSavLabor_college_AR1.jl`
- **Key parameters:** $T=18$, $\tau=0.25$, $r=0.03$, $\omega=1.0$

The baseline notebook. Solves the parent's problem, simulates $N$ households, and generates counterfactual plots for all structural parameters.

---

### `transfer_model_AR1.ipynb` — AR(1) Wage Shocks
- **Utility:** Log utility for parent
- **Wage shocks:** Persistent AR(1) process (Tauchen discretization)
- **Child module:** `ConSavLabor_college_AR1.jl`
- **Key parameters:** $T=18$, $\tau=0.25$, $\omega=0.7$

Extends the base model to allow **persistent wage risk**. The parent's state space gains a dimension for the persistent wage shock $p_t$.

---

### `transfer_CRRA.ipynb` — CRRA Utility with Retirement
- **Utility:** CRRA (constant relative risk aversion) for parent
- **Wage shocks:** i.i.d. transitory shocks
- **Child module:** `ConSavLabor_college_ret.jl` (includes retirement)
- **Key parameters:** $T=18$, $\tau=0.25$, $r=0.03$, $\omega=2.0$, `a_max=100`

Replaces log utility with CRRA and links to the child module that includes a retirement phase. Generates terminal-asset distribution plots and welfare counterfactuals.

---

### `transfer_CRRA_wage.ipynb` — CRRA with Stochastic Wages
- **Utility:** CRRA for parent
- **Wage shocks:** Stochastic wages (AR(1) or i.i.d.)
- **Child module:** `ConSavLabor_college_ret.jl`

Most general variant. Combines CRRA utility with stochastic wage dynamics.

---

## Julia Modules

### `ConSavLabor_college_AR1.jl`

Defines the `ConSavLaborCollege_AR1` struct and solver for the child's lifecycle problem. Key features:
- 5D solution arrays: `(T, Na, Nk, Np, Nt)` for college and work tracks
- Separate `solve_model_work!` and `solve_model_college!` functions
- Transfer/separation value functions: `sol_tr_college`, `sol_tr_work`
- Nonlinear and focused asset grids
- Tauchen AR(1) discretization for persistent wage shock $p_t$
- Gauss-Hermite quadrature for transitory shock $\varepsilon_t$

### `ConSavLabor_college_ret.jl`

Extends `ConSavLabor_college_AR1.jl` with a **mandatory retirement period** (`t_retire`). After retirement, the agent receives pension income and solves a pure consumption-saving problem.

---

## Key Parameters

| Parameter | Symbol | Description |
|---|---|---|
| `T` | $T$ | Parent problem horizon (= 18 child periods) |
| `tau` | $\tau$ | Labor income tax rate |
| `r` | $r$ | Interest rate |
| `beta` | $\beta$ | Discount factor |
| `rho` | $\rho$ | CRRA coefficient (or 1.0 for log) |
| `phi` | $\phi$ | Altruism / utility weight on child |
| `omega` | $\omega$ | Weight on child's lifecycle continuation value |
| `kappa_terminal` | $\kappa$ | Weight on parent's retained terminal assets |
| `psi_terminal` | $\psi$ | Weight on child's terminal human capital |
| `mu` | $\mu$ | Bargaining weight (parent vs. child) |
| `sigma_1t` | $\sigma_{1t}$ | Slope of parental care productivity |
| `sigma_2t` | $\sigma_{2t}$ | Slope of education expenditure productivity |
| `sigma_3t` | $\sigma_{3t}$ | Slope of HC persistence |
| `sigma_4t` | $\sigma_{4t}$ | Slope of child study time productivity |

---

## Solution Algorithm

Each notebook follows the same structure:

1. **Child solve:** Call `solve_model_work!` and `solve_model_college!` from the Julia module. This produces policy functions and value functions on a `(Na, Nk, Np, Nt)` grid.

2. **Parent backward induction:** For $t = T, T-1, \ldots, 1$:
   - At $t = T$ (separation): interpolate child's continuation value onto terminal assets; solve parent's terminal optimization.
   - For $t < T$: solve a constrained NLopt SLSQP problem at each grid point $(a_t, k_t, HC_t)$ given next-period interpolated value function.

3. **Simulation:** Draw $N$ households from initial distributions; simulate forward using optimal policy functions with drawn shocks.

4. **Counterfactuals:** Re-solve the model under alternative parameter values and compare simulated outcomes.

---

## Plots

The `plots/` directory organizes output by experiment type:

| Folder | Contents |
|---|---|
| `Baseline/` | Lifecycle paths, policy functions, terminal asset distributions under baseline calibration |
| `Parameters/` | Counterfactual sensitivity to each structural parameter (timestamped runs) |
| `Parameters/terminal_assets/` | Effect of parameters on terminal asset distribution |
| `Res_vs_Exp/` | Baseline vs. high/low child welfare scenarios |
| `SE/` | Subjective expectation vs. baseline |
| `Slide/` | Cleaned plots for presentations |
| `college decision/` | College attendance rates under counterfactuals |

---

## Usage

Open any notebook in Jupyter with the IJulia Julia kernel. The notebooks `include()` the relevant `.jl` module from the same directory before running the parent solver. Run cells sequentially — the child solve must complete before the parent backward induction begins.

```julia
# Example (transfer_model_AR1.ipynb)
include("ConSavLabor_college_AR1.jl")
child_model = ConSavLaborCollege_AR1(Na=50, Nk=50, Nt=10, ...)
solve_model_work!(child_model)
solve_model_college!(child_model)
# ... then parent backward induction ...
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `NLopt` | SLSQP nonlinear optimization |
| `Interpolations` / `Dierckx` | Multi-dimensional interpolation |
| `QuantEcon` | Tauchen AR(1) discretization |
| `FastGaussQuadrature` | Gauss-Hermite nodes and weights |
| `Plots` / `StatsPlots` | Visualization and PDF export |
| `Parameters` | `@with_kw` struct constructors |
| `Base.Threads` | Parallel grid-point loops |
| `Measures`, `LaTeXStrings` | Plot formatting |
