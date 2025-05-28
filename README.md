## File Structure
```
├── ConSavLabor/                            # Core consumption-saving and labor models
│   ├── consumption_saving.ipynb            # Basic T-period consumption-saving model
│   ├── ConSavLabor.ipynb                   # Labor supply with endogenous human capital
│   └── ConSavLabor_stochastic.ipynb        # Labor model with i.i.d. wage shocks
│
├── ConSavLabor_college/                    # College choice and belief heterogeneity models
│   ├── ConSavLabor_college.ipynb           # College vs. work model with lifecycle decisions
│   └── ConSavLabor_college_SE.ipynb        # Model with heterogeneous beliefs about college returns
│
├── FamilyModel/                            # Family decision-making and child human capital investment
│   ├── family.ipynb                        # Dynamic family model with education and care decisions
│   └── parent_child_model.ipynb            # Dynamic family model with parent and child interaction
│
├── README.md                                # Project documentation
├── .gitignore                               # Ignore list (e.g., .DS_Store, temp files)
```
---

## Model Descriptions

### `ConSavLabor/` — Core Consumption-Saving and Labor Models

**`consumption_saving.ipynb`**

Implements a basic **T-period consumption-saving model** using backward induction and simulation. Includes visualizations of policy and value functions, along with counterfactual analysis for different income and wealth scenarios.

**`ConSavLabor.ipynb`**

Implements a dynamic **consumption-saving-labor model** with endogenous human capital accumulation. Solves the model using backward induction, simulates individual behavior, and includes counterfactual analysis for taxes, wages, preferences, and initial wealth.

**`ConSavLabor_stochastic.ipynb`**

Solves and simulates a finite-horizon, $T$-period consumption-saving model with endogenous labor supply and i.i.d. transitory wage shocks. At each period $t = 1, \ldots, T$, the agent chooses consumption $c_t$ and labor supply $\ell_t$ to maximize lifetime utility. The wage in each period depends on a transitory shock $\varepsilon_t$, which enters the budget constraint.

---

### `ConSavLabor_college/` — College Choice and Belief Heterogeneity

**`ConSavLabor_college.ipynb`**

Implements a dynamic model of **college and labor supply decisions** over the life cycle. The agent chooses between attending college or entering the labor market at age 18, and then makes optimal decisions over consumption, saving, labor supply, and human capital accumulation.

**`ConSavLabor_college_SE.ipynb`**

Extends the college model to allow **heterogeneous beliefs about college returns**. Solves for optimal policies and simulates life-cycle choices and outcomes for each belief type.

---

### `FamilyModel/` — Family Decision-Making

**`family.ipynb`**

Implements a **T-period dynamic family model** with endogenous investment in child human capital. Each period, the family chooses consumption, labor supply, child care time, and education expenditure to maximize lifetime utility, balancing adult well-being and child development.

**`parent_child_model.ipynb`**

Implements a **T-period dynamic family model** where a parent and child jointly decide on parental consumption, labor supply, child care time, child's study time, and education expenditure. These choices maximize lifetime utility, balancing parental well-being and child development through a cooperative interaction weighted by the child's bargaining parameter.

---
## Optimization Library


This project implements several optimization algorithms. For details about the SLSQP algorithm, including the pseudocode and math, see [docs/SLSQP_algorithm.md](docs/SLSQP_algorithm.md).
