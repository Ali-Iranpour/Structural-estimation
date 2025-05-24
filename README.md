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
│   └── family.ipynb                         # Dynamic family model with education and care decisions
│
├── README.md                                # Project documentation
├── .gitignore                               # Ignore list (e.g., .DS_Store, temp files)
```

---

## Model Descriptions

### 🔧 `ConSavLabor/` — Core Consumption-Saving and Labor Models

#### `consumption_saving.ipynb`
Implements a basic **T-period consumption-saving model** using backward induction and simulation. Includes visualizations of policy and value functions, along with counterfactual analysis for different income and wealth scenarios.

#### `ConSavLabor.ipynb`
Implements a dynamic **consumption-saving-labor model** with endogenous human capital accumulation. Solves the model using backward induction, simulates individual behavior, and includes counterfactual analysis for taxes, wages, preferences, and initial wealth.

#### `ConSavLabor_stochastic.ipynb`
Solves and simulates a finite-horizon $begin:math:text$ T $end:math:text$-period consumption-saving model with endogenous labor supply and **i.i.d. transitory wage shocks**. At each period $begin:math:text$ t = 1, \\dots, T $end:math:text$, the agent chooses consumption $begin:math:text$ c_t $end:math:text$ and labor supply $begin:math:text$ \\ell_t \\in [0,1] $end:math:text$ to maximize lifetime utility. Wages depend on a transitory shock $begin:math:text$ \\varepsilon_t $end:math:text$, modifying the budget constraint.

---

### `ConSavLabor_college/` — College Choice and Belief Heterogeneity

#### `ConSavLabor_college.ipynb`
Implements a dynamic model of **college and labor supply decisions** over the life cycle. The agent chooses between attending college or entering the labor market at age 18, and then makes optimal decisions over consumption, saving, labor supply, and human capital accumulation.

#### `ConSavLabor_college_SE.ipynb`
Extends the college model to allow **heterogeneous beliefs about college returns**. Solves for optimal policies and simulates life-cycle choices and outcomes for each belief type.

---

### `FamilyModel/` — Family Decision-Making

#### `family.ipynb`
Implements a **T-period dynamic family model** with endogenous investment in child human capital. Each period, the family chooses consumption, labor supply, child care time, and education expenditure to maximize lifetime utility, balancing adult well-being and child development.