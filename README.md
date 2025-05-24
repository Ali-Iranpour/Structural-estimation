#### `consumption_saving.ipynb`

Implements a basic T-period consumption-saving model using backward induction and simulation. Includes visualizations of policy and value functions, along with counterfactual analysis for different income and wealth scenarios.

#### `ConSavLabor.ipynb`

Implements a dynamic T-period consumption-saving-labor model with endogenous human capital accumulation. Solves the model using backward induction, simulates individual behavior, and includes counterfactual analysis for taxes, wages, preferences, and initial wealth.

#### `ConSavLabor_stochastic.ipynb`
This code solves and simulates a finite-horizon $(T)$-period consumption-saving model with endogenous labor supply **and** i.i.d. transitory wage shocks. At each period $t = 1, \dots, T$, the agent chooses consumption $c_t$ and labor supply $\ell_t \in [0,1]$ to maximize lifetime utility. In contrast to the no-uncertainty case, wages now depend on a transitory shock $\varepsilon_t$, which modifies the budget constraint.


#### `ConSavLabor_college.ipynb`

This notebook implements a dynamic programming model of **college and labor supply decisions** over the life cycle. The agent chooses between attending college or entering the labor market at age 18, and then makes optimal consumption, saving, and labor supply decisions based on their human capital accumulation, wages, and preferences.


#### `family.ipynb`

Implements a T-period dynamic family model with endogenous human capital investment. The family chooses consumption, labor supply, child care time, and education expenditure each period to maximize lifetime utility, which depends on consumption, leisure, and the child’s human capital.
