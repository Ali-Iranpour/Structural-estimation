## Problem Statement

In this dynamic optimization problem, we aim to determine why the policy function with respect to human capital $hc_t$ is **flat**, i.e., the optimal choices of consumption $c_t$, education investment $e_t$, and labor supply $h_t$ do **not** depend on $hc_t$. Let’s explore this step-by-step. I solved the simple framework  to show this.

---

## 1. Value Function and Constraints

The value function is:

$$
V_t(a_t, k_t, hc_t) = \max_{c_t, e_t, h_t} \left\{ \phi_1 \log c_t + \phi_2 \log (1 - h_t) + \phi_3 \log hc_t + \beta V_{t+1}(a_{t+1}, k_{t+1}, hc_{t+1}) \right\}
$$

**Budget constraint:**

$$
0 = (1 - \text{TR}) w_t h_t - c_t - e_t
$$

**Human capital evolution:**

$$
hc_{t+1} = R\, e_t^{\sigma_2} hc_t^{\sigma_3}
$$

where $a_t$ and $k_t$ are state variables (assets and capital), $w_t$ is the wage (exogenous), $\text{TR}$ is the tax rate, $R$ is a constant, and $\sigma_2$, $\sigma_3$ are elasticities.



## 2. Substituting the Budget Constraint

Solving for $c_t$:

$$
c_t = (1 - \text{TR}) w_t h_t - e_t
$$

Plug back into the value function:

$$
V_t(a_t, k_t, hc_t) = \max_{e_t, h_t} \left\{ \phi_1 \log \left[ (1 - \text{TR}) w_t h_t - e_t \right] + \phi_2 \log (1 - h_t) + \phi_3 \log hc_t + \beta V_{t+1}\left(a_{t+1}, k_{t+1}, R\, e_t^{\sigma_2} hc_t^{\sigma_3}\right) \right\}
$$



## 3. First-Order Conditions (FOCs)

**For $e_t$:**

$$
\frac{\partial}{\partial e_t} = -\frac{\phi_1}{(1 - \text{TR}) w_t h_t - e_t} + \beta \frac{\partial V_{t+1}}{\partial hc_{t+1}} \cdot \sigma_2 \frac{R e_t^{\sigma_2} hc_t^{\sigma_3}}{e_t} = 0
$$

Using $c_t = (1 - \text{TR}) w_t h_t - e_t$ and $hc_{t+1} = R e_t^{\sigma_2} hc_t^{\sigma_3}$, this becomes:

$$
\frac{\phi_1}{c_t} = \beta V_{t+1}'(hc_{t+1}) \cdot \sigma_2 \frac{hc_{t+1}}{e_t}
$$

**For $h_t$:**

$$
\frac{\partial}{\partial h_t} = \frac{\phi_1 (1 - \text{TR}) w_t}{(1 - \text{TR}) w_t h_t - e_t} - \frac{\phi_2}{1 - h_t} = 0
$$

$$
\frac{\phi_1 (1 - \text{TR}) w_t}{c_t} = \frac{\phi_2}{1 - h_t}
$$



## 4. Guessing the Value Function Form

Assume the value function is linear in $\log hc_t$:

$$
V_t(a_t, k_t, hc_t) = A_t + B_t \log hc_t + \text{other terms}
$$

Then:

$$
V_{t+1}'(hc_{t+1}) = \frac{B_{t+1}}{hc_{t+1}}
$$

Substitute into the FOC for $e_t$:

$$
\frac{\phi_1}{c_t} = \beta \frac{B_{t+1}}{hc_{t+1}} \cdot \sigma_2 \frac{hc_{t+1}}{e_t}
$$

$$
e_t = \frac{\beta B_{t+1} \sigma_2}{\phi_1} c_t
$$

Let $k = \frac{\beta B_{t+1} \sigma_2}{\phi_1}$, so $e_t = k c_t$.


## 5. Solving the System

From the budget constraint:

$$
c_t + e_t = (1 - \text{TR}) w_t h_t
$$

$$
c_t (1 + k) = (1 - \text{TR}) w_t h_t
$$

$$
c_t = \frac{(1 - \text{TR}) w_t h_t}{1 + k}, \qquad e_t = \frac{k (1 - \text{TR}) w_t h_t}{1 + k}
$$

From the FOC for $h_t$:

$$
\frac{\phi_1 (1 - \text{TR}) w_t}{\frac{(1 - \text{TR}) w_t h_t}{1 + k}} = \frac{\phi_2}{1 - h_t}
$$

$$
\frac{\phi_1 (1 + k)}{h_t} = \frac{\phi_2}{1 - h_t}
$$

$$
h_t = \frac{\phi_1 (1 + k)}{\phi_1 (1 + k) + \phi_2}
$$



## 6. Determining $B$ Using the Envelope Condition

By the envelope theorem:

$$
V_t'(hc_t) = \frac{\phi_3}{hc_t} + \beta V_{t+1}'(hc_{t+1}) \cdot \sigma_3 \frac{hc_{t+1}}{hc_t}
$$

Since $V_t'(hc_t) = \frac{B_t}{hc_t}$:

$$
\frac{B_t}{hc_t} = \frac{\phi_3}{hc_t} + \beta \frac{B_{t+1}}{hc_{t+1}} \cdot \sigma_3 \frac{hc_{t+1}}{hc_t}
$$

$$
B_t = \phi_3 + \beta B_{t+1} \sigma_3
$$

In a stationary environment ($B_t = B$):

$$
B = \frac{\phi_3}{1 - \beta \sigma_3}
$$

Thus,

$$
k = \frac{\beta \sigma_2 \phi_3}{\phi_1 (1 - \beta \sigma_3)}
$$

which is a constant.



## 7. Final Policy Functions

The optimal policy functions are:

$$
h_t^* = \frac{\phi_1 (1 + k)}{\phi_1 (1 + k) + \phi_2}
$$

$$
c_t^* = \frac{(1 - \text{TR}) w_t h_t^*}{1 + k}
$$

$$
e_t^* = \frac{k (1 - \text{TR}) w_t h_t^*}{1 + k}
$$

**Since $k$ is constant and $w_t$ is exogenous (not a function of $hc_t$), $h_t^*$, $c_t^*$, and $e_t^*$ do not depend on $hc_t$. Therefore, the policy functions are flat in $hc_t$.**