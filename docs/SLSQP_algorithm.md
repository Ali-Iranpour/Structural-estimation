# SLSQP Algorithm

This document explains the **Sequential Least Squares Quadratic Programming (SLSQP)** algorithm, its mathematical formulation, and step-by-step pseudocode as implemented in this repository.

---

## **Problem Statement**

Given an optimization problem:

- **Objective:**  
  Minimize \( f(x) \)
- **Subject to:**
  - **Equality constraints:**  
    \( h_j(x) = 0 \), for \( j = 1, \ldots, m \)
  - **Inequality constraints:**  
    \( g_k(x) \geq 0 \), for \( k = 1, \ldots, p \)
  - **Bounds:**  
    \( l_i \leq x_i \leq u_i \)

---

## **Algorithm Description**

**SLSQP** is a powerful method for constrained nonlinear optimization. At each iteration, it approximates the objective function by a quadratic model and the constraints by linear models, solving a quadratic programming (QP) subproblem to propose a search direction.

SLSQP is especially useful when your problem includes both equality and inequality constraints and the functions involved are smooth (continuously differentiable).

---

## **Algorithm Steps**

### **1. Initialization**
- **Start with an initial guess:**  
  \( x^{(0)} \)
- **Set iteration counter:**  
  \( k = 0 \)
- **Choose an initial Hessian approximation:**  
  \( B^{(0)} \) (usually the identity matrix)

---

### **2. Repeat Until Convergence**

#### **2.1. Linearize Constraints**
At the current iterate \( x^{(k)} \), approximate the constraints using a first-order Taylor expansion:

- **Equality constraints:**  
  \[
  h_j(x) \approx h_j(x^{(k)}) + \nabla h_j(x^{(k)})^T (x - x^{(k)}) = 0
  \]

- **Inequality constraints:**  
  \[
  g_k(x) \approx g_k(x^{(k)}) + \nabla g_k(x^{(k)})^T (x - x^{(k)}) \geq 0
  \]

---

#### **2.2. Quadratic Approximation to Objective**
Approximate the objective function \( f(x) \) near \( x^{(k)} \) by a quadratic model:

\[
q(d) = f(x^{(k)}) + \nabla f(x^{(k)})^T d + \frac{1}{2} d^T B^{(k)} d
\]
where \( d = x - x^{(k)} \).

---

#### **2.3. Quadratic Programming (QP) Subproblem**
Solve the following QP to find the search direction \( d^{(k)} \):

\[
\begin{align*}
\min_{d} \quad & \nabla f(x^{(k)})^T d + \frac{1}{2} d^T B^{(k)} d \\
\text{subject to:} \quad & h_j(x^{(k)}) + \nabla h_j(x^{(k)})^T d = 0 \\
                        & g_k(x^{(k)}) + \nabla g_k(x^{(k)})^T d \geq 0 \\
                        & l_i \leq x_i^{(k)} + d_i \leq u_i
\end{align*}
\]

---

#### **2.4. Line Search**
- Select a step size \( \alpha^{(k)} \) to ensure a sufficient decrease in the objective and feasibility:
  \[
  x^{(k+1)} = x^{(k)} + \alpha^{(k)} d^{(k)}
  \]
- Ensure that the new iterate \( x^{(k+1)} \) satisfies all constraints and bounds.

---

#### **2.5. Update Hessian Approximation**
- Update the Hessian approximation \( B^{(k)} \) (e.g., using the BFGS quasi-Newton update).

---

#### **2.6. Convergence Check**
- Check whether:
  - The gradient norm \( \|\nabla f(x^{(k+1)})\| \) is below a chosen tolerance
  - All constraint violations are below tolerance
- If not converged, set \( k = k+1 \) and repeat.

---

## **Pseudocode**

```pseudo
Initialize x^(0), B^(0), k = 0
while not converged:
    // 1. Linearize constraints at x^(k)
    // 2. Quadratic approximation of f(x) at x^(k)
    // 3. Solve QP subproblem to find search direction d^(k)
    // 4. Line search: choose step size α^(k)
    x^(k+1) = x^(k) + α^(k) d^(k)
    // 5. Update Hessian approximation B^(k)
    // 6. Check for convergence
    if converged:
        break
    k = k + 1
return x^(k+1)
```

## References

- Kraft, D. (1988). *A software package for sequential quadratic programming*. DFVLR Obersfaffeuhofen Report.
- Nocedal, J. & Wright, S. J. (2006). *Numerical Optimization* (2nd Ed.). Springer.