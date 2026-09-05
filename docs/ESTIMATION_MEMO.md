---
title: "Estimation: the parameters, and how each is identified"
subtitle: "Parent block"
date: "6 September 2026"
geometry: margin=2.5cm
fontsize: 11pt
---

# Purpose

This records the implemented specification, the moments intended to inform the
parameters, and the distinction between estimated and calibrated quantities. It also
sets out the open question about the technology and welfare-weight age profiles.
The proposed extensions and robustness exercise below have not been implemented.

# 1. The model

## Preferences

The parents' period utility, with $l_{p,t} = 1 - h_{p,t} - \tau_{p,t}$:

$$U_{p,t} = \phi_1 \frac{c_{p,t}^{1-\rho}}{1-\rho}
          + \phi_2 \frac{l_{p,t}^{1-\eta}}{1-\eta}
          + \phi_3 \ln HC_t$$

Parental leisure nets out **both** work and time with the child: each uses one unit of
the parent's time budget, while only work earns a wage. The child's period utility, with
$l_{c,t} = 1 - \tau_{p,t} - i_{c,t}$:

$$U_{c,t} = \lambda_1 \ln l_{c,t} + \lambda_2 \ln HC_t$$

The family's joint utility, with the parents' welfare weight $\tilde\mu_t$:

$$U_{f,t} = \phi_1 \frac{C_t^{1-\rho}}{1-\rho}
          + \phi_2 \frac{l_{p,t}^{1-\eta}}{1-\eta}
          + \tilde\alpha_1 \ln l_{c,t}
          + \tilde\alpha_2 \ln HC_t$$

$$\tilde\alpha_1 = (1-\tilde\mu_t)\,\lambda_1,
\qquad
\tilde\alpha_2 = \tilde\mu_t\,\phi_3 + (1-\tilde\mu_t)\,\lambda_2$$

So the weight on the child's leisure is set entirely by the child's welfare share, while
the weight on skill mixes the parents' valuation $\phi_3$ and the child's $\lambda_2$.

The welfare weight is $\tilde\mu_t = 1$ for $t < 6$; the child has no separate study
choice in those periods. Under the current calibration it declines thereafter:

$$\tilde\mu_t = \mu_0 + \mu_1\,(t-5), \qquad t \ge 6$$

## Human capital

$$HC_{t+1} = R_t \;\tau_{p,t}^{\sigma_{1,t}} \; e_{p,t}^{\sigma_{2,t}} \;
             HC_t^{\sigma_{3,t}} \; i_{c,t}^{\sigma_{4,t}}$$

$$\sigma_{j,t} = \exp\!\big(\sigma_{j0} + \sigma_{j1}(t-1)\big), \quad j = 1,2,3$$

$$\sigma_{4,t} = \begin{cases}
0, & t < 6\\[2pt]
\exp\!\big(\sigma_{40} + \sigma_{41}(t-5)\big), & t \ge 6
\end{cases}$$

$\tau_p$ is parental time, $e_p$ monetary investment, and $i_c$ the child's own study
time. Before $t=6$, the implemented production function omits the study-time factor.
From $t=6$, its elasticity uses $(t-5)$, so the first active value is
$\exp(\sigma_{40}+\sigma_{41})$.

**The zero restriction applies to the early entries of `sigma_4_vector`, not to
$\sigma_{41}$.** The code treats $t$ as child age, solves ages 1–17, and activates study
at age 6: “zero through age 6 inclusive” would describe a different boundary.
There is no attribution to Sahber of a restriction $\sigma_{41}=0$.

Currently $\sigma_{41}=0.02$, so the elasticity rises after activation. Self-productivity
is calibrated at $\sigma_{3,t}=\exp(-0.90)\approx0.407$.

## Budget

$$a_{t+1} = (1+r)a_t + y + \lambda (w_t h_{p,t})^{1-\tau} - c_{p,t} - e_{p,t},
\qquad a_{t+1} \ge 0$$

with the HSV/Benabou progressive tax; $\tau$ is progressivity and $\lambda$ the level.

## Terminal value at separation

$$V_p^{\text{term}} = \psi\,\ln k + \kappa_{\text{term}}\,\ln a$$

where $k$ is the child's human capital at 18 and $a$ the parents' retained assets.
This is the parents' terminal component. The full continuation value also incorporates
the child's lifecycle value, the transfer decision, and the college/work comparison;
it is not just the expression above.

# 2. What changed

**The preference weights are time-invariant.** $\phi_1,\phi_2,\phi_3,\lambda_1,\lambda_2$
were age profiles $\phi_{j0} + \phi_{j1}(t-1)$; they are now scalars.

**$\phi_1 = 1$ and $\lambda_1 = 1$.** These are the chosen normalisations in the current
specification. Neither is estimated.

**$\psi = 0$.** The parents no longer place a separate terminal weight on the child's
skill; they value it through the flow term $\phi_3 \ln HC_t$ and through altruism.

**Initial human capital comes from the data.** $HC_1$ was an arbitrary $U(0,1)$ draw. The
model's human capital is now measured in the units of the Woodcock–Johnson PCA composite,
and the initial draw is taken from the data: the composite is not administered before age
3, so the mean and standard deviation of $\ln HC$ are fitted log-linearly on ages 3–17
and extrapolated back, giving $\ln HC_1 \sim N(5.953,\,0.067^2)$ — a median of
approximately 385 and a 10–90 range of approximately 353–419.

The timing issue is resolved. The fit is now evaluated at **child age 1**, which is the
child age of the first simulation column: the family stage runs $t = 1 \ldots 17$ over
child ages 1–17 and has no age-0 period. The code previously placed the age-0
extrapolation, $N(5.929,\,0.070^2)$, in that column, understating initial skill by
0.024 log points, or 2.4% in levels.

# 3. The parameters and the moments intended to identify them

| Parameter | Role | Main informative moments / current status |
|---|---|---|
| $\phi_1$ | weight on consumption | **fixed at 1** (normalisation) |
| $\phi_2$ | weight on parental leisure | mean **hours worked** |
| $\phi_3$ | parents' weight on the child's skill | **parental time** and **monetary investment** |
| $\lambda_1$ | child's weight on own leisure | **fixed at 1** (normalisation) |
| $\lambda_2$ | child's weight on own skill | the child's **own study time** |
| $R_0$ | level of HC productivity | early and late **log HC**, jointly with investment moments |
| $\sigma_{10},\sigma_{11}$ | level and age slope of parental-time elasticity | early and late **parental time**, jointly with HC |
| $\sigma_{20},\sigma_{21}$ | level and age slope of monetary-investment elasticity | early and late **monetary investment**, jointly with HC |
| $\sigma_{40}$ | log-level parameter of study-time elasticity | early and late **child study time**, jointly with HC |
| $\sigma_{41}$ | age slope of log study-time elasticity | **fixed at 0.02** |
| $\mu_1$ | slope of the welfare weight | **fixed at −0.04**; see §5 |

The current SMM estimates **nine parameters against ten moments**. The estimated
parameters are $\phi_2,\phi_3,\lambda_2,R_0,\sigma_{10},\sigma_{11},\sigma_{20},
\sigma_{21},\sigma_{40}$. The targets are mean consumption and hours, plus early and
late parental time, monetary investment, child study time, and log HC.

These are joint identification arguments, not one-to-one assignments. For example,
$\phi_2$ affects the leisure–work trade-off, but hours also respond to other parameters.
Both valuation and technology can change investment **and therefore resulting HC**.
HC moments add information that may help distinguish these channels; they do not,
by their presence alone, prove separate identification.

The strength of these arguments depends on matching data and model age coverage and
aggregation, and on the residual weights. In particular, dividing a log-HC residual
by the target's log level makes its weight depend on the units used to measure HC.

# 4. For the next step

These are not estimated yet. Listed with candidate informative moments.

| Parameter | Role | Candidate informative moments |
|---|---|---|
| $\kappa_{\text{term}}$ | weight on the parents' retained assets in their terminal value | **parental net worth after the child leaves** — family assets at child age 17, and post-separation wealth for families whose child has finished college |
| $\kappa_0$ | level of the psychic cost of college | the **overall college enrolment rate** |
| $\kappa_\theta$ | ability gradient in the psychic cost (negative) | the **enrolment gradient in child test scores** — enrolment by achievement quantile |
| $\kappa_{\text{ParEd}}$ | parental-education shift (negative) | the **enrolment gap by parental education** |

The three psychic-cost parameters all move enrolment. The level, ability gradient,
and parental-education gap provide three distinct sources of variation. The overall
rate alone cannot regularly identify all three, and three moments alone do not
guarantee that their effects are sufficiently independent.

$\kappa_{\text{term}}$ governs the retention–transfer trade-off, making retained wealth
a particularly relevant moment. It can also affect enrolment through transfers.

Their present values are transported from Colas (Table 2) by **sign and ratio only** —
$\kappa_{\text{ParEd}}/\kappa_\theta = 0.205$ is hers. The levels are not transportable:
her utility is $c^{1-\gamma}/(1-\gamma)$ at $\gamma = 1.9$ with consumption in dollars,
ours is $\rho = 1.5$ with $c$ of order 0.1–5.

# 5. Open question: the technology and welfare-weight age profiles

**Currently neither $\mu_1$ nor $\sigma_{41}$ is estimated.** Their values are
$\mu_1=-0.04$ and $\sigma_{41}=0.02$; $\sigma_{40}$ is estimated. The two slope
parameters are calibrated assumptions. The resulting study-time profile is nevertheless
endogenous, and its early and late means are already targeted by SMM.

The earlier claim that $\mu_1$ and $\sigma_{41}$ necessarily enter as a single
combination was incorrect. To see the distinction, set $s=t-5>0$. With $\mu_0=1$,

$$1-\tilde\mu_t=-\mu_1s,$$

and the technology-to-leisure-weight ratio is

$$\mathcal R_t=\frac{\exp(\sigma_{40}+\sigma_{41}s)}{(-\mu_1)s\lambda_1}.$$

For $\mu_1<0$,

$$\log\mathcal R_t=\sigma_{40}+\sigma_{41}s-\log(-\mu_1)-\log s-\log\lambda_1.$$

In this ratio, $\sigma_{41}$ changes the age slope, while $\mu_1$ changes a level
term. The level combination instead involves $\sigma_{40}-\log(-\mu_1)$.
Even that observation is not a proof of non-identification in the full model:
$\mu_1$ also changes the family's HC valuation and its continuation incentives.

For an interior study choice in the logarithmic-leisure region, the condition is

$$\frac{(1-\tilde\mu_t)\lambda_1}{l_{c,t}}
=\beta_t\,\mathbb E_t[V_{HC,t+1}]\,HC_{t+1}\,
\frac{\sigma_{4,t}}{i_{c,t}}.$$

The continuation derivative, resulting HC, and other choices all respond to the
parameters. Constraints can also bind. A ratio alone therefore cannot establish
which parameters the observed age profile identifies.

The available choices are:

1. **Keep both fixed, as currently implemented.** State their calibrated values
   explicitly and interpret the estimated parameters conditional on them.
2. **Estimate $\sigma_{41}$ and retain a stated calibration for $\mu_1$.** An optional
   calibration rule $\tilde\mu_{17}=0.5$ gives $\mu_1=-1/24\approx-0.0417$;
   the current value gives $\tilde\mu_{17}=0.52$. This is a proposed rule, not an
   instruction attributed to Sahber.
3. **Estimate $\mu_1$ and retain a stated calibration for $\sigma_{41}$.** The present
   value is $0.02$. Nothing in the early-age zero restriction requires changing it
   to zero.

Adding either parameter would give ten parameters and ten moments; identification
would still need evidence. Adding both while retaining all nine current parameters
would give eleven parameters and ten moments, so the moment Jacobian could not have
full column rank. That is a restriction of the present moment set, not proof of
an inherent equivalence between the two parameters.

No parameter choice or calibration is changed by this memo.

# 6. Additional robustness exercise from the paper

**10/10 — Plot how all estimated parameters respond to each target moment.**

Following the identification diagnostics in Sections 4.4 and 5.4 of
*structural-robustness-Jan26.pdf*, change one target moment at a time, hold the other
targets fixed, and re-estimate **all nine estimated parameters jointly** at every
perturbation. Plot each parameter estimate against the perturbed target value.
This measures how estimation reallocates a change in the data across parameters.

Use the agreed baseline target definitions and weighting matrix. For each of the ten
targets, choose a small range around its baseline, preferably expressed in standard
errors of the estimated moment. These are standard errors of the sample moment,
not standard deviations of individual observations. If they are unavailable, report
the chosen perturbations explicitly without giving them a confidence-interval
interpretation.

Keep the weighting matrix, residual scaling, random draws, parameter bounds, and
solver settings fixed across perturbations. Reuse the cached child solution because
the current nine parameters affect only the parent block. Nearby solutions can supply
starting points, but retain alternative starts to check for a different minimum;
validate the resulting estimates on the final estimation grid.

Produce ten sets of nine parameter curves, with the baseline target and estimate
marked. Record objective values, moment residuals, solver failures, and binding
parameter bounds alongside the plots. Large movements, jumps, or boundary solutions
flag sensitivity that needs explanation. The fixed $\mu_1$ and $\sigma_{41}$ are not
estimated-parameter curves in the current specification.

The exercise is a conditional sensitivity diagnostic, not a confidence region or
proof of global identification. It is proposed here and has not been run. This is
the only additional robustness exercise adopted from the paper in this memo.
