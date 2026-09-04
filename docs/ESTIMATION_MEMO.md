---
title: "Estimation: the parameters, and how each is identified"
subtitle: "Parent block"
date: "30 August 2026"
geometry: margin=2.5cm
fontsize: 11pt
---

# Purpose

This records the specification you gave, what identifies each parameter, and one
question where two parameters turn out to compete for the same variation. The model
equations are reproduced first so the identification arguments can be read against them.

Everything below is implemented and the model solves.

# 1. The model

## Preferences

The parents' period utility, with $l_{p,t} = 1 - h_{p,t} - \tau_{p,t}$:

$$U_{p,t} = \phi_1 \frac{c_{p,t}^{1-\rho}}{1-\rho}
          + \phi_2 \frac{l_{p,t}^{1-\eta}}{1-\eta}
          + \phi_3 \ln HC_t$$

Parental leisure nets out **both** work and time with the child, so time with the child is
priced at the same rate as time working. The child's period utility, with
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

The welfare weight is $\tilde\mu_t = 1$ for $t < 6$ — during childhood the parents decide
alone — and declines thereafter:

$$\tilde\mu_t = \mu_0 + \mu_1\,(t-5), \qquad t \ge 6$$

## Human capital

$$HC_{t+1} = R_t \;\tau_{p,t}^{\sigma_{1,t}} \; e_{p,t}^{\sigma_{2,t}} \;
             HC_t^{\sigma_{3,t}} \; i_{c,t}^{\sigma_{4,t}},
\qquad \sigma_{j,t} = \exp\!\big(\sigma_{j0} + \sigma_{j1}(t-1)\big)$$

$\tau_{p}$ is parental time, $e_{p}$ monetary investment, $i_{c}$ the child's own study
time, which enters only from $t = 6$. Self-productivity $\sigma_3 < 1$ is required for a
bounded solution.

## Budget

$$a_{t+1} = (1+r)a_t + y + \lambda (w_t h_{p,t})^{1-\tau} - c_{p,t} - e_{p,t},
\qquad a_{t+1} \ge 0$$

with the HSV/Benabou progressive tax; $\tau$ is progressivity and $\lambda$ the level.

## Terminal value at separation

$$V^{\text{term}} = \psi\,\ln k + \kappa_{\text{term}}\,\ln a$$

where $k$ is the child's human capital at 18 and $a$ the parents' retained assets.

# 2. What changed

**The preference weights are time-invariant.** $\phi_1,\phi_2,\phi_3,\lambda_1,\lambda_2$
were age profiles $\phi_{j0} + \phi_{j1}(t-1)$; they are now scalars.

**$\phi_1 = 1$ and $\lambda_1 = 1$.** Utility is defined only up to *relative* weights, so
two of the five must be normalised. $\phi_1$ is fixed at 1 as you specified;
$\lambda_1$ is fixed at 1 for the same reason on the child's side.

**$\psi = 0$.** The parents no longer place a separate terminal weight on the child's
skill; they value it through the flow term $\phi_3 \ln HC_t$ and through altruism.

> $\psi$ had previously been raised from 1.0 to 4.0 because a low terminal weight made
> the final period value skill far less than every earlier one, and $\tau_p$ collapsed at
> $t = 17$. **I checked whether that recurs at zero. It does not** — the model solves and
> $\tau_p$ runs $0.570 \to 0.239$ at $t=13 \to 0.314$ at $t=17$. The flow term and
> altruism carry what the terminal weight used to.

**Initial human capital comes from the data.** $HC_0$ was an arbitrary $U(0,1)$ draw. The
model's human capital is now measured in the units of the Woodcock–Johnson PCA composite,
and $HC_0$ is drawn from the data: the composite is not administered before age 3, so the
mean and standard deviation of $\ln HC$ are fitted log-linearly on ages 3–17 and
extrapolated back to age 0, giving $\ln HC_0 \sim N(5.929,\,0.070)$ — a median of 376 and
a 10–90 range of 344–411. The simulated draw reproduces this (median 375.6, p10 344,
p90 412).

# 3. The parameters and what identifies each

| Parameter | Role | Identified by |
|---|---|---|
| $\phi_1$ | weight on consumption | **fixed at 1** (normalisation) |
| $\phi_2$ | weight on parental leisure | mean **hours worked** |
| $\phi_3$ | parents' weight on the child's skill | **parental time** and **monetary investment** |
| $\lambda_1$ | child's weight on own leisure | **fixed at 1** (normalisation) |
| $\lambda_2$ | child's weight on own skill | the child's **own study time** |
| $\mu_1$ | slope of the welfare weight | see the question in §5 |

$\phi_2$ works through the intratemporal condition: it prices leisure against the
after-tax wage, so mean hours pin it.

$\phi_3$ and $\lambda_2$ are the two valuation parameters. $\phi_3$ raises both forms of
parental investment — time and money — because both enter the same production function;
$\lambda_2$ raises the child's own study time, which is the margin the child controls
once $\tilde\mu_t < 1$.

**One consequence worth stating.** Valuation and technology both raise investment: a
higher $\phi_3$ and a more productive technology look the same in $\tau_p$ and $e_p$
alone. What separates them is the *resulting level of human capital* — technology raises
it, valuation does not. This is why putting $HC$ into the data's units mattered: it makes
the skill level usable as a moment, and without it $\phi_3$ and $\lambda_2$ cannot be
told apart from the production-function elasticities.

# 4. For the next step

These are not estimated yet. Listed with what would identify each.

| Parameter | Role | Would be identified by |
|---|---|---|
| $\kappa_{\text{term}}$ | weight on the parents' retained assets in their terminal value | **parental net worth after the child leaves** — family assets at child age 17, and post-separation wealth for families whose child has finished college |
| $\kappa_0$ | level of the psychic cost of college | the **overall college enrolment rate** |
| $\kappa_\theta$ | ability gradient in the psychic cost (negative) | the **enrolment gradient in child test scores** — enrolment by achievement quantile |
| $\kappa_{\text{ParEd}}$ | parental-education shift (negative) | the **enrolment gap by parental education** |

The three $\kappa$ terms of the psychic cost all move enrolment, so they need three
*distinct* enrolment moments: the level, the slope in ability, and the gap by parental
education. Estimated against the overall rate alone they would not be separately
identified.

$\kappa_{\text{term}}$ is a different object: it governs how much wealth parents retain
rather than transfer, so post-separation wealth identifies it, not enrolment.

Their present values are transported from Colas (Table 2) by **sign and ratio only** —
$\kappa_{\text{ParEd}}/\kappa_\theta = 0.205$ is hers. The levels are not transportable:
her utility is $c^{1-\gamma}/(1-\gamma)$ at $\gamma = 1.9$ with consumption in dollars,
ours is $\rho = 1.5$ with $c$ of order 0.1–5.

# 5. Question

**Can $\mu_1$ and $\sigma_{41}$ both be estimated?**

Your suggestion was that $\mu_1$ is identified from $\phi_3$ and $\lambda_2$ — only
$\phi_3$ operates before age 6, all three after. That is right in direction, and with
$\lambda_1 = 1$ it sharpens: the child's effective weight on its own leisure is exactly
$(1-\tilde\mu_t)$, so **$\lambda_2$ sets the level of the child's study time and $\mu_1$
sets its slope in age**.

The difficulty is that the child's first-order condition for study time is driven by

$$\frac{\sigma_{4,t}}{(1-\tilde\mu_t)\,\lambda_1},
\qquad \sigma_{4,t} = \exp\!\big(\sigma_{40} + \sigma_{41}(t-1)\big)$$

and $\sigma_{41}$ and $\mu_1$ both bend that ratio with age. **The age profile of study
time therefore cannot identify both** — they enter as a single combination, and an
optimiser given both would sit on a ridge and return whichever point it started nearest.

One of the two must be fixed. Provisionally I have held $\sigma_4$ flat in age
($\sigma_{41} = 0$) and left $\mu_1$ at its current value, on the reasoning that $\mu_1$
is a welfare weight fixed by the structure of the problem while $\sigma_4$ is a
technology parameter that can reasonably be constant. But the choice is yours, and there
are three coherent options:

1. **Fix $\mu_1$ by an explicit rule and estimate $\sigma_{41}$.** For instance, set
   $\mu_1$ so the parents' weight reaches $0.5$ at $t = 17$ (equal weights as the child
   leaves), giving $\mu_1 = -1/24$; or so it reaches $0$ at $t = 17$ (the child fully
   decides at separation), giving $\mu_1 = -1/12$.
2. **Estimate $\mu_1$ and hold $\sigma_4$ flat.** This is what is currently implemented.
   It treats the age profile of study time as evidence about the child's growing
   autonomy rather than about the technology.
3. **Fix both** from outside the estimation, and use the study-time profile as an
   untargeted check on whether the model reproduces it.

If you have a rule in mind for $\mu_1$, option 1 is the one that buys the most: it makes
the technology's age profile an estimated object rather than an assumption.
