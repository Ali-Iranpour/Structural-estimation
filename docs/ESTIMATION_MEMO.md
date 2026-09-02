---
title: "Estimation plan: parameters, identification, and moments"
subtitle: "Parent block, SMM by TikTak"
date: "30 August 2026"
geometry: margin=2.5cm
fontsize: 11pt
---

# Summary

This memo records the estimation as it now stands: which parameters are estimated,
what identifies each one, which data moments are targeted, and the three places where
the specification you gave needed a decision. It closes with the parameters left for
the next step and what would identify them.

Everything below is implemented and the model solves. Nothing has been estimated yet
under the new specification — that run comes after you have seen this.

# 1. What changed in the specification

**Preference weights are time-invariant.** `phi_1, phi_2, phi_3, lambda_1, lambda_2`
were age profiles (`phi_j0 + phi_j1*(t-1)`) held in per-period vectors. They are now
scalars. Utility is defined only up to *relative* weights, so two of the five must be
normalised: **`phi_1 = 1` and `lambda_1 = 1`**.

**`psi_terminal = 0`.** This is the weight on the child's human capital in the parent's
terminal value at separation, `psi*log(k) + kappa*log(a)`. At zero the parent values the
child's skill only through the flow term `phi_3*log(HC)` and through altruism `omega`.

> Worth knowing: `psi_terminal` had been raised 1.0 → 4.0 precisely because a low
> terminal weight made the last period value skill far less than every earlier one, and
> `tau_p` collapsed at t = 17. **I checked, and it does not recur.** At `psi = 0` the
> model solves and `tau_p` runs 0.570 → 0.239 at t = 13 → 0.314 at t = 17. The flow term
> and altruism carry it.

**Human capital is now in the data's units.** `HC` *is* the Woodcock–Johnson PCA
composite (a W-score, roughly 300–600), not an abstract index on `[0.001, 10]`. This was
necessary — see §3.

**Initial human capital comes from the data.** `sim_hc_init` was `Uniform(0,1)`, which is
arbitrary. It is now lognormal, fitted to the PCA and extrapolated back to age 0.

# 2. Estimated parameters and what identifies each

Nine parameters against ten moments.

| Parameter | Identified by | Data moment |
|---|---|---|
| `phi_2` | parents' willingness to give up leisure for income | mean **hours worked** |
| `phi_3` | how much parents value the child's skill | **parental time** and **monetary investment**, jointly with the HC level |
| `lambda_2` | how much the *child* values its own skill | the child's **own study time** |
| `R_0` | the level of the HC technology (TFP) | the **HC level** |
| `sigma_1_0` | elasticity of HC to parental **time** | mean `t_p`, ages 1–9 |
| `sigma_1_1` | its age slope | mean `t_p`, ages 10–17 |
| `sigma_2_0` | elasticity of HC to **money** | mean `e_p`, ages 1–9 |
| `sigma_2_1` | its age slope | mean `e_p`, ages 10–17 |
| `sigma_4_0` | elasticity of HC to the child's **own study** | mean `i_c`, ages 6–17 |

**Fixed, not estimated:** `phi_1 = 1` and `lambda_1 = 1` (normalisations);
`sigma_3` (self-productivity, 0.407); `mu_1` (see §3).

## Why the human-capital moments are essential

Valuation and technology are otherwise collinear. `phi_3` and `lambda_2` say how much
parent and child *value* skill; `R_0`, `sigma_1`, `sigma_2`, `sigma_4` say how
efficiently skill is *produced*. **Both raise investment.** Looking only at `t_p`, `e_p`
and `i_c`, no combination of moments can separate them — the optimizer would sit on a
ridge and return whatever its starting point was near.

What separates them is the resulting **HC level**: technology raises it, valuation does
not. That moment only became usable once HC was placed in the data's units, which is why
the rescaling in §1 was not cosmetic.

# 3. Three places the specification needed a decision

## 3.1 `mu_1` cannot be estimated alongside `sigma_4_1`

Your reasoning — that `mu_1` is identified from `phi_3` and `lambda_2`, because only
`phi_3` operates before age 6 and all three after — is right in direction. The sharper
statement is that with `lambda_1 = 1` the child's effective weight on its own leisure is
exactly `(1 - mu_t)`, so **`lambda_2` sets the level of study time and `mu_1` sets its
slope in age.**

But the child's study first-order condition is driven by the ratio

$$\frac{\sigma_{4,t}}{(1-\mu_t)\,\lambda_1}, \qquad \sigma_{4,t}=\exp(\sigma_{40}+\sigma_{41}(t-1))$$

and `sigma_41` and `mu_1` both bend that ratio with age. **The study-time profile cannot
identify both.** One must go.

I dropped **`sigma_4_1`** and kept `mu_1` fixed: `mu_1` is a welfare weight pinned by the
model's structure, whereas `sigma_4` is a technology parameter that can reasonably be
held flat in age. `sigma_4` is therefore constant in `t`. If you would rather estimate
`sigma_4_1`, `mu_1` must first be fixed by an explicit rule — and I would want that rule
from you rather than guessing it.

## 3.2 Extrapolating the PCA back to age 0

The test is not administered before age 3, so both the mean and the SD of log HC are
fitted log-linearly on ages 3–17 and extrapolated:

$$\text{mean log HC} = 5.9290 + 0.02392\,\text{age} \quad (R^2=0.835)$$
$$\text{sd log HC} = 0.0698 - 0.00310\,\text{age} \quad (R^2=0.638)$$

giving `log HC_0 ~ Normal(5.9290, 0.0698)` — a median of **376** and a 10–90 range of
**344–411**. The simulated draw reproduces this (median 375.6, p10 344, p90 412).

Log-linear rather than quadratic, on two grounds: it is the same functional form the
production function uses, and a quadratic — though it tracks ages 3–17 slightly better —
bends to **303** at age 0 against the linear **376**, a 24% difference from extrapolating
three years beyond its support.

## 3.3 The model over-produces human capital by ~50%

Putting HC in data units made a pre-existing error visible:

| child age | 1 | 5 | 9 | 13 | 17 |
|---|---|---|---|---|---|
| **model HC** | 376 | **815** | 753 | 717 | **803** |
| **data** | 385 | 423 | 466 | 513 | 564 |

This is not a rescaling artefact. The cascade faithfully preserved the old dynamics, and
the old model grew HC by 5.6× where the data grows by 1.5×. The error was always present;
it was invisible while HC had no units.

`R_0` is the parameter that sets that level, so **it has been added to the estimated
set** — nine parameters, still over-identified against ten moments.

# 4. The moments

All in the model's own units: one model unit is \$10,000/year, time is a share of the
112-hour non-sleep week per parent, HC is a log W-score.

| Moment | Source | Target | N |
|---|---|---|---|
| `mean_c_p` | `cons_exhous_real_w99` | 3.1577 | 6,742 |
| `mean_h_p` | `(wh_mom + wh_dad)/2 / 112` | 0.3070 | 15,665 |
| `mean_t_p_early` | `par_time_tot / 112`, ages 1–9 | 0.4672 | 475 |
| `mean_t_p_late` | `par_time_tot / 112`, ages 10–17 | 0.3232 | 590 |
| `mean_e_p_early` | `m_method2_final_w99`, ages 1–9 | 0.3532 | 8,178 |
| `mean_e_p_late` | `m_method2_final_w99`, ages 10–17 | 0.4414 | 7,182 |
| `mean_i_c_early` | `study_hrs / 112`, ages 6–9 | 0.0386 | 171 |
| `mean_i_c_late` | `study_hrs / 112`, ages 10–17 | 0.0500 | 584 |
| `mean_hc_early` | `x_gach` (log PCA), ages 3–9 | 6.0802 | 253 |
| `mean_hc_late` | `x_gach` (log PCA), ages 10–17 | 6.2558 | 549 |

Two notes. The child's study moment starts at **age 6**, not 1: there is no child decision
before `T_CHILD_VOICE = 6`, so averaging in ages 1–5 would target a number the model
cannot produce. And `t_p` and `e_p` are split early/late because a single mean cannot
separate an age *slope* from a *level*.

## A caveat that must travel with any estimate of `phi_2`

`t_p` is measured with `par_time_tot` — active **plus** nearby/supervisory presence. That
measure does not fit an exhaustive time budget: per parent,
`leisure + work + own active care = 112.00` exactly, but with `par_time_tot` it is
**133.25**, twenty-one hours over. Since the model enforces `l_p + h_p + t_p = 1`,
targeting `h_p` and `t_p` **forces** model leisure to about 33 hrs/wk against the 59
hrs/wk the same dataset measures. That gap lands in `phi_2`. **The estimated `phi_2`
should not be read as a clean taste for leisure.** Switching the target to per-parent
active time closes the identity exactly, if you prefer that.

# 5. Optimiser settings

TikTak (Arnoud, Guvenen & Kleineberg 2022), at the paper's own ratio:
**N = 1,000 Sobol' points, N\* = 100 restarts** (`N* = 0.1N`).

The Sobol stage is parallel (~8 minutes on 20 workers). The 100 restarts are
**sequential by construction** — restart *j* starts from the best point found by
`1..j-1` — so they dominate: budget roughly **25 hours**. If that is too long, cut
restarts rather than Sobol points; more Sobol points are nearly free and give the local
stage better seeds.

# 6. Next step: parameters not yet estimated

These are deliberately left for a second stage, with what would identify each.

| Parameter | What it is | Would be identified by |
|---|---|---|
| `kappa_terminal` | weight on the parents' retained assets in their terminal value | **parental net worth after the child leaves** — `assets_real` at child age 17, and post-separation wealth for families whose child has finished college |
| `kappa_0` | level of the psychic cost of college | the **overall college enrolment rate** |
| `kappa_theta` | ability gradient in the psychic cost (negative: ability lowers it) | the **enrolment gradient in child test scores** — enrolment by `g_ACH` quantile |
| `kappa_ParEd` | parental-education shift (negative) | the **enrolment gap by parental education** — `BothCollege` vs not |

The three `kappa` terms of the psychic cost all move enrolment, so they need three
*distinct* enrolment moments: the level, the slope in ability, and the gap by parental
education. Estimating them against the overall rate alone would not identify them.

`kappa_terminal` is separate: it governs how much wealth parents retain rather than
transfer, so it is identified by post-separation wealth, not by enrolment.

Their current values are transported from Colas (Table 2) by **sign and ratio only** —
`kappa_ParEd/kappa_theta = 0.205` is hers; the levels are not, because her utility is
`c^(1-gamma)/(1-gamma)` at `gamma = 1.9` with consumption in dollars while ours is
`rho = 1.5` with `c` of order 0.1–5.

# 7. Open questions

1. **`mu_1`.** It is currently fixed at −0.04, giving a parental weight of 0.52 at t = 17.
   Should it be pinned by an explicit rule — equal weights at 18, or zero parental weight
   at 18 — or estimated in place of `sigma_4_1`?
2. **`t_p` measure.** Keep `par_time_tot`, accepting that `phi_2` absorbs a 21-hour
   time-budget inconsistency, or switch to per-parent active time, which closes the
   identity exactly?
3. **Sample.** The targets are built from all one-child families. The model assumes
   parents aged 26 at birth, which matches the *cohort* file (father 25–30). The
   restriction moves consumption by −7.7%, monetary investment by −17.8% and assets by
   −28%, while time moments barely move. Should the targets be rebuilt on the cohort?
