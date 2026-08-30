# The child's wage process and psychic cost

What is in the code, with the calibration and its sources, followed by the case for it
and the literature behind it. (This merges the former `WAGE_PROCESS.md` proposal, which
still said "nothing in code/ has been changed" long after it was implemented.)

Implemented in `code/src/child_lifecycle.jl`; `parent_family.jl`, `diagnostics.jl`
and `run_all.jl` follow. Baseline to roll back to: commit `cebcb83`.

---

## 1. What changed

| | Before | After |
|---|---|---|
| Wage | `w = w₀(1 + α·HC)·z`, `α = 0.08` | log-linear, below |
| Human capital | `HC' = HC + h`, `+2.0` per college year | **`HC' = HC`** — fixed at `θ` for life |
| College | an increment to the same stock | its own state `E`, paid through `β_E` in the wage |
| Psychic cost | `κ/(HC+1)^4` | `κ₀ + κ_θ·log θ + κ_ParEd·BothCollege` |
| Post-college work | `sol_*_college .= sol_*_work` | genuinely solved at `E = 1` |
| Hours ceiling | `clamp(k_max − capital, 1e-3, 1.0)` | `1.0` |
| `k_max` | 30 | 8 |

### The wage

```
ln w_t = ln w₀ + β_E·E + (α_θ + α_θE·E)·(log θ − m_θ)
                       + (γ₁ + γ₁E·E)·age + (γ₂ + γ₂E·E)·age²  + ln z_t
```

`θ` is childhood human capital at 18, from the parent block, **fixed for life**.
`E ∈ {0,1}` is college. `age = 17 + t`, so `t = 1` is age 18. `z` is the existing
AR(1), unchanged.

### The psychic cost

```
κ_X = κ₀ + κ_θ·log θ  +  κ_ParEd·BothCollege
```

A flow cost in each enrolled year, decreasing in ability.

---

## 2. Calibration and where each number comes from

| Parameter | Value | Source |
|---|---|---|
| `γ₁` | 0.0234 | Daruich & Fernández Table B3, high school |
| `γ₂` | −0.000199 | " |
| `γ₁E` | +0.0318 | Table B3, college minus high school |
| `γ₂E` | −0.000314 | " |
| `β_E` | −0.294 | Table B3 constants, 1.953 − 2.247 |
| `α_θ` | 0.654 | Daruich Table B4, high school |
| `α_θE` | +0.322 | Table B4, 0.976 − 0.654 (ratio 1.49) |
| `κ_θ` | −0.0342 | sign from Colas Table 2, level calibrated (§3) |
| `κ_ParEd` | −0.0070 | Colas ratio `κ_ParEd/κ_θ = 0.205`, level follows |
| `κ₀` | 0.0462 | calibrated (§3) |
| `ln w₀` | `log(w) − 0.4144` | normalization (§3) |
| `m_θ` | 0.724 | centring only; see §4 |

Daruich's Table B3 is PSID 1968–2016, quadratic in **biological age**, estimated
separately by education with a Heckman correction for selection into work. The
implied college premium runs **0.254 at age 22 to 0.511 at age 50**, which is the
fan-out the old specification could not produce.

`α_θ` is an **elasticity** with respect to childhood human capital, which is what
both papers estimate: they regress log wages on *log* ability. Colas Table 2 gives
0.31 / 0.47, the same 1.5 ratio at half the level; Daruich is preferred here as the
closer dataset.

---

## 3. Three numbers that are NOT from a paper

Stated plainly, because they look like calibration but are normalizations.

**`ln w₀`.** Fixes the units of the wage. Set so the mean simulated wage matches the
old specification at the same grids: 44.60 against 43.99, a 1.4% gap. This keeps
assets, consumption and the tax base on their existing scale, so the rest of the
calibration still applies. Re-derive by shifting `ln w₀` by
`log(target mean wage / simulated mean wage)`.

**`κ₀` and the level of `κ_θ`.** Colas's estimates are `κ₀ = 0.44`,
`κ_θ = −8.3×10⁻⁴`, `κ_ParEd = −1.7×10⁻⁴` (their starred entries are 10,000× the
parameter). The **levels cannot transport**: their utility is
`c^(1−γ)/(1−γ)` at `γ = 1.9` with consumption in dollars, ours is `ρ = 1.5` with `c`
of order 0.1–5, so a `κ` of 4×10⁻⁴ is meaningless on our scale. What transports is
the **sign** of both terms and the **ratio** `κ_ParEd/κ_θ = 0.205`. The level is set
to reproduce the total discounted psychic cost of the old `κ/(HC+1)^4` form over the
four college years, across the range of `θ` the model visits.

---

## 4. Why `α_θ` is an elasticity and not a per-SD effect

The first implementation standardized, `θ̃ = (log θ − m)/s`, so that `α_θ` would read
as a per-SD effect at 0.18. **It diverges.** The investment incentive is
`d ln w / d log θ = α_θ / s_θ`, and at the dispersion the parent block actually
produces, `sd(log θ) ≈ 0.2`, that is an elasticity near 0.9 — above both papers.
Parents over-invest without limit:

| iteration | `s_θ` in | mean log θ | sd log θ | mean θ | college |
|---|---|---|---|---|---|
| 1 | 1.000 | 0.724 | 0.198 | 2.11 | 9.0% |
| 2 | 0.198 | 1.349 | 0.168 | 3.91 | 13.0% |

Mean `θ` nearly doubles in one step, heading for the `k_max` ceiling, with no fixed
point. There is also a circularity: the wage sets parental investment, which sets
`θ`'s distribution, which sets the standardizers.

As an elasticity both problems disappear. `m_θ` is a pure level shift that `ln w₀`
absorbs, so it has no behavioural content and nothing has to be iterated.

⚠️ This is the concrete form of the warning in `WAGE_PROCESS.md` §6.5: `α_θ` and
`R_0` are not separately identified. Fixing `α_θ` from outside the model, as here,
is one of the two ways out; the other is to normalize `R_0` and estimate `α_θ`.

---

## 5. Implementation notes

**College is solved, not copied.** `solve_model_college!` now runs in two stages: the
graduate's working life as a work problem with `E = 1` written into the college
arrays from `t_college+1`, then the study years with that as the continuation. The
old code copied the high-school solution, so a graduate and a high-school worker at
the same stock faced an identical wage. No new array dimension was needed.

**`κ_ParEd` costs no state.** The term is additive in utility, constant across the
college years (`θ` is fixed), and does not interact with consumption. It therefore
shifts the college value by a closed-form annuity and leaves every college policy
unchanged, so `pared_value_offset` applies it once at the college-vs-work
comparison. This is exact, not an approximation, and it works *only* because human
capital no longer accumulates — under the old law of motion `κ/(k+1)^4` varied year
to year and could not be factored out.

**Beliefs now concern `β_E`.** `simulate_model_family_hetero!` keyed heterogeneous
beliefs on `college_boost`; they are now about the log wage premium, which is what
Bleemer (2018) measures. Every decision is still taken under the biased number. What
disappears is the reconciliation term `HC' = H̃C + b* + (T_E−1)(b* − b_m)`, which
existed only to reconcile a perceived *stock* with the true one over four years.
⚠️ **Caller change:** `child_models[m]` must now differ in `beta_E`, not
`college_boost`.

**Dead parameter.** `college_boost` no longer affects anything.

---

## 6. What this does to the results

Standalone child, old against new at identical grids, mean wage matched:

| | old | new |
|---|---|---|
| college share | 6.7% | 39.6% |
| mean wage | 43.99 | 44.60 |
| mean hours | 0.263 | 0.249 |
| mean consumption | 4.84 | 4.48 |
| mean assets | 10.12 | 8.63 |

The college share is the substantive change: college now pays a real premium that
rises with ability, and a graduate no longer forgoes four years of stock
accumulation. **`college_cost` and `κ₀` will need re-estimating** against the 25%
target; `college_cost` is already in the SMM parameter set.

### The psychic cost turns negative at high ability

`κ_X = 0.0462 − 0.0342·log θ` crosses zero at **θ ≈ 3.86**, which is inside the
support (mean θ is 3.47). Above that, college carries a psychic *benefit*: the ablest
find it intrinsically rewarding. Colas's functional form is linear in `log θ` and
permits this too, and it causes no numerical trouble since the term is an additive
constant in utility. But it is a property of the calibration, not a result, and
`κ₀` is the natural lever on the college margin: raising it to about 0.071 would keep
the cost non-negative across the whole grid at the cost of a lower college share.
**`κ₀` belongs in the SMM parameter set** alongside `college_cost`, which is already
there.

### Where θ ends up, and the grid

The higher return to childhood human capital raises parental investment. At
production grids `θ` is now distributed:

| p50 | p90 | p99 | p99.9 | max |
|---|---|---|---|---|
| 3.43 | 4.28 | 5.22 | 6.83 | 15.52 |

against a mean near 1.2 under the old specification. `k_max` was set to **8** on this,
which covers **99.92%** of children; the remaining 0.08% are handled by the existing
`Flat()` extrapolation, well inside the project's own 1% tolerance in
`check_solver_domain`.

The long max is a thin lognormal tail, not an instability: the HC technology is
log-linear with shocks, so the right tail is fat while the bulk is tight. It is worth
knowing that the tail is sensitive to `k_max` itself — raising it from 6 to 8 moved
the max from 6.53 to 15.52 while the mean did not move at all (3.47 → 3.45). The
child's `k_max` sets the domain of the terminal-value spline the parent optimises
against, and `terminal_value` is `ψ·log(HC) + κ·log(a)`, unbounded above, so what
stops investment for the highest-investing parents is a corner that moves with the
grid. Reading the max alone would overstate this; the percentiles are the right
summary.

Related, and pre-existing: 0.038% of the parent block's simulated HC exceeds the
**parent's own `hc_max = 6.0`** (`parent_family.jl:179`). Also a thin tail, also
within tolerance, but that ceiling is now closer to binding than it was.

Checks that pass: the college premium now **rises** in `θ` (0.358 → 0.566 across its
support, against 0.495 → 0.376 before); human capital is exactly constant over life;
the hours ceiling never binds; and the graduate and high-school value functions
genuinely differ.

---

## 7. Heterogeneous beliefs

Beliefs were about `college_boost`, the subjective human-capital increment per college
year. They are now about **`beta_E`**, the log college wage premium, which is what
Bleemer and Zafar (2018) actually elicit.

### The mapping

The survey elicits `rce`, a single believed ratio of college to high-school earnings
(1.5 = "50% more"). The model does not carry a single premium: it carries a profile,
`beta_E + gamma1E*age + gamma2E*age²`, worth 0.25 at 22 and 0.51 at 50. So a scalar
belief has to be anchored to a horizon. The anchor is the **career average over the
graduate's working life**, ages 22–68:

```
beta_E^m = beta_E* + [log(rce_m) − anchor*]
         = log(rce_m) − mean_a(gamma1E·a + gamma2E·a²)      = log(rce_m) − 0.7374
```

The true intercept cancels, so the mapping depends only on the interaction terms.
A believer's premium profile is the estimated one shifted in parallel: beliefs move the
**level**, the **shape** stays at the estimated one. `beta_E_from_rce(model, rce)` in
`child_lifecycle.jl` does this and is exact — the career average of a believer's own
profile reproduces `log(rce_m)` to machine precision.

This replaces `college_boost = (rce − 1)/0.4`, which inverted the old
`w0*(1 + alpha*HC)` wage at `HC = 0` to re-express the belief as a stock increment.
Under a log wage no inversion is needed: the belief is already in the model's units.

### The belief distribution is near-unbiased, which is a useful property

The existing draw is `rce = 1 + 2·Beta(2,5)` on [1,3], binned in 20 steps of 0.1.

| | value |
|---|---|
| `E[log rce]` | 0.432 |
| true career-average premium | 0.443 |
| `sd[log rce]` | 0.199 |

Beliefs are therefore **unbiased on average with genuine dispersion**, so the exercise
isolates the effect of belief *heterogeneity* rather than mixing it with a level bias.
Kept as is by decision. If you later want Bleemer and Zafar's underestimation result,
that is a deliberate downward shift of the mean, not a property of the current draw.

Verified: the college share rises monotonically in the believed premium, from 0% at
`rce = 1.1` to 67% at `rce = 2.7`.

### Notebook change

`college_boost_belief_bin` is replaced by `beta_E_belief_bin` in cells 40, 78, 79, 80
and the single-belief variant. Everything else in those cells is unchanged, including
copying `base_child`'s work solution into each belief model — that is the `E = 0`
high-school solution, which beliefs do not touch. `solve_model_college!` gives each
belief model its own `E = 1` graduate working life at its own `beta_E`.

```julia
# was:  college_boost_belief_bin = (rce_mid .- 1) ./ 0.4
#       college_boost_belief_bin = round.(college_boost_belief_bin, digits=3)
beta_E_belief_bin = round.(beta_E_from_rce(base_child, rce_mid), digits=4)

# and in the child-model loop:
#   college_boost = college_boost_belief_bin[m]   ->   beta_E = beta_E_belief_bin[m]
```

`psi_from_belief_linear` in those cells has been **deleted**. It was calibrated on the
`college_boost` scale (`b_min = 0.125`, `b_anchor = 1.8`), which no longer exists, and
it was already dead: `ERRORS.md` N5 records that `psi_terminal` is deliberately held
common across belief types, and every model was built with `psi_terminal = 4.0`
regardless. It also shadowed `m` immediately before `for m in 1:num_bins`. Removed from
cells 40, 78, 79 and 80; N5 updated to record it.


## Two constants that carry no behaviour, and one that carries all the level

Moved here from `child_lifecycle.jl` so the constructor reads; the code keeps a pointer.

**`m_theta = 0.724` is a centring constant only.** The wage carries
`alpha_theta·(log θ − m_theta)`, so `m_theta` shifts the level and nothing else. It is set
near `E[log θ]` from the parent block, so `lnw0` reads as the log wage of a child of average
childhood human capital. Changing it is exactly offset by `lnw0` and has no behavioural
content.

**`lnw0 = log(w) − 0.4144` is a pure normalisation, not a paper value.** It fixes the units
of the wage, set so the mean simulated wage matches the previous `w0·(1 + α·HC)`
specification at the same grids — which keeps assets, consumption and the tax base on their
existing scale, so the rest of the calibration still applies. If the profile parameters move,
re-derive it: shift `lnw0` by `log(target mean wage / simulated mean wage)`.

**The psychic-cost levels do NOT transport from Colas; the ratio does.** Colas Table 2 gives
`kappa_0 + kappa_theta·log θ + kappa_fem·female + kappa_ParEd·ParEdu` with (starred entries
10,000× the parameter) `kappa_0 = 0.44`, `kappa_theta = −8.3e-4`, `kappa_ParEd = −1.7e-4`.
The signs and the ratio transport; the levels cannot — their utility is `c^(1−γ)/(1−γ)` at
`γ = 1.9` with consumption in dollars, ours is `ρ = 1.5` with `c` of order 0.1–5, so a
`kappa` of 4e-4 is meaningless here. The level is instead set to reproduce the total
discounted psychic cost of the old `kappa/(HC+1)^4` form over the four college years, across
the range of `θ` the model actually visits. The ratio `kappa_ParEd / kappa_theta = 0.205` is
Colas's.

---

# Appendix — the case for this specification, and the literature behind it

Kept from the former `WAGE_PROCESS.md`, which argued for the change before it was made.
The proposal machinery (recommendation, problem list, provisional calibration, open
questions) is dropped: it is superseded by what is above and by [`SMM.md`](SMM.md). What
survives is the material that is still reference rather than history — the limitation the
paper has to state, how four related papers handle the adult block, and how measured
childhood skill connects to adult human capital.

## A. The limitation to state in the paper

One sentence, in the model section: *the model abstracts from on-the-job human capital
accumulation, so the dynamic cost of progressivity through forgone experience is not
captured; the tax's human-capital margin operates through parental investment and through
college enrolment.* If Sahber judges that channel essential to the tax result, the fallback
is §4 with experience in place of age — 5× the compute, and better run as a robustness check
than as the baseline.

## B. How four papers model wages and human capital after the child grows up

This section covers only the **adult** phase — from the end of schooling onward. All four
papers have a childhood skill-formation block upstream of it; what matters for the present
question is what they do with that skill once the agent enters the labour market.

### 5.0 The taxonomy

The first thing to note is that the four papers split evenly on the most basic question:
**does adult human capital accumulate at all?**

| | Adult human capital | Accumulates? | Education enters through | Ability enters through | Shock structure |
|---|---|---|---|---|---|
| **Adda et al. (2017)** | skill stock `x`, ≈ uninterrupted work experience | **Yes** — learning by doing, plus atrophy | occupation-specific intercept *and* slope | permanent type `f^P_i` | iid on log wage |
| **Lee & Seshadri (2019)** | Ben-Porath stock `h` | **Yes** — explicit time investment `n` | education-specific rental price `w_S` | learning ability `a`, inherited AR(1) | iid multiplicative on the stock ⇒ **permanent** |
| **Colas et al. (2021)** | none — a deterministic path | **No** | education-specific age polynomial *and* ability loading | `β_θ^e·log θ` | one draw `v^e*` at entry, then **nothing** |
| **Daruich & Fernández (2023)** | none — efficiency units | **No** | education-specific rental price, age profile *and* ability loading | `α^e·log θ^C` | AR(1) throughout, + out-of-work and superstar states |
| **This model (current)** | `HC`, one state for skill + college + experience | Yes — learning by doing, no depreciation | `+college_boost` to the same stock | same stock | AR(1) |

Two of the four do not accumulate adult human capital at all. That is worth knowing: it
means the proposal in §4 is not obliged to keep learning-by-doing, and equally that keeping
it is well precedented. What *no* paper does is what this model currently does — put
childhood skill, education and experience into a single additive stock with one return.

---

### 5.1 Adda, Dustmann and Stevens (2017), *The Career Costs of Children*

**Human capital.** Skills `x` accumulate by **+1 per year of full-time work and +0.5 per
year of part-time work**, and *depreciate when out of work*:

```
x_{i,t+1} = x_it · ρ(x_it, o_it),    ρ < 1
ρ(x,o) = ρ₁(o)·1{x∈[0,5)} + ρ₂(o)·1{x∈[5,7)} + ρ₃(o)·1{x∈[7,∞)}
```

Depreciation is thus **piecewise in the skill level** (nodes at 5 and 7 years, chosen from
reduced-form regressions) and occupation-specific. Because skills grow like experience but
decay out of work, "a skill level of `x` is equivalent to `x` years of *uninterrupted* work
experience."

**Wage.**

```
ln w_it = f^P_i + a_O(o_it) + a_X(o_it)·x_it + a_XX(o_it)·x²_it + η_it
```

with `f^P_i` a **permanent unobserved productivity type**, `a_O` an occupation intercept,
`a_X`, `a_XX` occupation-specific returns to skill, and `η_it` **iid**. They also estimate
a threshold `x̄` beyond which the marginal effect of skill is zero — profiles are flat after
about 15 years of accumulated experience.

**Their estimates (Table 3, p. 312):**

| | Routine | Abstract | Manual |
|---|---|---|---|
| Log wage constant `a_O` | 3.39 | 3.60 | 3.32 |
| Experience `a_X` | 0.100 | 0.090 | 0.123 |
| Experience² `a_XX` | −0.00382 | −0.00210 | −0.00463 |
| Atrophy at 3 yrs | −0.06% | −0.11% | −0.03% |
| Atrophy at 6 yrs | −0.50% | **−6.90%** | −3.45% |
| Atrophy at 10 yrs | −0.61% | −2.65% | −3.08% |

Two features are directly relevant to the proposal:

- **The high-return type has both a higher intercept and a flatter curvature**, so the gap
  fans out. Abstract jobs start 0.21 log points above routine, and their footnote 31 records
  that "after 10 years of uninterrupted work experience, wages in abstract jobs increase by
  about 2 percent more per additional year." This is exactly the `β_E` + `γ₁E, γ₂E`
  structure proposed in §4 — a level premium alone cannot reproduce it.
- **Depreciation peaks mid-career**, not at the start. Their reading is that interruptions
  are most costly "at career stages in which learning is intense or individuals compete for
  key workplace positions." A constant `δ` is a simplification of this, and a defensible one
  given there is no interruption margin here (§9).

For scale: Kim and Polachek (1994) estimate 2–5% annual atrophy, Albrecht et al. (1999)
about 2%, neither allowing occupation or career-stage variation.

**Connection to childhood skill: none — it is absorbed by a latent type.** Adda et al. have
no childhood block. Everything pre-market sits in `f^P_i`, one element of a four-dimensional
vector of unobserved characteristics (productivity/ability, taste for leisure, taste for
children, potential infertility). They group ability and taste for leisure together
precisely because "four-dimensional heterogeneity is very demanding in terms of
identification and computation," and identify the type distribution from the panel structure
of wages and choices rather than from any measured test score. This is the fourth logical
option and worth naming: **do not measure childhood skill at all, and let a discrete latent
type carry it.**

---

### 5.2 Colas, Findeisen and Sachs (2021), *Optimal Need-Based Financial Aid*

**Human capital: none.** There is no accumulation after schooling. Wages are a
deterministic function of education, ability and age, and the *only* uncertainty is a single
draw at labour-market entry: "we assume that agents do not know the value of `v^e*_i` at the
beginning of the model, but that its value is revealed as soon as the agents finish their
education and enter the labour market… After `v^e*_i` there is no further uncertainty about
an agent's wage path."

**Earnings** (their eq. 11, estimated first; wages are then backed out through the static
labour-supply FOC):

```
log y^e_it = β₀^es + β_θ^e·log θ_i + β^e_t1·t + β^e_t2·t² + β^e_t3·t³ + v^e*_i ,   e ∈ {H, G}
```

Everything is education-specific: the constant, the **ability loading**, and a **cubic age
polynomial**. Dropouts share the high-school coefficients but get their own constant.

**Their estimates (Table 2):**

| | Log AFQT `β_θ` | Female | Constant | Var(`v_i`) |
|---|---|---|---|---|
| College | **0.47** | −0.14 | 3.06 | **0.42** |
| High school | **0.31** | −0.25 | 7.11 | **0.36** |

Two contributions:

- **Ability–education complementarity.** `β_θ^G/β_θ^H = 1.52`. In their words, a ratio
  "larger than 1 … implies a complementary relationship between initial ability and
  education." This is the `α_θE > 0` of §4, and the direct answer to the sign problem in
  §3.2.
- **College is risky.** `v^e*` is unknown at enrolment and has a *higher* variance for
  college. Combined with risk aversion and larger parental transfers to college-goers, this
  is one of their explanations for the parental-income enrolment gradient. See open question
  §11.3.

Their footnote 31 is the mechanism that makes this matter for policy: "As the college wage
premium is increasing in ability, this implies that increase of tax payments of marginal
enrollees is increasing in parental income" — i.e. the ability loading drives the fiscal
externality gradient, which is what determines whether optimal aid is progressive.

**Connection to childhood skill: a regression, estimated outside the structural model.**
`θ` is the AFQT score, made comparable across cohorts and age-at-test using the method of
Altonji, Bharadwaj and Lange (2012). The bridge to adult outcomes is `β_θ^e`, obtained in
two steps: age coefficients from the NLSY79 (old enough to observe late-career earnings),
then the age-purged variable `log ỹ^e_it = log y^e_it − β^e_t·t − β^e_t2·t² − β^e_t3·t³`
regressed on gender and log AFQT in the NLSY97, **separately by education level**, by random
effects. They are explicit about why two datasets: individuals in the NLSY97 "are only
observed until their mid-30s," so "combining both data sets has proven to be a fruitful way
in the literature to overcome the limitations of each individual data set."

Their endogenous-ability extension also solves the *scale* problem directly, and it is worth
quoting because it is exactly the situation of a CDS test score. Having produced end-of-
childhood skill `θ̂` in the units of Agostinelli and Wiswall (2016), they write: "We assume
that our measures of ability `θ` is a linear projection of this log skill measure,
`θ = α₀ + α₁ ln θ̂`, where we choose `α₁` and `α₀` to match the mean and variance of our
AFQT measure." That is a two-parameter linear anchoring in logs — the same operation as the
`θ̃` standardization proposed in §4.

---

### 5.3 Lee and Seshadri (2019), *On the Intergenerational Transmission of Economic Status*

**Human capital: Ben-Porath**, from college (`j = 3`) onward:

```
h_{j+1} = ε_{j+1}·[ a·(n_j·h_j)^β + h_j ]
```

`a` is **learning ability**, fixed at birth; `n_j ∈ [0,1]` is time spent *accumulating*
human capital; `β` is a single exponent on both time and the stock; `ε_j` is an iid "market
luck" shock.

Two structural features distinguish this from the other three:

- **Accumulation is an explicit choice, not a by-product of working.** The agent allocates
  `n_j` to skill-building, and pays for it in forgone earnings:

  ```
  e_j = w_S · h_j · (1 − n_j)
  ```

  where `w_S` is the **education-specific skill rental price** (`S ∈ {HS, college}`). The
  college premium therefore lives entirely in the price of an efficiency unit, not in the
  wage equation's shape.
- **`ε` multiplies the entire stock**, so shocks to the growth rate of human capital become
  **permanent earnings shocks** — a unit root in `log h`, rather than the stationary AR(1)
  used here.

**Ability is inherited**, as an AR(1) across generations (their eq. 3):

```
log a' = (1 − ρ_a)(μ_a − σ²_a/2) + ρ_a·log a + η'
```

"capturing intergenerational persistence not explained by economic behaviour."

**Connection to childhood skill: an anchor, estimated jointly with everything else.** This
is the closest paper to the present model. Childhood human capital `h'_3` is produced by a
multi-stage CES technology in parental time and goods,

```
h'_3 = z·{ q₂X₂^φ₂ + (1−q₂)[q₁X₁^φ₁ + (1−q₁)(q₀X₀^φ₀)^{φ₁/φ₀}]^{φ₂/φ₁} }^{1/φ₂}
```

and is then **the initial condition of the adult Ben-Porath process** — the same `h`, in the
same units, with no separate mapping required. The whole bridge is the scalar `z`, which
"is an anchor that transforms children's human capital, which we will later proxy by test
scores in the data, into adult outcomes, which we will measure using earnings (Cunha et al.
2010; Del Boca et al. 2014)."

⚠️ Their footnote 9 is the warning that matters most for this project: "In practice, the
exact interpretation is slightly different, mainly because **it is not separately identified
from `q₀`**. In our context, in addition to transforming test scores into meaningful units,
`z` will also capture the productivity of initial investments." The anchor and the
technology's productivity parameter are not separately identified. See §6.5.

Worth recording for the thesis: they find "college is mostly selection and explains little
of life cycle inequality."

---

### 5.4 Daruich and Fernández (2023), *Universal Basic Income: A Dynamic Assessment*

**Human capital: none accumulated.** Wages are a rental price times endogenous efficiency
units (their eqs. 2 and 13):

```
w_ij = w_e · h^e_ij ,        h^e_ij = ε^e_j · ν^e_ij
ln ν^e_ij = α^e·ln θ^C_i + ζ^e_ij ,     ζ^e_ij = ρ^e·ζ^e_i,j−1 + η^e_ij ,  η ~ N(0, σ^e_η)
```

- `w_e` is the **education-group unit wage**, determined in **general equilibrium** by a
  CES production function over high-school and college labour — so the college premium is
  endogenous to the education distribution.
- `ε^e_j` is an **education-specific quadratic age profile**, estimated on PSID with year
  fixed effects and a **Heckman correction for selection into work**.
- `α^e` is the **education-specific return to cognitive skill** `θ^C`.
- `ζ` is an **AR(1) throughout working life**, discretized by **Rouwenhorst** — the same
  method used in this project.

Their two headline findings on this block corroborate Colas independently, on different
data (NLSY + PSID rather than NLSY79 + NLSY97):

> "the returns to skill are **1.5 times greater for college-educated workers** than
> high-school ones. College agents draw their **initial productivity from a distribution
> with a somewhat higher variance** than high-school agents, but shocks received later are
> similar."

So `α^e_G/α^e_H ≈ 1.5` against Colas's 1.52, and higher entry variance for college against
Colas's 0.42 vs 0.36. **Two independent estimates, two datasets, the same two conclusions.**

Three further features worth noting:

- **Skills are a vector** — cognitive `θ^C` and non-cognitive `θ^NC`, from Cunha, Heckman
  and Schennach (2010). Only **cognitive** skill enters wages; **both** enter the psychic
  cost of college.
- **Psychic cost of college** (their eq. 14): `κ(ξ,θ) = κ₀ + κ_θC·ln θ^C + κ_θNC·ln θ^NC + ξ`,
  with `ξ` normal and its mean depending on **parental education**. Directly comparable to
  this model's `κ/(HC+1)^4` — and see open question §11.2.
- **An out-of-work shock** (`ζ = −∞`, wage zero) whose probability rises with age and is
  higher for high-school workers; on re-entry the agent draws the *lowest* productivity
  state, implying wages 30–90% below the age-education group average. This is a scarring
  mechanism that reproduces Adda's atrophy without carrying a skill stock — an alternative
  to `δ` worth knowing about.
- Initial skills follow an AR(1) from parental skills in logs, as in Lee & Seshadri.

**Connection to childhood skill: a three-step residual regression.** The most explicitly
documented recipe of the four, and the closest template for what is proposed in §6:

1. **Age profiles from PSID.** `ε^e_j` estimated as a quadratic in the head-of-household's
   age, separately by education, with year fixed effects and a **Heckman correction for
   selection into work**: `w_it = b₀ + b₁Age + b₂Age² + b₃λ_it + n_t + ν_it`.
2. **Residual from NLSY.** "Armed with the age profile, we then use (13) to recover `ν^e_ij`
   as a residual in the NLSY data."
3. **Regress the residual on childhood skill, by education.** "an estimate of `α^e` is
   recovered by regressing the estimate of `ν^e_ij` against the log of cognitive skills as
   measured by the AFQT score."
4. The remaining residual `ζ` is fitted as an AR(1) by Minimum Distance (Rothenberg et al.
   1971) and discretized by Rouwenhorst.

They split datasets for a reason worth noting: "We need to use NLSY for this step since the
PSID in general does not have information that is pertinent for measures of skills such as
an AFQT score. The PSID, instead, is preferred for estimating the age profiles since the age
of the sample does not covary perfectly with the year of the survey." The skill
distribution's moments are taken directly from Cunha, Heckman and Schennach (2010), using
"the same normalization as Cunha et al. (2010) to be consistent."

---

### 5.5 What the four agree on

Stripping out what is specific to each paper's question, four things are common ground:

1. **Wages are log-linear**, never `1 + α·HC`. All four write `ln w` (or `ln y`) as a sum.
2. **Education shifts the profile, it does not add to a common skill stock.** In every
   paper education enters as its own object — an occupation/education intercept
   (Adda), an education-specific polynomial (Colas), a rental price (Lee & Seshadri,
   Daruich & Fernández) — never as an increment to the same variable that experience
   accumulates into.
3. **The return to ability is higher for the educated**, wherever both are present. Colas:
   1.52. Daruich & Fernández: ≈1.5. Adda's analogue is the occupation-specific `a_X`.
4. **The profile is concave in whatever drives it** — quadratic in experience (Adda,
   Daruich & Fernández), cubic in age (Colas), or Ben-Porath curvature (Lee & Seshadri).
   None is flat, as `w_vec` currently is.

The proposal in §4 is a hybrid: an accumulated experience stock with depreciation from
Adda, education-specific profiles and the ability interaction from Colas and
Daruich & Fernández, and the anchoring of childhood skill into earnings units from
Lee & Seshadri.

---

## C. Connecting measured childhood human capital to adult human capital

This is the open empirical problem: `HC` at age 17 will be measured from the PSID Child
Development Supplement, and it has to be turned into the `θ` that enters the adult wage
equation. Sahber's suggested starting point is to regress the child's initial wage on their
last-period human capital. This section assesses that, sets out what the four papers do
instead, and proposes a procedure.

### 6.1 Three distinct problems, often conflated

| | Problem | Consequence if ignored |
|---|---|---|
| **Scale** | A CDS test score has arbitrary units; `θ` in the model has units set by the HC technology | The coefficient is uninterpretable and not comparable to any published estimate |
| **Measurement error** | One test score is a noisy measure of latent skill | Attenuation — the estimated return to skill is biased toward zero |
| **Selection** | Education is chosen, and wages are observed only for workers | `E[v \| E=1] ≠ E[v \| E=0]`; the regression coefficient is not the structural parameter |

The scale problem is what the literature calls **anchoring** (Cunha and Heckman 2008; Cunha,
Heckman and Schennach 2010). It is the reason Lee and Seshadri carry `z` and Colas et al.
carry `α₀, α₁`.

### 6.2 What the four papers do

| Paper | Bridge from childhood skill to adult HC | Estimated where |
|---|---|---|
| **Adda et al.** | none — a permanent latent type `f^P_i` absorbs it | inside the structural model |
| **Lee & Seshadri** | anchor `z`; childhood HC *is* the initial adult stock, same units | inside, jointly — and **not separately identified** from `q₀` |
| **Colas et al.** | `β_θ^e·log θ`, θ = AFQT; plus `θ = α₀ + α₁ ln θ̂` to convert units | **outside**, by regression, separately by education |
| **Daruich & Fernández** | `α^e·log θ^C`, θ^C = AFQT | **outside**, three-step residual regression, separately by education |

Note the split: the two papers that *measure* childhood skill both estimate the wage loading
**outside** the structural model and impose it. Neither tries to estimate it jointly with
the skill-formation technology. §6.5 explains why that is not an accident.

### 6.3 Assessment of "regress initial wage on last-period HC"

The shape is right — it is what Colas and Daruich & Fernández do. Four things need changing
before it is usable.

**(a) "Initial wage" is the worst available left-hand side.** Entry wages are dominated by
search and match quality, and `z` here is near-permanent (ρ = 0.95), so a single entry wage
is mostly noise plus a permanent draw you cannot separate. Both papers regress on a *wage
residual purged of age effects* and pooled over the panel — Colas via a random-effects
estimator on all available years, Daruich & Fernández on `ν^e_ij` recovered as a residual.
Use the individual's permanent component (a random effect, or an average of age-purged log
wages over the observed years), not the first observation.

**(b) The four-year college gap is not a dummy problem.** Adding a college dummy to a pooled
regression imposes `β_θ^H = β_θ^G` — precisely the restriction §3.2 argues against, and the
one both Colas (1.52) and Daruich & Fernández (≈1.5) reject. **Estimate two separate wage
equations, one per education group.** The timing gap is then handled by the age (or
experience) profile inside each equation, not by the dummy. A dummy on its own would also
conflate the college premium with four years of forgone experience, since it compares a
graduate at 22 with a high-school worker who has been working since 18.

**(c) Selection is not fixable by the regression.** Education is chosen on exactly the
unobservables that enter the wage. Daruich & Fernández correct for selection into *work*
with a Heckman estimator, but selection into *education* is handled structurally, by the
model's taste shock. The practical implication for this project is in (d).

**(d) Treat the regression coefficient as a moment, not as the parameter.** Since the model
will be SMM-estimated anyway, the clean route is **indirect inference**: run the regression
on real data, run the *identical* regression on simulated data, and choose `α_θ, α_θE` so
the two coefficients match. The model then supplies its own selection, so the auxiliary
regression does not have to be causally interpretable — only reproducible. This is standard
practice and it dissolves (c) rather than patching it.

### 6.4 A concrete procedure for PSID + CDS

The CDS is well suited to this — better than the NLSY split Colas and Daruich & Fernández
were forced into — because CDS children are followed into adulthood within the same panel
(the Transition into Adulthood Supplement), so childhood scores and adult wages are linked
at the individual level. Verify the exact battery and waves for the sample before relying on
this.

1. **Build `θ` from multiple measures.** The CDS carries several Woodcock–Johnson subtests
   across waves. Using more than one as measures of a single latent factor corrects the
   attenuation in (6.1); this is the measurement-system approach of Cunha, Heckman and
   Schennach (2010). At minimum, standardize **within age at test**, since raw scores rise
   mechanically with age, and take the last available wave before 18.
2. **Anchor.** Set `θ̃ = (ln θ − m_θ)/s_θ` with `(m_θ, s_θ)` computed once and frozen, so
   `α_θ` reads as "log-wage effect of a 1 SD increase in childhood human capital" and is
   directly comparable to the published estimates in §8. This is Colas's `α₀ + α₁ ln θ̂`
   with the two coefficients pinned by the mean and variance.
3. **Age/experience profiles from the PSID main sample**, separately by education, with year
   fixed effects and a selection correction — Daruich & Fernández's step 1. The main panel
   is large and long; the CDS/TAS subsample is neither.
4. **Residual regression by education.** Form the age-purged residual for the CDS/TAS
   individuals observed working, average over available years, and regress on `θ̃`
   separately for `E = 0` and `E = 1`. This yields `α_θ` and `α_θ + α_θE`.
5. **Feed the two coefficients into the SMM as moments**, per (6.3d), rather than imposing
   them as parameters.

⚠️ **The binding constraint will be sample size and age coverage.** CDS children are
observed only into their late twenties/early thirties, so late-career wages — which is
where `γ₂`, `γ₁E` and `γ₂E` are identified — are not available in that subsample. This is
the identical problem Colas et al. faced and solved by combining datasets. The analogue
here: **profiles from the PSID main panel, the skill→wage loading from CDS/TAS.** Step 3
above already does this; it should be stated as a deliberate design choice in the thesis,
not left implicit.

#### What each paper actually uses

Every one of the four splits the work across sources, for the same reason.

| Paper | Wage / earnings data | Skill measure |
|---|---|---|
| **Adda et al.** | German IAB administrative social security records (semiannual) + GSOEP (annual) | none — a permanent unobserved type `f^P_i` absorbs pre-market ability |
| **Colas et al.** | NLSY79 for age profiles (older cohort, late-career wages observed); NLSY97 for everything else | AFQT, made comparable across cohorts and age-at-test via Altonji, Bharadwaj & Lange (2012) |
| **Lee & Seshadri** | **PSID** for life-cycle earnings and volatility (HS *n*=754, college *n*=712) | **CDS Letter–Word test scores**, difficulty-adjusted |
| **Daruich & Fernández** | **PSID** for age profiles, "since the age of the sample does not covary perfectly with the year of the survey" | AFQT in the **NLSY** — "the PSID in general does not have information … such as an AFQT score" |

**Lee & Seshadri is the direct precedent: they use PSID + CDS, exactly your data.** Three of
their choices are worth copying:

- **Which test.** The CDS carries several, but they use Letter–Word because it "is the only
  test administered to all children of all ages."
- **How they adjust it.** Let `d_q` be the fraction of children answering question `q`
  correctly, pooled over ages; question `q` is then worth `d_q` points, and the weighted sum
  is normalized to [0, 100]. This is a difficulty weighting, and it is cheaper than a full
  measurement system while still addressing the raw score's age-dependence.
- **What they call it.** "We take these normalized, adjusted test scores as `log h̃`" — the
  adjusted score is treated as **log** childhood human capital, not its level. That is a
  substantive choice, and it is the one consistent with a log-linear technology and a log
  wage equation.

They also record that the **variance** of adjusted scores rises monotonically with age from
age 2 — which matters for step 1 above, since standardizing within age is not optional.

One thing they do *not* do: correct for measurement error with multiple measures. Their
footnote 26 concedes the Letter–Word score is cognitive only and that "including multiple
skills is beyond the scope of this paper." Cunha, Heckman and Schennach (2010) is the
reference if you want the measurement-system route.

### 6.5 Making `α_θ` and `R_0` identified

`R_0` is in the estimated set (`smm.jl:173`) and `α_θ` is proposed as estimated too. There
are two distinct problems here; they have different severity and different fixes.

**Problem 1 — an exact level redundancy. This is the binding one.** The technology is

```
log HC' = log R + σ₁ log τ_p + σ₂ log e_p + σ₃ log HC
```

so `R → λR` shifts `log θ` by a constant `c = ln λ·(1 + σ₃ + … + σ₃^16)`. In the wage that
becomes `α_θ·c/s_θ`, a constant, absorbed one-for-one by `ln w₀`. **`R_0` and `w₀` are
exactly confounded whenever `α_θ ≠ 0`**: you can estimate at most two of
`{R_0, w₀, α_θ}`, and the third is redundant. This is Lee and Seshadri's footnote 9 —
"it is not separately identified from `q₀`" — in this model's notation.

*Today this is masked*, because `w₀ = 20.0` and `α = 0.08` are both fixed while only `R_0`
is estimated. It becomes live the moment `α_θ` enters the search.

**Problem 2 — a scale confound, and it is resolvable.** With `(m_θ, s_θ)` frozen, the wage
carries dispersion `α_θ·sd(log θ)/s_θ`, so wage dispersion alone identifies only the
*product* of `α_θ` and the true dispersion of childhood HC — which `σ₁, σ₂, σ₃` govern.

This is less serious than it looks: **the `σ`s are already identified from the input side**,
by the `e_p`, `τ_p` and `i_c` profiles among the existing twelve moments. Given the `σ`s,
`α_θ` is identified off wage dispersion. The flat direction only appears if you try to
identify the `σ`s from wage data as well.

#### The recommendation

1. **Normalize `R_0 = 1` and drop it from the estimated set.** Human capital has no natural
   scale, so the TFP of its production function is a units choice, not an economic
   parameter — fixing it is what makes the `σ`s and `α_θ` separately meaningful. This is the
   Agostinelli–Wiswall point about which normalizations a skill technology actually
   requires. Note `R_0` is weakly identified today in any case: no moment in the current set
   speaks to the *level* of child human capital.
2. **Estimate `α_θ` and `α_θE` against two new moments.** Candidates, in order of
   preference:
   - the **intergenerational elasticity of earnings** (child's log earnings on parent's) —
     a headline number, and Lee and Seshadri calibrate to an IGC of 0.34;
   - the **auxiliary regression coefficients** of §6.4 — child's log wage on standardized
     childhood skill, run separately by education, matched by indirect inference (§6.3d).
     The college/high-school difference identifies `α_θE`;
   - `Var(log wage)` at a given age, as a fallback.
3. **State the normalization in the paper**, one line: *human capital is measured in units
   such that `R_0 = 1`.*

**The alternative, which is what the literature does:** fix `α_θ` and `α_θE` at values
estimated *outside* the structural model (§6.4) and estimate `R_0` freely. Colas et al. and
Daruich & Fernández both take this route, which is why neither attempts a joint estimate.

Either works. What does not work is estimating `R_0`, `w₀` and `α_θ` together — the search
then has an exactly flat direction, and this should be settled before the next SMM run.

### 6.6 Further reading

Directly on anchoring and the measurement of childhood skill. **None of these are in
`docs/papers/` — verify before citing.**

- **Cunha, Heckman and Schennach (2010)**, *Econometrica* 78(3) — the canonical treatment of
  anchoring and of measurement error via a measurement system. Their central warning is that
  estimated technology parameters are **not invariant** to the anchoring choice, which is
  the general form of §6.5. Used by Daruich & Fernández for the skill distribution.
- **Cunha and Heckman (2008)**, *Journal of Human Resources* 43(4) — the earlier statement
  of the anchoring problem.
- **Agostinelli and Wiswall (2016)** — on which normalizations the skill-formation technology
  actually requires, and which are arbitrary. Colas et al. build their endogenous-ability
  extension on it.
- **Altonji, Bharadwaj and Lange (2012)**, *Journal of Labor Economics* 30(4) — making AFQT
  comparable across cohorts and age at test; used by Colas et al.
- **Heckman, Stixrud and Urzua (2006)**, *Journal of Labor Economics* 24(3) — the canonical
  factor-structure estimate of how cognitive and non-cognitive skill map into wages, with
  education-specific returns and correction for measurement error.
- **Del Boca, Flinn and Wiswall (2014)**, *Review of Economic Studies* 81(1) — cited by Lee
  and Seshadri alongside CHS for the anchoring.
- **Abbott, Gallipoli, Meghir and Violante (2019)**, *JPE* 127(6) — closest in structure to
  this model (parental transfers, college, general equilibrium); AFQT mapped into wages
  education-specifically. Cited by both Colas et al. and Daruich & Fernández.
- **Huggett, Ventura and Yaron (2011)**, *AER* 101(7) — Ben-Porath with initial human capital
  and learning ability fixed at labour-market entry; finds initial conditions account for
  most of lifetime earnings inequality. The source of Lee and Seshadri's `β`.
- **Ben-Porath (1967)**, *JPE* 75(4), "The Production of Human Capital and the Life Cycle of
  Earnings" — the original; ~10 pages and it makes Lee and Seshadri §II.A obvious.
- **Browning, Hansen and Heckman (1999)**, *Handbook of Macroeconomics* — the range of
  Ben-Porath exponent estimates Lee and Seshadri calibrate against.
- **Johnson (2013)** — the transfer-measurement and dataset-combination approach Colas et al.
  follow.

---

## D. Sources

- Adda, J., C. Dustmann and K. Stevens (2017), "The Career Costs of Children", *JPE*
  125(2). Eqs. (2)–(3), p. 304; estimates Table 3, p. 312. — `docs/papers/`
- Colas, M., S. Findeisen and D. Sachs (2021), "Optimal Need-Based Financial Aid", *JPE*
  129(2). Eq. (11) and Table 2, online appendix §3.3. — `docs/papers/`
- Lee, S. Y. and A. Seshadri (2019), "On the Intergenerational Transmission of Economic
  Status", *JPE* 127(2). Eqs. (1)–(2), p. 860; eqs. (3)–(5), pp. 861–862. — `docs/papers/`
- Daruich, D. and R. Fernández (2023), "Universal Basic Income: A Dynamic Assessment".
  Eqs. (2), (13), (14); wage process and return to skills, §3. — `docs/papers/`
- Cunha, F., J. Heckman and S. Schennach (2010), "Estimating the Technology of Cognitive
  and Noncognitive Skill Formation", *Econometrica* 78(3). — via Daruich & Fernández
- Kim, M. and S. Polachek (1994); Albrecht, J. et al. (1999) — atrophy rates, via Adda et al.
- Card, D. (1999), "The Causal Effect of Education on Earnings", *Handbook of Labor
  Economics*. — returns to schooling, 0.07–0.11
- Topel, R. (1991), "Specific Capital, Mobility, and Wages", *JPE* 99(1). — tenure returns,
  0.01–0.03
- Own parental wage regression, N = 9,455, adj. R² = 0.181 (`parent_family.jl:214`).
