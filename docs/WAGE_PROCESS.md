# The child's wage process: proposed respecification

**Status: proposal for discussion. Nothing in `code/` has been changed.**

Scope: `code/src/child_lifecycle.jl` only. The parent block's wage equation
(`parent_family.jl:1012`) is cited below as *evidence* and as a consistency benchmark; no
change to it is proposed here.

---

## 1. Final recommendation

The child's wage is currently `w_t = w₀·(1 + α·HC_t)·z_t` with `α = 0.08`,
`HC_{t+1} = HC_t + h_t`, and `HC += 2.0` for each of four college years
(`child_lifecycle.jl:243, 496, 537`). Three changes are proposed.

### 1.1 The three changes

**A. Wage equation — logs, with an age profile and an ability–college interaction.**

```
ln w_t = ln w₀ + β_E·E + (α_θ + α_θE·E)·θ̃ + (γ₁ + γ₁E·E)·t + (γ₂ + γ₂E·E)·t² + z_t
```

`t` is age (the model period), `θ̃` is standardized childhood human capital at 18 carried as
a fixed state, `E ∈ {0,1}` is college, and `z` is the existing AR(1), centred so `E[z] = 1`.
This separates three objects the current form welds together, and it reverses the sign of
the ability–college interaction (§3.2).

**B. No adult human-capital accumulation.** The `HC` state leaves the child's problem
entirely; `college_boost` leaves the law of motion and becomes `β_E` in the wage. This
follows Colas et al. and Daruich & Fernández, neither of which accumulates adult human
capital. It removes the `h_hi = clamp(k_max − capital, …)` hours cap, the depreciation
parameter, and the `∂k'/∂h` term in the Bellman gradient — and it is **5× cheaper** than the
current code (§7.0, §10).

**C. Psychic cost — logs, and let parental education in.**

```
κ_X = κ₀ + κ_θ·ln θ   ( + κ_ParEd·BothCollege )
```

replacing `κ/(HC+1)^4` (`child_lifecycle.jl:633`). Both Colas
(`κ₀ + κ_θ log θ + κ_fem + κ_ParEd·ParEdu`) and Daruich & Fernández
(`κ₀ + κ_θC ln θ^C + κ_θNC ln θ^NC + ξ`) use a log form, both make it decrease in ability,
and both let **parental education** shift it. `BothCollege` is already a state in the parent
block, so `κ_ParEd` is free and gives a direct channel for the intergenerational persistence
of education. This also retires open error C2 (the `^4` vs `^2` discrepancy) by replacing
the power form rather than patching it.

The spend saved in **B** should go to resolution: `Na = 80`, `Np = 7`, `Nθ = 11` costs about
what the model costs today. `Np = 7` matters — `GUIDE.md` records that raising the *parent's*
`Np` from 3 to 7 moved the college share **17.85% → 22.40%**, and the child still runs at
`Np = 5` with that convergence study never done.

### 1.2 What this does to the two things you need

**Progressive-taxation counterfactuals — improved, with one channel lost.**

The instinct is that dropping accumulation weakens a tax counterfactual. For this model it is
the other way round, because the first-order problem is the *shape* of the earnings path, not
its endogeneity. With `w_vec = fill(w, T)` the child's earnings are **flat in age**, so a
progressive tax bites identically at 25 and at 55. Real profiles hump — the parent block's
own estimates give 0.306 log points of growth to peak for non-college and 0.473 for college —
so progressivity in fact bites hardest in the forties. Getting `γ₁, γ₂` right changes both
the revenue raised and the distortion imposed; that is first-order for your counterfactual,
and it is fixed by **A**.

Of the four channels through which higher progressivity operates, three survive intact:

| Channel | Status |
|---|---|
| tax → parental after-tax income → `e_p`, `τ_p` → child human capital | **intact** — the thesis's core mechanism, and it lives in the parent block |
| tax → after-tax return to college → enrolment → education composition | **intact** — the Bovenberg–Jacobs / Colas fiscal externality |
| tax → child's static labour supply, interacted with the profile shape | **intact and improved** by A |
| tax → child's hours → experience → future wages | **lost** |

Colas analyse optimal financial aid under a progressive tax, and Daruich & Fernández analyse
a UBI — both large redistribution questions — with no adult accumulation.

**The belief machinery — the bias is preserved, the reconciliation term is not.**

Beliefs move from `college_boost` to `β_E`, the log wage premium. To be clear about what
survives: **every decision is still taken under the biased number.** What disappears is only
the algebraic correction that reconciles a perceived *stock* with the true one.

*Timing, explicitly:*

| Stage | Wage parameter in force | What it governs |
|---|---|---|
| Half-period, age 18 | `β_E^m` (**believed**) | enrolment choice `d` and the transfer `tr` |
| College, `t = 1…T_E` | `β_E^m` (**believed**) | consumption/saving while enrolled, and the continuation value at graduation |
| Labour market, `t > T_E` | `β_E*` (**true**) | the realized wage and every policy from then on |

So an optimistic family transfers too much and enrols too readily, the child consumes
against a wage that will not arrive, and the error is revealed as a one-time surprise at
labour-market entry — which is also Colas's timing for `v^e*`, revealed "as soon as the
agents finish their education and enter the labour market." What is *not* needed is
`HC' = H̃C + b* + (T_E−1)(b* − b_m)`: that term exists only to reconcile four years of
accumulated drift in a perceived stock, and with no stock there is no drift.

*Implementation — and it costs nothing.* Under a log wage `β_E` is a pure level shift, so a
believer of type `m` is arithmetically identical to a truthful agent at a shifted `θ̃`:

```
Δ_m = (β_E^m − β_E*) / (α_θ + α_θE)
```

- **Perceived continuation** (used in the college Bellman and the enrolment/transfer
  decision): `V^W(a', z'; θ̃ + Δ_m, E = 1)`
- **Actual continuation** (used from `t = T_E + 1` in simulation): `V^W(a', z'; θ̃, E = 1)`

One set of work arrays serves every belief bin — you index at `θ̃ + Δ_m` or at `θ̃`. The
`θ̃` grid must be extended to cover the belief range; that is the entire cost.

The college-stage solve needs no per-bin copy either. With no accumulation, `θ` is fixed
through college, so the psychic cost is a **constant per period** and is additively separable
from consumption (`util_college`, `:633`, has no `c`–`κ` interaction). Therefore

```
V^E(a; θ, m) = Ṽ^E(a; θ̃ + Δ_m) − κ_X(θ)·Σ_{s=0}^{T_E−1} β^s
```

exactly: solve `Ṽ^E` once without the psychic cost, then subtract a closed-form annuity.
This factorization **only works without accumulation** — today `k` changes during college,
so `κ/(k+1)^4` varies year to year and cannot be factored out.

Against this, `parent_family.jl:1598-1611` currently builds a *separate child model per
belief bin*, each with its own `college_boost`, college arrays and transfer arrays.

This is work to do in `parent_family.jl` alongside the child-side change, not a caveat: the
belief machinery has to be re-pointed for the rest of the specification to be coherent, and
doing so makes it both cheaper and closer to its cited source.

### 1.2b Fixing `w_vec = fill(w, T)`

`w_vec` is built once in the constructor (`:203`) and `model.w` is never read again, so
mutating `model.w` after construction silently does nothing — while `model.alpha` *is* read
live in `wage_func`. The two behave differently, and the difference is invisible until an
estimation run quietly ignores a parameter.

**The fix is to delete `w_vec`.** Under the recommendation the age profile is
`γ₁t + γ₂t²`, computed inline from live struct fields:

```julia
@inline function wage_func(m::ConSavLaborCollege_AR1, θ̃::Float64, t::Int,
                           E::Float64, z::Float64)
    lw = m.lnw0 + m.β_E*E + (m.α_θ + m.α_θE*E)*θ̃ +
                            (m.γ₁ + m.γ₁E*E)*t + (m.γ₂ + m.γ₂E*E)*t^2
    return exp(lw) * z
end
```

Every parameter is read at call time, so nothing can go stale. The cost is one `exp` per
call, which is negligible against the surrounding SLSQP solve. If a cache is wanted later,
it must be a `(T, 2)` table rebuilt by an explicit `refresh_wage_cache!(model)` after any
parameter change — never a vector frozen at construction.

This also disposes of `WAGE_SCALING_FACTOR = 0.584`: in logs it is an additive constant, so
it folds into `ln w₀` and the duplicated `const` in both modules can go.

### 1.3 Dropped from earlier drafts

Experience as a state, depreciation `δ`, the full-time-equivalent units question, and
Ben-Porath — all consequences of **B**. §7.3 shows Ben-Porath would not have delivered the
ability–college complementarity anyway. §7.1–7.5 are retained only as the argument for why
accumulation, *if* kept, cannot simply extend the childhood technology.

### 1.4 The one limitation to state in the paper

One sentence, in the model section: *the model abstracts from on-the-job human capital
accumulation, so the dynamic cost of progressivity through forgone experience is not
captured; the tax's human-capital margin operates through parental investment and through
college enrolment.* If Sahber judges that channel essential to the tax result, the fallback
is §4 with experience in place of age — 5× the compute, and better run as a robustness check
than as the baseline.

The four reasons the current form has to change are quantified in §3; one of them reverses
the sign of the model's central mechanism. §5 sets out how Adda, Dustmann and Stevens (2017),
Colas, Findeisen and Sachs (2021), Lee and Seshadri (2019) and Daruich and Fernández (2023)
each handle the adult block.

---

## 2. What is implemented now

| Object | Code | Value |
|---|---|---|
| Wage | `w₀(1 + αHC)z` | `wage_func`, `:243` |
| Return to skill | `α`, a single scalar | 0.08 |
| Human capital | `HC' = HC + h` | `:496` |
| College | `HC += 2.0` × 4 years | `:537`, `college_boost = 2.0` |
| Base wage profile | `w_vec = fill(w, T)` | **flat**, `:203` |
| Shock | AR(1), ρ = 0.95, σ = 0.2, Rouwenhorst, N = 5 | `:167` |
| Initial condition | `θ = HC` at 18, from the parent block | `sim_k_init`, support [0, 5] |

`HC` is one continuous state carrying childhood skill, college, and experience
simultaneously, and they enter as perfect substitutes.

---

## 3. Four problems, quantified

### 3.1 One parameter is asked to be three different elasticities

Because `θ`, college and experience are the same state, `α` prices all three. Evaluated at
`HC = 3` (early career):

| What α is being asked to be | Model implies | Literature |
|---|---|---|
| Return to one year of college | **0.121** | Card (1999): 0.07–0.11 |
| Return to one year of full-time work | **0.063** | Topel (1991), tenure: 0.01–0.03 |
| Effect of 1 SD of childhood HC | **0.093** | ≈ 0.15–0.20 for 1 SD AFQT |

`α = 0.08` was calibrated from the *schooling* literature. It cannot simultaneously be a
return to schooling, a return to experience, and an anchor for childhood skill.

Note also that `college_boost = 2.0` is not an estimate — it hard-codes "one year of
college = two years of full-time work."

### 3.2 The college premium falls in childhood human capital — the wrong sign

College adds a **fixed** +8 to a stock entering the wage **in levels**. Since `1 + αHC` is
concave in logs, a fixed additive increment has a declining log return:

| θ (childhood HC) | HS wage index | College wage index | log premium |
|---|---|---|---|
| 0.0 | 1.000 | 1.640 | **0.495** |
| 2.5 | 1.200 | 1.840 | 0.427 |
| 5.0 | 1.400 | 2.040 | **0.376** |

across the actual support of `sim_k_init`. Two independent papers estimate the opposite:

| | Return to ability, HS | Return to ability, college | Ratio |
|---|---|---|---|
| Colas et al. (2021), Table 2 | `β_θ^H` = 0.31 | `β_θ^G` = 0.47 | **1.52** |
| Daruich & Fernández (2023), §3 | `α^H` | `α^G` | **≈1.5** |

Colas read this as "a complementary relationship between initial ability and education";
Daruich and Fernández report simply that "the returns to skill are 1.5 times greater for
college-educated workers than high-school ones." Different datasets (NLSY79 + NLSY97 versus
NLSY + PSID), same answer.

**Why this matters here specifically.** The parent block's entire purpose is to raise `θ`
through `τ_p` and `e_p`. Under the current form, raising `θ` *lowers* the return to
college — parental investment and college are **substitutes** in this model. Colas et al.,
Cunha–Heckman and Keane–Wolpin all find them complements. This sign was not a modelling
choice; it is an artefact of `1 + αHC`.

### 3.3 There is no life-cycle wage profile

`w_vec = fill(w, T)`, so the `t` argument to `wage_func` is inert and every bit of wage
growth comes from `HC`. The child's wage never humps and never declines.

For contrast, the parent block's *estimated* profile
(`parent_family.jl:214`, from the Stata wage regression, N = 9,455):

| | slope | curvature | peak | total log growth |
|---|---|---|---|---|
| Not both-college | 0.0230 | −0.000432 | age 52 | 0.306 |
| Both college | 0.0404 | −0.000861 | age 48 | 0.473 |

and the implied college gap:

| model t | age | gap |
|---|---|---|
| 0 | 25 | 0.308 |
| 10 | 35 | 0.439 |
| 20 | 45 | **0.483** (peak) |
| 30 | 55 | 0.442 |

The gap **fans out by 57%** from entry to peak. A level-only college premium would impose
0.308 for all 51 periods, which the same regression rejects.

Two consequences beyond realism:

- College is an NPV comparison against `a_min = 0` and β ≈ 0.97. Front-loading the whole
  return makes college look more affordable to a constrained child than it is — and the
  constrained margin is exactly where the college share is determined (`ERRORS.md`, P11:
  "the binding constraint is the parental asset distribution against the college
  threshold").
- The tax `λ(wh)^(1−τ)` is convex, so *where in the life cycle* the extra earnings land
  determines the fiscal return to college — Colas et al.'s fiscal externality, and the
  object any redistribution result runs through.

### 3.4 No depreciation, and an hours cap that is a grid artefact

`HC' = HC + h` has no atrophy, so a low-hours career carries no penalty beyond forgone
accumulation. Adda, Dustmann and Stevens (2017) make depreciation their central mechanism:
`x' = x·ρ(x,o)` with ρ < 1 out of work.

There is also a numerical cost. With no depreciation nothing bounds `HC`, so hours are
capped by the grid:

```julia
h_hi = clamp(model.k_max - capital, 1e-3, 1.0)   # child_lifecycle.jl:352
```

A high-`HC` worker is forced to work fewer hours **because the grid ends**, not because of
anything economic. Depreciation makes `X → h̄/δ` endogenously and removes the hack.

### 3.5 Secondary: `E[z] ≠ 1`

`p_grid = exp.(mc.state_values)` with `E[log z] = 0`, so by Jensen `E[z] = 1.2235`. `w₀` is
therefore 22.3% below the mean entry wage rather than equal to it. Harmless while `w₀` is
fixed; a nuisance the moment it is estimated. sd(log z) = 0.64, against 0.23 in the parent
block — a 2.8× difference between two generations of the same economy.

---

## 4. Proposed specification

```
ln w_t = ln w₀
       + β_E·E                          college premium at entry
       + (α_θ + α_θE·E)·θ̃              childhood HC, education-specific return
       + (γ₁ + γ₁E·E)·X_t               experience, education-specific slope
       + (γ₂ + γ₂E·E)·X_t²              curvature
       + z_t                            AR(1), unchanged
```

| Symbol | Meaning | Status in the solution |
|---|---|---|
| `θ` | childhood HC at 18, `HC_{T_L}` from the parent block | **fixed for life**; new state, Nθ = 5 |
| `θ̃` | `(ln θ − m_θ)/s_θ`, standardizers computed once and **frozen** | makes `α_θ` a unit-free "1 SD of childhood HC" effect |
| `E` | college indicator | new state, {0,1} |
| `X_t` | experience in full-time equivalents, `X_{t+1} = (1−δ)X_t + h_t/h_FT`, `X₁ = 0` | replaces `HC`, NX = 25; see the units note in §7 |
| `z_t` | AR(1), ρ = 0.95, σ = 0.2, Rouwenhorst | unchanged, but centred so `E[z] = 1` |

**Removed:** `α` and `(1 + α·HC)`; `college_boost` from the law of motion — college now
buys `β_E` in the wage, not experience; the `h_hi` grid hack.

One consequence worth stating: a graduate now enters work at 22 with `X = 0` while a
high-school worker has `X ≈ 1.6`. That is the correct education-versus-experience trade-off
and it is currently absent, since college *adds* experience.

**Nested restriction.** Setting `α_θE = γ₁E = γ₂E = 0` and `δ = 0` collapses this to a
plain Mincer log form. Both should be reported, so the added structure is testable rather
than asserted.

### Why the standardization

`θ` is produced by `log HC' = log R + σ₁ log τ_p + σ₂ log e_p + σ₃ log HC`, whose
parameters are themselves SMM-estimated — so the units and dispersion of `θ` move whenever
the HC technology is re-estimated. Standardizing once and freezing `(m_θ, s_θ)` keeps
`α_θ` interpretable and stable across estimation runs.

This is the same problem Lee and Seshadri (2019) solve with their anchor `z`, which
"transforms children's human capital, which we will later proxy by test scores in the data,
into adult outcomes, which we will measure using earnings."

⚠️ **This section is the *fallback*, not the recommendation.** `X_t` here is experience,
which presumes a human-capital state; §1.1B and §7.0 drop that state, in which case `X_t`
becomes age `t` and everything else in this section stands unchanged. Keep this version only
if the dynamic cost of progressivity through forgone experience turns out to be essential to
the tax result (§1.4) — and then as a robustness check, since it costs 5× (§10). §7.4 is a
third position: keep accumulation but move the profile into the technology, so the wage can
drop `γ₂` and possibly `γ₁E, γ₂E`.

---

## 5. How the four papers model wages and human capital *after* the child grows up

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

## 6. Connecting measured childhood human capital to adult human capital

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

## 7. The post-18 human capital technology

### 7.0 First question: is one needed at all?

**Probably not — and dropping it is the recommendation of this section.** Two of the four
papers have no adult human-capital accumulation whatsoever. College shows up entirely in the
**wage equation**, and the cost of college entirely in the **enrolment decision**. That is
sufficient, it is what the closest papers to this project do, and here it is five times
*cheaper* than the status quo rather than five times more expensive.

**What Colas, Findeisen and Sachs actually do.** Their work-stage state vector is
`(X, I, e, a_t, w_t)` — characteristics, parental income, education, assets, current wage.
**There is no human-capital state.** Wages are

```
log y^e_it = β₀^es + β_θ^e·log θ_i + β^e_t1·t + β^e_t2·t² + β^e_t3·t³ + v^e*_i
```

— an education-specific constant, an education-specific ability loading, and an
education-specific cubic in **age**. Nothing accumulates. The only post-education
uncertainty is the single draw `v^e*` at labour-market entry; after that "there is no
further uncertainty about an agent's wage path."

The entire cost of college sits in the enrolment problem:

```
U^E(c_t, ℓ^E_t; X, ε) = c_t^{1−γ}/(1−γ) − κ_X − ζ_{ℓE} + ε_t
κ_X = κ₀ + κ_θ·log θ + κ_fem·I(female) + κ_ParEd·ParEdu
```

`κ_X` is a **flow** psychic cost "incurred every year the agent is enrolled", **decreasing
in ability** (`κ_θ`) and in parental education. Alongside it: tuition, forgone earnings, a
logistic taste shock `ε^E` at enrolment, dropout risk, and a cost of working while enrolled.

Daruich and Fernández do the same thing: efficiency units are an age profile times a
persistent shock, with no accumulation, and school taste is
`κ(ξ,θ) = κ₀ + κ_θC·ln θ^C + κ_θNC·ln θ^NC + ξ` with the mean of `ξ` depending on parental
education.

**The mapping to this model is nearly one-for-one.** `κ/(HC+1)^4` is already a flow psychic
cost decreasing in ability; `ε₀` is already a taste shock at enrolment; `college_cost` is
already tuition; forgone earnings are already there because `sol_h_college = 0`. What
changes is that `college_boost` moves out of the human-capital law of motion and becomes
`β_E` in the wage.

#### What this buys

| | Today | Colas-style |
|---|---|---|
| Work-solve SLSQP calls | 637,500 | **127,500 — 5× fewer** |
| Memory per array | 51 MB | **1.0 MB — 50× smaller** |
| Child state | `(a, HC, z)` | `(a, z)` + fixed types `(θ, E)` |

And the saving is redeployable. Holding total cost at roughly today's level:

| `Na` | `Np` | `Nθ` | cost vs today |
|---|---|---|---|
| 50 | 5 | 5 | 0.20× |
| **80** | **7** | **11** | **0.99×** |
| 100 | 7 | 13 | 1.46× |

That matters more than it looks. `GUIDE.md` records that raising the *parent's* `Np` from 3
to 7 moved the college share **17.85% → 22.40%**, while doubling any state grid moved it by
≤0.15pp — and **the child still runs at `Np = 5`, with that convergence study never done for
it.** Dropping the HC state pays for `Np = 7` on the child, a finer asset grid, and a
well-resolved `θ` dimension, all at today's cost.

Three nuisances also disappear: the `h_hi = clamp(k_max − capital, …)` hours cap (§3.4),
the depreciation parameter `δ`, and the `∂k'/∂h` term in the Bellman gradient.

#### What it costs

- **No learning by doing.** Hours no longer build future wages. The substantive loss is the
  *dynamic* cost of progressive taxation: higher marginal rates reduce hours, and with
  learning-by-doing that would compound into lower future wages. For a thesis on
  redistribution this channel is worth naming explicitly as absent. That said, Colas
  analyse optimal financial aid and Daruich & Fernández analyse a UBI — both large
  redistribution questions — without it. (The dynamic-human-capital-taxation literature is
  the place to check how much this matters; not read here, verify before citing.)
- **The profile runs on age, not experience.** This supersedes the earlier decision to use
  experience: you cannot have an experience profile without carrying experience as a state,
  and carrying it *is* accumulation under another name.
- ⚠️ **The heterogeneous-beliefs machinery must be re-pointed.** `parent_family.jl:1611`
  sets `belief_values` from `college_boost`. With `college_boost` gone, beliefs must be
  about `β_E` — the wage return to college — instead. This is arguably an improvement,
  since Bleemer (2018), the cited source, is about beliefs regarding the *earnings* return.
  But it is a change in `parent_family.jl`, outside the scope stated at the top of this
  note, and needs Sahber's agreement.

#### The specification under this option

```
ln w_t = ln w₀ + β_E·E + (α_θ + α_θE·E)·θ̃ + (γ₁ + γ₁E·E)·t + (γ₂ + γ₂E·E)·t² + z_t
```

identical to §4 except that `X_t` (experience) is replaced by `t` (age), which is not a
state. `θ` and `E` remain fixed types. Everything in §3 is still fixed: logs not levels,
the complementarity sign, a real profile, and `E[z] = 1`.

**§7.1–7.5 below apply only if the accumulation is kept.** They are retained because the
argument in §7.2 — that the childhood technology cannot be carried forward — is what makes
the case for dropping it, and because §7.5's point about connecting 0–18 to 18+ holds either
way.

---

### Sections 7.1–7.5: only if accumulation is kept

The technology would have to change at 18, and differ between the work branch and the
college branch. This sets out what form it should take, and why the obvious elegant answers
do not work.

### 7.1 The change of functional form at 18 is not a defect

**All four papers change functional form at the school-to-work transition.** Lee and
Seshadri run a three-stage CES technology in parental time and goods through childhood
(their eqs. 4–5) and then switch to Ben-Porath from `j = 3` onward (eq. 1). Colas and
Daruich & Fernández stop accumulating altogether. Adda have no childhood block at all.

So the discontinuity between `log HC' = log R + σ₁ log τ_p + σ₂ log e_p + σ₃ log HC` before
18 and something else after is standard, and should be stated as a deliberate choice with a
reason rather than apologised for. The reason is economic: **before 18 human capital is
produced by investment in a growing child; after 18 it is maintained and incremented by the
agent's own time.** Different inputs, different technology.

### 7.2 The childhood technology cannot simply be carried forward

At the current calibration, `σ₃ = exp(−0.36) = 0.698`. If that self-productivity applied
after 18, a childhood-HC difference would decay as `σ₃^t`:

| periods after entry | share of the initial difference remaining |
|---|---|
| 5 | 16.5% |
| 10 | 2.7% |
| 20 | 0.07% |
| 30 | ≈ 0 |

**Childhood human capital would be numerically irrelevant by the mid-thirties**, and with it
the entire intergenerational mechanism the thesis is about. This is why every paper in this
literature sets adult self-productivity to (essentially) one: Adda's `x' = x + 1`, Lee and
Seshadri's `+ h_j` term, and — the limiting case — Colas and Daruich & Fernández not
accumulating at all, so the ability loading is permanent by construction.

**Requirement 1: adult self-productivity must be 1 (or `1 − δ` with δ small).**

### 7.3 Where ability–college complementarity can come from — three routes, two dead ends

There are exactly three places the complementarity of §3.2 can be generated. Two of them do
not survive contact with the magnitudes.

**Route A — the wage equation.** `α_θE > 0`, as in Colas and Daruich & Fernández. Direct,
imposed, and calibratable straight from published numbers (1.52 and ≈1.5). ✅

**Route B — a separate learning-ability parameter multiplying investment.** Lee and
Seshadri's `h' = ε[a·(n·h)^β + h]`, where `a` is inherited ability, an object *distinct from
the human-capital stock*. The complementarity is between `a` and investment `n`, not between
the stock and investment. ✅ but costs an extra state or fixed type.

⚠️ **It is worth being explicit that Ben-Porath on its own does not deliver
complementarity.** With `β < 1` the investment term `A·(n·HC)^β` is concave, so a college
period adds proportionally *more* to a low-HC agent:

| HC | log gain from one college period, `HC' = 0.99·HC + HC^0.6` |
|---|---|
| 1.0 | 0.688 |
| 3.0 | 0.491 |
| 5.0 | 0.416 |

That is the same declining pattern as the current `1 + α·HC` — reproduced by a different
functional form. Adopting Ben-Porath without `a` would not fix §3.2.

**Route C — education-specific self-productivity**, `σ_E > σ_W` in a multiplicative
technology. Under `HC' = A·HC^σ` the loading of `log θ` after four periods is `σ^4`, so
matching Colas's ratio requires `σ_E/σ_W = 1.52^{1/4} = 1.110`. Combined with Requirement 1
(`σ_W ≈ 1`), that forces `σ_E ≈ 1.11 > 1` — **explosive**. ❌

**Conclusion: use Route A.** It is what §4 already proposes, and it is what both papers that
measure childhood skill actually do. Route B is available as an extension if inherited
ability is added (open question §11.4), and the two are complements, not substitutes.

### 7.4 If accumulation is kept, this is the form it should take

Given Requirements 1 and Route A:

```
Work branch,    every period after schooling:
    HC_{t+1} = (1 − δ)·HC_t + A_W(E)·(h_t / h_FT)^φ

College branch, t = 1 … T_E  (full-time study, no hours choice):
    HC_{t+1} = (1 − δ)·HC_t + A_E
```

| Parameter | Role | Analogue |
|---|---|---|
| `δ` | depreciation | Adda's `ρ(x,o)`; §3.4 |
| `A_W(E)` | productivity of on-the-job learning, **education-specific** | Adda's `a_X(o)`; the parent block's `BC×Age` |
| `φ` | concavity in hours, `φ < 1` | generates a concave profile without needing `γ₂` |
| `A_E` | productivity of a college year | replaces `college_boost` |

Three things this buys:

- **Self-productivity is 1 − δ**, so childhood HC persists (Requirement 1).
- **The two branches differ in the right way**: college is a fixed full-time increment with
  no hours margin — matching the current code, where `sol_h_college = 0` — while work is a
  concave function of chosen hours. `A_E` versus `A_W` is now an estimable ratio rather
  than the hard-coded "one college year = two work years."
- **`A_W(E)` gives the education-specific profile slope** without a separate `γ₁E, γ₂E` in
  the wage; the fan-out comes from the accumulation.

This is a **simplification of §4, not an addition to it**: if the profile shape is generated
by `φ` and `A_W(E)` inside the technology, the wage can drop `γ₂` and possibly `γ₁E, γ₂E`,
becoming

```
ln w_t = ln w₀ + β_E·E + (α_θ + α_θE·E)·θ̃_1 + α_X·ln HC_t + z_t
```

where `θ̃_1` is the *initial* (age-18) standardized skill, carried as a fixed state, and
`HC_t` is the accumulating stock. Which of the two versions to take is a judgement call —
§4 puts the profile in the wage, §7.4 puts it in the technology — and it is worth asking
Sahber directly which he prefers, since the identification differs.

### 7.5 What this means for connecting 0–18 to 18+

Under §7.4 the connection is the simplest possible one, and that is the point:

**`HC` at 18 from the parent block IS the initial condition of the adult stock, in the same
units. There is no mapping.** This is Lee and Seshadri's structure exactly — childhood
human capital `h'_3` produced by the CES technology *is* the `h_3` that enters the adult
Ben-Porath recursion.

Everything then reduces to a single question: **what is one unit of model `HC` worth in
wage terms?** That is the anchor — `z` in Lee and Seshadri, `α₀, α₁` in Colas, `α_θ` here.
And §6.5's warning applies with full force: the anchor is not separately identified from the
productivity of the childhood technology. Lee and Seshadri say so in their footnote 9; the
implication here is that `α_θ` and `R_0` cannot both be estimated freely.

The empirical procedure of §6.4 is unchanged, but its interpretation sharpens: the
regression of adult log wages on standardized childhood skill no longer identifies a wage
loading directly, because the childhood measure passes through the accumulation before it
reaches the wage. It identifies a *reduced-form* pass-through. That is fine — it reinforces
§6.3(d): run the same regression on simulated data and match the coefficients, rather than
plugging the estimate in as a structural parameter.

---

## 8. Provisional calibration

To be treated as placeholders. Ali's intention is to estimate these by SMM once earnings
moments exist; §9 sets out what that would require.

| Parameter | Meaning | Provisional | Source |
|---|---|---|---|
| `α_θ` | 1 SD childhood HC → log wage, HS | 0.31 | Colas Table 2 |
| `α_θE` | additional for college | +0.16 | Colas (0.47 − 0.31); Daruich & Fernández independently ≈1.5× |
| `β_E` | college premium at `X = 0` | ≈ 0.31 | Colas; cf. own parent regression, 0.308 |
| `γ₁` | return to one FTE year of experience | 0.08–0.10 | Adda `a_X` = 0.090–0.123 |
| `γ₂` | curvature | −0.002 to −0.004 | Adda `a_XX` = −0.00210 to −0.00463 |
| `γ₁E`, `γ₂E` | education-specific slope, curvature | flatter curvature for college | Adda: abstract vs routine, +2pp/yr at 10 yrs; own parent regression `BC×Age` = +0.0174 |
| `δ` | experience depreciation | 0.01–0.02 | Adda ρ(x,o); Kim & Polachek 0.02–0.05, Albrecht et al. ≈0.02 |

At `γ₁ = 0.08`, `γ₂ = −0.003` the profile peaks at 13.3 FTE years with total growth of 0.53
log points — in the right region for a US life-cycle wage profile, and close to Adda's
routine-occupation estimates.

⚠️ **A units trap that must be resolved before these numbers are used.** Mincer
coefficients are per *year of full-time work*. In this model `h` is a **share of the time
endowment**, and full-time is not `h = 1` — the parent block's target is `h_p ≈ 0.35`, and
`WAGE_SCALING_FACTOR = 0.584` is described as an "adjustment for hours worked per year."
If `X` accumulates raw `h`, then a full-time career reaches `X ≈ 0.4 × years`, and applying
`γ₁ = 0.08` silently delivers 0.032 per calendar year rather than 0.08.

The clean fix is to accumulate in **full-time equivalents**:

```
X_{t+1} = (1 − δ)·X_t + h_t / h_FT
```

with `h_FT` measured from the calibrated model and recorded. Then `X` is FTE years and the
published coefficients apply directly. This needs doing either way — the alternative is to
rescale every `γ` by `h_FT`, which is the same thing done less transparently.

**Three further caveats, stated rather than buried:**

1. Colas's `β_θ` is estimated on **log AFQT**. Our `θ` comes from a different production
   function in different units. The *ratio* 1.52 is unit-free and transports; the *level*
   does not. Hence the standardization — and `α_θ` still needs its own anchor, ideally a
   target of the form "1 SD of childhood HC raises adult wages by x%".
2. `β_bothcollege = 0.308` is **both parents college-educated**, not an individual college
   premium. It is used here for the fan-out *shape*, not the level.
3. The parent's coefficients are estimated on **age**; the child's `γ`s run on
   **experience**. Since `X` accumulates at `h` with full-time `h = 1`, `X` is
   full-time-equivalent years and standard Mincer *experience* coefficients apply directly.
   The parent regression is cited only for the education-interaction ratio.

---

## 9. Identification, for when these enter the SMM

At present **no wage or earnings moment exists.** All 12 SMM targets are parental time use,
consumption, education spending, terminal assets and the college share, and the code records
them as "midpoints of the ranges asked for" (`smm.jl:126`). Every wage parameter — the six
parental βs, `α`, `w₀`, `ρ`, `σ_p`, `college_boost` — is fixed.

What each new parameter would need:

| Parameter | Identifying moment |
|---|---|
| `γ₁`, `γ₂` | mean log wage growth 25→45; age at peak wage |
| `β_E`, `γ₁E`, `γ₂E` | college/HS log wage gap **at entry** and **at peak** — two moments, not one |
| `α_θ`, `α_θE` | intergenerational elasticity of earnings; college share by `θ` quintile |
| `δ` | weakly identified — there is no non-employment margin in the child's problem, so **calibrate rather than estimate** |

One design consequence of the intent to estimate: `θ` must stay a **fixed state grid**. It
is tempting to fold `ln w₀ + β_E·E + (α_θ + α_θE·E)·θ̃` into a single scalar wage index and
save a dimension, but that index would move with every parameter draw and break the
interpolation the search depends on.

---

## 10. Computational cost

The two options in §7 differ by a factor of 25, so this depends entirely on which is taken.

| | Work arrays | SLSQP calls | vs today |
|---|---|---|---|
| **Today** | `(T,Na,Nk,Np,Nt)` = (51,50,50,5,10) | 637,500 | — |
| **§7.0, no accumulation** | `(T,Na,Np,Nθ,2)` = (51,50,5,5,2) | **127,500** | **0.20×** |
| **§4/§7.4, experience** | `(T,Na,NX,Np,Nθ,2)` = (51,50,25,5,5,2) | 3,187,500 | 5.0× |

Today's `Nt` dimension is **degenerate on the work arrays** — every slice written with `.=`
and read at index 1 — so 9/10 of those 51 MB is wasted. The college arrays also shrink to
`t = 1..t_college`, since for `t > t_college` they are a copy of the work solution.

**Under §7.0** the memory per array falls from 51 MB to 1.0 MB (50×), and the saving is
better spent on resolution than banked — see the grid table in §7.0. No parallelism is
needed at all.

**Under §4/§7.4** CPU rises ~5×, with two mitigations: the work solve is Tier-0 cached on
`(college_cost, r)` only, so SMM pays it roughly 18–32 times per run rather than once per
draw; and the 10 (θ, E) cells are fully independent solves sharing no mutable state.

⚠️ **In that case they cannot be parallelised with threads.** Commit `7ed4f2c` established
that NLopt.jl dies silently under concurrent `optimize` — the process exits 0 with no output
at 8 threads while the identical single-threaded run completes. The (θ, E) loop is a loop of
`solve_model_work!` NLopt calls, so it hits precisely that blocker. That commit's own
recommendation applies: **use processes, not threads** — `Distributed` + `pmap` over the 10
cells.

---

## 11. Open questions for Sahber

The first two need answers before implementation; the rest can be recorded and deferred.

0. **The headline one: should adult human capital accumulate at all?**
   **Recommendation: no** (§1.1B, §7.0) — college enters through the wage, its cost through
   the enrolment decision, exactly as in Colas et al. and Daruich & Fernández. Simpler, what
   the two closest papers do, 5× cheaper. The cost is the dynamic effect of progressivity
   through forgone experience (§1.2, §1.4). **The question for you: is that channel
   essential to the tax counterfactual, or is it acceptable as a stated limitation?**

1. **Psychic cost.** **Recommendation given** in §1.1C: `κ₀ + κ_θ·ln θ`, replacing
   `κ/(HC+1)^4` and retiring open error C2. **The open part: should `κ_ParEd·BothCollege`
   be included?** It is free — `BothCollege` is already a parent state — and both Colas and
   Daruich & Fernández find parental education shifts the psychic cost. It would give a
   direct channel for the intergenerational persistence of education, which may or may not
   be something you want the model to generate endogenously instead.

2. **College risk.** Colas's `v^e*` is unknown at enrolment and revealed on entry
   (Var 0.42 college vs 0.36 HS); Daruich and Fernández independently find college agents
   "draw their initial productivity from a distribution with a somewhat higher variance."
   This makes the monetary return to college risky — one of Colas's main explanations for
   the parental-income enrolment gradient. This model has a taste shock `ε₀` and
   heterogeneous *beliefs* about the college return (`b_m`), but no monetary risk. Should
   it be added, and does it duplicate the belief machinery? A cheap version exists: draw
   the entering `z` from an education-specific variance, which costs no new state.

3. **Inherited ability.** Lee and Seshadri give ability an AR(1) across generations,
   separate from investment; Daruich and Fernández do the same for both skill components.
   This model has no such channel, so it assigns the entire intergenerational correlation
   to parental investment by construction. Is that a deliberate restriction? It matters for
   any statement about how much redistribution can move intergenerational mobility.

4. **Non-cognitive skill.** Daruich and Fernández carry a **vector** of skills: only
   cognitive skill enters wages, but both cognitive and non-cognitive enter the psychic
   cost of college. This model's `HC` is scalar. Worth flagging as a limitation, or is it
   out of scope?

5. **`R_0`, `w₀` and `α_θ` cannot all be estimated** (§6.5) — `R_0` and `w₀` are exactly
   confounded once `α_θ ≠ 0`. **Recommendation:** normalize `R_0 = 1`, drop it from the
   search, and estimate `α_θ` against an IGE moment. The alternative — fix `α_θ` from an
   outside regression and keep estimating `R_0` — is what Colas and Daruich & Fernández do.
   **Settle this before the next SMM run** regardless of the other answers; as the parameter
   set stands the search has an exactly flat direction.

6. **`WAGE_SCALING_FACTOR = 0.584`** — resolved by the recommendation: in logs it is an
   additive constant, so it folds into `ln w₀` and the duplicated `const` disappears
   (§1.2b). Flagging only because it silently rescaled every wage in every result to date.

---

## 12. What to read, and in what order

Page references are to the PDFs in `docs/papers/`, using the page numbers printed on the
pages. Total is roughly 35 pages of close reading.

### First — Lee and Seshadri (2019). ~10 pages.

The closest paper to this model: a childhood technology in parental time and goods whose
output is the initial condition of an adult human-capital process. Read it for the
*architecture*, not the estimates.

| Read | Pages | For |
|---|---|---|
| §II.A "Adulthood Human Capital Accumulation" | 860 | Eqs. (1)–(2): Ben-Porath, learning ability `a`, permanent shocks. **One page, read it twice.** |
| §II.B "Childhood Skill Formation" | 860–862 | Eqs. (3)–(7): the CES multi-stage technology, and the anchor `z`. **Footnote 9 is the identification warning of §6.5.** |
| §II.C "Recursive Formulation of Decisions" | 862–865 | How college (`j = 3`) and work (`j = 4+`) use the *same* technology with different time allocation `n` |
| Intro para on Ben-Porath calibration | 858 | Where the exponent `β` comes from |

### Second — Daruich and Fernández (2023). ~6 pages.

The most explicitly documented estimation recipe of the four. Read it for the *procedure*.

| Read | Pages | For |
|---|---|---|
| §2 "Wage process" and "The production function" | 7–8 | Eqs. (2)–(4): rental price × efficiency units, and the GE determination of `w_e` |
| §3 "Wage Process and Return to Skills" | 14–15 | **The three-step estimation. This is the template for §6.4.** |
| §3 "School Taste" | 16 | Eq. (14): psychic cost in skills and parental education |
| Appendix Tables B3, B4 | appendix | The age-profile and return-to-skill estimates |

### Third — Colas, Findeisen and Sachs (2021). ~8 pages.

Read the **online appendix before the main text** — the wage material is almost all there.

| Read | Pages | For |
|---|---|---|
| Online appendix §3.3 "Wage Estimation" | appendix 16–18 | Eq. (11), Table 2, the two-step procedure, and the earnings→wage mapping through the labour-supply FOC |
| Online appendix, endogenous ability | appendix 39 | **`θ = α₀ + α₁ ln θ̂` — anchoring one skill measure into another's units. Two paragraphs, exactly your CDS problem.** |
| §3.2 "Estimation and Data" | 14–15 | Why NLSY79 and NLSY97 have to be combined — your CDS age-coverage problem |
| §3.1.2 "College Problem" | 12 | `κ_X = κ₀ + κ_θ log θ + κ_fem + κ_ParEd` |
| Footnote 31 | 23 | Why the ability loading drives the fiscal externality |

### Fourth — Adda, Dustmann and Stevens (2017). ~4 pages.

No childhood block, so read it only for the adult side: depreciation and
type-specific profiles.

| Read | Pages | For |
|---|---|---|
| "Skills and wages" | 304 | Eqs. (2)–(3) and footnotes 12–14: PT/FT accumulation, piecewise atrophy, the threshold `x̄` |
| "Occupation and labor supply" | 302 | What an "occupation" is — read `occupation` as `education` throughout |
| Table 3 and surrounding text | 312–313 | The estimates in §8, and **footnote 31 on the fan-out** |
| Unobserved heterogeneity | 301–302 | The `f^P_i` type approach — the alternative to measuring skill at all |

### Then, outside `docs/papers/` — three worth obtaining

Not in the repository; listed in priority order.

1. **Huggett, Ventura and Yaron (2011)**, *AER* 101(7), "Sources of Lifetime Inequality."
   The Ben-Porath life-cycle model with initial human capital and learning ability fixed at
   labour-market entry, and a decomposition of how much lifetime earnings inequality each
   accounts for. **This is the paper that tells you how much of adult inequality your `θ`
   can plausibly explain** — a direct external check on `α_θ`. Lee and Seshadri take their
   `β` from it.
2. **Cunha, Heckman and Schennach (2010)**, *Econometrica* 78(3). Read the sections on
   anchoring and on the measurement system. The general statement of §6.5: estimated
   technology parameters are **not invariant** to the anchoring choice.
3. **Ben-Porath (1967)**, *JPE* 75(4). About ten pages, and it makes everything in Lee and
   Seshadri §II.A obvious.

### Reading order if time is short

Lee & Seshadri pp. 860–862 (architecture) → Daruich & Fernández pp. 14–15 (procedure) →
Colas online appendix pp. 16–18 and p. 39 (anchoring). That is about five pages and covers
both questions in §6 and §7.

---

## 13. Sources

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
