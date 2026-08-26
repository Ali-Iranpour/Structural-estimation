# Recommendations after the wage respecification

Written 2026-08-26, against commit `4a0833c`. Ordered by value, not by effort.
Everything here is backed by a number I measured, not a hunch.

---

## 1. Identification: nothing in the SMM speaks to wages

The wage now has **eight parameters** (`lnw0`, `beta_E`, `alpha_theta`, `alpha_thetaE`,
`gamma1`, `gamma1E`, `gamma2`, `gamma2E`) and the psychic cost three. **None is
estimated**, and none of the 13 targeted moments is a wage, earnings or dispersion
moment. The 15 estimated parameters are the HC technology, preferences, altruism,
`college_cost` and `r`.

That is defensible while the wage is fully calibrated from Daruich and Colas, and it is
what those papers do — estimate the wage outside the structural model and impose it.
But two consequences should be stated in the thesis rather than discovered by a referee:

- **`kappa_0` is the only free lever on the college margin besides `college_cost`, and
  it is not in the search.** With the college share moving 19.1% → 19.6% by luck rather
  than design, and the target at 25%, this is the first thing to add.
- The identification worry in `WAGE_PROCESS.md` §6.5 is **currently resolved by
  construction**: `alpha_theta` is fixed from outside and `R_0` is estimated, which is
  exactly one of the two admissible routes. If `alpha_theta` ever enters the search,
  `R_0` must leave it.

**Do:** add `kappa_0` to `PARS`. If wage moments ever become available, add the
college/high-school gap at entry *and* at peak — two moments, not one, or `gamma1E` and
`gamma2E` are not separately identified.

---

## 2. The psychic cost is near zero where the model actually lives

`kappa_X = 0.0462 − 0.0342·log θ` crosses zero at **θ = 3.86**. Mean θ is **3.44**.

| | value |
|---|---|
| total discounted psychic cost at mean θ | 0.0166 |
| the `kappa_ParEd` offset at `BothCollege = 1` | 0.0268 |
| ratio | **1.61×** |

So the ability term has almost cancelled the constant across the range the model
visits, the psychic-cost channel is doing very little, and what survives is dominated by
the parental-education shifter — which then moves the college share by only 0.13pp
(45.67% → 45.80%). A large part of the sample is above the zero-crossing and receives a
psychic *benefit* from college.

This is a calibration artefact, not a modelling result: the level was fitted to
reproduce the old `kappa/(HC+1)^4` cost over a θ range the model no longer occupies,
because θ roughly tripled when the return to childhood HC rose.

**Do:** refit `kappa_0` over the θ range the model now visits, or estimate it (§1).

---

## 3. Free compute: the taste-shock dimension is 88% dead weight

Both work and college solution arrays are `(T, Na, Nk, Np, Nt)`. The `Nt` taste-shock
dimension genuinely varies **only for `t ≤ t_college`, 4 periods of 51**. Everywhere
else it is written with `.=` and read at index 1.

At production grids that is **51 MB per array, six arrays, of which about 88% is
wasted**. It also costs cache traffic in the inner SLSQP loop, which is the hot path.

This got more expensive with the respecification, since the college arrays now hold the
`E = 1` graduate working life for `t > t_college` and are degenerate in `Nt` there too.

**Do:** carry `Nt` only on the study-year slice. Roughly 8× less memory in the child
block and a faster inner loop, for no change in results. This is the single largest
computational win available and it is behaviour-preserving.

---

## 4. The child's `Np = 5` has never been convergence-tested

`GUIDE.md` records that raising the **parent's** `Np` from 3 to 7 moved the college
share **17.85% → 22.40%**, while doubling any *state* grid moved it by ≤ 0.15pp. The
shock grid is by far the most consequential numerical choice in the model.

The child still runs at `Np = 5`. `run_all.jl` overrides `Na`, `Nk` and `Nt` but not
`Np`, and no study of it appears anywhere.

**Do:** run the child at `Np ∈ {5, 7, 9}` and report the college share. If it moves
like the parent's did, every number in the thesis is grid-dependent in a way nobody has
measured. This is cheap and it is the highest-risk unknown in the numerics.

---

## 5. `E[z] = 1.2235`, so `lnw0` is not the log mean wage

`p_grid = exp.(mc.state_values)` with `E[log z] = 0` gives, by Jensen,
`E[z] = exp(σ_y²/2) = 1.2235` at ρ = 0.95, σ = 0.2. The level is absorbed into `lnw0`,
which I calibrated empirically, so **nothing is wrong numerically** — but `lnw0` now
carries a 22% shift that has nothing to do with wages, and it will silently move if
`sigma_p` or `p_ar1` are ever changed or estimated.

**Do:** centre the grid, `p_grid = exp.(mc.state_values .- σ_y^2/2)`, and re-derive
`lnw0`. One line, and it makes `lnw0` mean what its name says.

---

## 6. `Bernoulli(0.3)` for BothCollege now matters more than it did

`parent_family.jl:316` draws the both-college indicator from a hardcoded 0.3.
`ERRORS.md` P7b has this open: "the number is hardcoded and still needs an empirical
source."

It used to affect only the parent's wage. It now also drives `kappa_ParEd` in the
child's psychic cost, so it moves the college share as well as parental income.

**Do:** measure it in the estimation sample, as P7b already says. The share of
households where both parents hold a degree is a one-line calculation wherever
`wage2_styled.do` runs.

---

## 7. The terminal value is unbounded in HC, so `k_max` does economic work

Raising `k_max` from 6 to 8 moved max θ from 6.53 to **15.52** while the mean did not
move (3.47 → 3.45). The child's `k_max` sets the domain of the terminal-value spline the
parent optimises against, and `terminal_value = ψ·log(HC) + κ·log(a)` is unbounded
above, so what stops the highest-investing parents is a corner that moves with the grid.

Only 0.08% of children are affected, well inside the project's own 1% tolerance, so
this is not urgent. But a grid parameter should not be the thing that bounds investment.

**Do:** either bound the HC term in `terminal_value`, or report the `k_max` sensitivity
explicitly so the corner is visible rather than implicit. Related: the parent's own
`hc_max = 6.0` is now exceeded by 0.038% of its simulated HC.

---

## 8. Dead code to remove

- **`w_vec`** (`child_lifecycle.jl:289`). `wage_func` no longer reads it; the age
  profile is computed inline from `gamma1`/`gamma2`. The field, the `fill(w, T)` and the
  frozen-at-construction trap it carried can all go. `w` survives only as the seed for
  `lnw0`, which would be clearer as a direct `lnw0` keyword.
- **`college_boost`** — the field, the constructor keyword, and the `@unpack` in
  `simulate_model_child!` that no longer uses it. College buys `beta_E` now.

Both are the kind of vestigial parameter this codebase's own comments warn about.

---

## 9. Modelling, for the thesis rather than the code

- **Inherited ability is still absent.** Lee and Seshadri give ability an AR(1) across
  generations; Daruich and Fernández do the same for both skill components. Without it
  the model attributes the *entire* intergenerational correlation to parental
  investment by construction, which weakens any claim about how much redistribution can
  move mobility. This is the most substantive gap remaining.
- **College carries no monetary risk.** Colas's `v^{e*}` is unknown at enrolment with a
  higher variance for college; Daruich finds the same. There is a cheap version: draw
  the entering `z` from an education-specific variance, which costs no new state.
- **No dynamic cost of progressivity.** Stated in `WAGE_PROCESS_IMPLEMENTED.md` §1.4 and
  worth one sentence in the paper, since the tax's human-capital margin now runs only
  through parental investment and college enrolment.
