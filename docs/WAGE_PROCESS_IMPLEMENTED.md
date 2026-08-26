# The respecified wage process and psychic cost, as implemented

Companion to [`WAGE_PROCESS.md`](WAGE_PROCESS.md), which argues the case. This
records what is actually in the code, with the calibration and its sources.

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
