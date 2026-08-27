# Known Errors

Audit of the model code against [`model.txt`](model.txt), and the record of what was fixed.

**Open findings carry full detail below. Everything already closed is one line each in
[Resolved](#resolved) at the end.**

Last updated **2026-08-06**. Every finding below was reproduced against the current code
before being recorded; measured numbers come from a `Na=20, Nk=20, Nt=6, a_max=100` child
solve unless stated.

| Severity | Meaning |
|---|---|
| 🟠 **High** | Materially biases results, or hides failure so you cannot tell whether they are biased. |
| 🟡 **Medium** | Affects interpretation, robustness, or reproducibility. |
| ⚪ **Low** | Cosmetic, maintainability, or latent. |
| ⏸️ **Deferred** | Real, but out of scope by instruction. |

### Which files are live

| File | Status |
|---|---|
| `code/run_all.jl` | **LIVE** — reproducible end-to-end run (baseline path only) |
| `code/src/parent_family.jl` | **LIVE** — parent problem |
| `code/src/child_lifecycle.jl` | **LIVE, canonical** — child module |
| `code/src/paths.jl`, `manifest.jl`, `diagnostics.jl`, `tables.jl` | **LIVE** — infrastructure |
| `code/transfer_CRRA_wage.ipynb` | **LIVE** — counterfactuals |
| `code/src/child_lifecycle_ret.jl`, `child_lifecycle_ar1.jl` | **SUPERSEDED** — reference only. Do not fix. |

### Verification

```bash
cd code && julia --project=.. run_all.jl      # baseline + diagnostics + tables + PDF
python3 tools/nb_smoketest.py                 # notebook, every code cell, shrunken grids
```

Both are green. Latest full run: parent converged share **1.0000**, Bellman residual
**0.00e+00**, `V` monotone in assets at **100.00%** of adjacent pairs, simulated states
off-grid **0.01%**, college share **19.1%**.

The diagnostics no longer understate the problem: `check_simulation` counts
non-finite states, `check_solver_domain` measures the *solution* leaving the grid (which
forward simulation cannot see), and `check_feasibility_mask` checks the NaN pattern against
both theoretical masks. **One caveat remains — see P5.**

---

## Open findings

| # | Issue | File | Severity |
|---|---|---|---|
| P11 | HC block recalibrated for a declining `τ_p`; **college share collapsed 19.1% → 0.1%** | parent_family | 🟠 |
| P12 | `σ₂₁ × 1.5` counterfactual no longer solves under the new HC block | parent_family | 🟡 |
| P10 | Leisure restored (`φ₂` 20.0 → 0.8); `τ_p` level now targeted — see P11 | parent_family | 🟡 |
| P5 | Linear continuation moves policies — **child solver only; parent fixed** | child_lifecycle | 🟡 |
| P7b | `BothCollege` share hardcoded at `Bernoulli(0.3)`, no empirical source | parent_family | 🟡 |
| G3 | `create_focused_grid` builds a non-monotone grid when the range is under 3.0 | both | ⚪ |
| C2 | Psychic cost uses `^4`, model says `^2` | child_lifecycle | ⏸️ |
| C8 | Duplicate `discrete_draw`; unused `Nt` dimension | child_lifecycle_ar1 | ⏸️ |

---

## 🟠 P11 — The HC recalibration gave a declining `τ_p` but killed the college margin

**What was asked and delivered.** `σ₃₁` was `+0.06`, so self-productivity *rose* with age
(0.09 → 0.24) and the persistence chain from an early investment to `T` was
`≈ 0.15^16 = 1e-14`. Skill had no memory, so there was no reason to front-load, and `τ_p`
came out **rising** (0.004 → 0.058). The block was recalibrated together and now gives:

| t | 1 | 5 | 9 | 13 | 15 | 17 |
|---|---|---|---|---|---|---|
| **τ_p** | **0.400** | 0.352 | 0.310 | 0.258 | 0.196 | 0.082 |
| i_c | 0 | 0 | 0.161 | 0.094 | 0.071 | 0.032 |
| h_p | 0.195 | 0.246 | 0.279 | 0.315 | 0.378 | 0.491 |

Monotone decline from 0.40, passing 0.20 at t = 15. `h_p` holds at 0.290 against its 0.285
target. `i_c` is no longer flat *in time* either.

**What it cost, and this is unresolved.** Mean terminal parental assets fell **20.97 → 9.39**,
and the college/work frontier sits at parental assets of **46–55**. With the asset
distribution halved, almost nobody clears it: **college share 19.1% → 0.1%** (4 of 5,000).

The mechanism is a genuine trade-off, not a bug. Raising `φ₃`/`λ₂` from 0.03/0.3 to 1.0/1.0
makes the child's skill ~30× more valuable, so the parent pours resources into `τ_p` and
`e_p` instead of saving — and college is financed out of savings. **Time went from leisure,
not from work** (`l_p` 0.70 → 0.41, `h_p` unchanged), so labor income is intact; it is
education spending and the transfer that crowd out the asset stock.

**Raising `R_0` does not fix it** — tested: at `R_0 = 2.2` human capital reaches 11.7 and
the psychic cost `κ/(HC+1)⁴` falls to 0.0002, yet the college share is **0.05%, lower still**;
`R_0 ≥ 2.8` fails to solve. The binding constraint is the parental asset distribution
against the college threshold, not the psychic cost.

**The levers, and the choice is yours.** Lower `college_cost` or the `a_req` threshold so
college does not require the top of the asset distribution; raise `omega` so more of the
parent's resources arrive as a transfer; or accept a smaller `φ₃`/`λ₂` and a flatter `τ_p`.
Which one depends on whether the college share or the time-use profile is the moment you
most want to match — they are in direct tension here.

## ✅ P12 — Counterfactual arms failing to solve — **fixed 2026-08-27, 25/25 now solve**

Started as "one arm fails", grew to four as the arm set was corrected, and turned out to be
**three independent defects stacked on the same symptom** (`Period N: only X% of grid points
converged`). Each was found by measurement, not inspection.

**1. The child's terminal spline reported a slope where its value was flat.** Dierckx clamps
a `Spline2D`'s *value* outside its data range but keeps returning the *boundary derivative*.
Measured at `a_next = 10`, child `k_max = 8`:

| HC | V | dV/dHC |
|---|---|---|
| 8.0 | −7.164 | 1.080 |
| 10.0 | −7.164 | 1.080 |
| 15.0 | −7.164 | 1.080 |

So above the ceiling SLSQP was told another unit of HC pays 1.08 while the objective did not
move; the line search could never realize its predicted decrease and the solve ended
`ROUNDOFF_LIMITED`. **This was the dominant cause — it alone took 21/25 to 24/25.** Fixed by
`eval_child_value` in `parent_family.jl`, which evaluates value *and* gradient at the same
clamped point and zeroes the derivative where the clamp binds. Confirmed it was not a
resolution problem: `Nhc` 30 → 40 → 50 gave 22/25 every time.

**2. `hc_max` was below the solver's own domain.** The HC technology is unbounded above, so
the solver is asked for `HC_next` at every grid corner. Measured at the old `hc_max = 6.0`:
`HC_next` reached **10.02**, with **1.25%** of stored transitions off-grid and served by
extrapolation. The *simulation* never needed the room (mean 1.71, p99 4.11), which is why
raising the ceiling moved baseline moments by ~0.1%. Note `child_lifecycle.jl` had already
sized its own `k_max` to 8.0 because HC reaches 6.53 — the parent's ceiling was simply never
raised to match. Now `hc_max = 10`, matched by the child's `k_max = 10`.

Raising it is not monotone: at `hc_max = 25` with `Nhc` fixed the focused grid leaves only 6
nodes above 3.0, tail resolution collapses, and the failure mode flips from extrapolation to
`maxeval` (baseline 100.0% → 96.5%).

**3. One arm was genuinely explosive, not broken.** `σ₃₀ + 0.4` against `σ₃₀ = −0.36` gives
`σ₃ = exp(0.04) = 1.041 ≥ 1`, i.e. `HC_{t+1} ∝ HC_t^1.04` — no bounded solution exists. The
guard added for this rejects it correctly. Resolved by moving the baseline to `σ₃₀ = −0.90`
(`σ₃ = 0.407`, the ~0.4 originally asked for), which leaves the ±0.4 arm spanning 0.607 and
0.272 — both stable.

**Side effect, and its fix.** Lowering `σ₃` flipped `τ_p` from flat to rising (0.287 → 0.457):
with less carryover, investment near 18 matters relatively more. Offset by steepening `σ₁₁`
from −0.02 to −0.08, measured over `τ_p` at t = 1/9/17:

| `σ₁₁` | t=1 | t=9 | t=17 | |
|---|---|---|---|---|
| −0.02 | 0.287 | 0.208 | 0.457 | rising |
| −0.05 | 0.295 | 0.173 | 0.330 | rising |
| **−0.08** | **0.300** | **0.143** | **0.226** | **declining** |
| −0.11 | 0.303 | 0.117 | 0.150 | declining, τ_p₁₇ low |

The profile is U-shaped at every value — structural, since late investment sits closest to
the age-18 payoff — so this targets "declining overall", not monotone.

## ⚪ Naming: three different labels for the parent's college indicator — **corrected 2026-08-27**

`k` in `parent_family.jl` is the parent's **BothCollege indicator**: binary, drawn once from
`Bernoulli(0.3)`, constant in `t`, entering only `wage_func`. It was labelled "physical
capital" (struct fields), "Parent Human capital grid" (`k_grid`) and "Simulated capital"
(`sim_k`) in three different places. `Nk = 2` is exact, not a discretization.

Checked whether the dimension could be dropped: it cannot. `k_grid = [0.0, 1.0]` already *is*
the two types, and two types is the minimum that represents a binary. Separately, the child
module's `k_grid` is a **different object** — the child's human capital θ — and θ is load-
bearing there: it enters the wage as `(α_θ + α_θE·E)·log θ` with `α_θ = 0.654`, and the
psychic cost of college as `κ_θ·log k`. Removing it would sever childhood investment from
adult earnings. Comments corrected in both files; no code change.

## ✅ The age-18 handoff — **verified 2026-08-27**

`parent.sim_hc[:, T+1] → child.sim_k_init` is wired in both drivers (`run_all.jl:103`,
`smm.jl:361`), so the parent's `hc_grid` and the child's `k_grid` are connected as intended.
The two ceilings are now matched at 10.0; they had drifted apart (parent 15, child 8), which
clipped the handoff and, worse, made HC above the child's ceiling worth **zero** to the
parent's terminal problem, since the terminal spline is only defined over `k_grid`.

## ✅ N16 — Heterogeneous-belief child assets went negative — **fixed 2026-08-27**

27.2% of simulated child assets were negative, reaching **−192.8** against `a_min = 0`. Not a
plotting artifact and not a transfer shortfall: assets fell through college, hit exactly 0 at
the first post-college period, then **diverged monotonically** (−2.36, −4.76, −7.19, …).

**Cause — the belief was applied for too long.** The bias is meant to govern the college-vs-work
DECISION and the college years: a student compares the two paths believing the biased premium
holds for life, and consumes through college on that belief, but **on graduating observes the
true wage and re-optimizes against it**. `simulate_model_family_hetero!` instead kept using the
belief-specific graduate policy (`child_models[m].sol_c_grad`) for the whole working life,
while realized income always used the true `beta_E` from `base_child`. So a graduate went on
consuming against a premium they had already been proved wrong about, with no income behind
it — and nothing stopped them, because `snap_parent` by design (C15) only corrects
float-sized violations.

**Fix:** post-graduation policies now come from `base_child` (the true `beta_E`) via
`interp_c_grad_true` / `interp_h_grad_true`. This requires `base_child` to carry a college
solution, so the notebook now calls `solve_model_college!` / `optimal_transfer_college!` on it;
the simulator errors clearly if it is missing rather than reading unassigned arrays.

A budget-feasibility cap (`c ≤ resources − a_min`) was added alongside, since a belief-optimal
college-years policy can still in principle outrun resources. It reports through a warning
rather than being silent. The two together, measured:

| | negative assets | cap fires | mean assets at t=8 |
|---|---|---|---|
| before | 27.17% (min −192.8) | — | −0.308 |
| cap only (belief policy kept) | 0.00% | **23.4%** | 0.687 |
| **cap + correct timing** | **0.00%** | **0.19%** | **1.091** |

The cap falling from 23.4% to 0.19% is what identifies the timing as the real defect: with the
right policy, income and consumption agree and the cap is the safety net it should be, catching
only interpolation slack at the constraint. Graduates also accumulate faster, as they should,
having stopped spending against a premium they do not earn.

The homogeneous simulator is unaffected (policy and wage come from the same model): measured
19 of 104,000 entries below zero, all at −0.000, i.e. float-sized.

## ✅ N17 — The notebook never set `sim_bc_init` — **fixed 2026-08-27**

`sim_bc_init` defaults to `zeros(simN)` and appeared **zero times** in the notebook, while
`run_all.jl:106` has always set it from `parent.sim_k[:, 1]`. It feeds `pared_value_offset`,
the `kappa_ParEd * BothCollege` term in the child's psychic cost of college. So every child in
the notebook was treated as having non-college parents and the parental-education channel in
the college decision was silently switched off — the notebook and `run_all.jl` were solving
different models. Added at all 11 handoff sites.

## ✅ N18 — Baseline lines invisible in counterfactual plots — **fixed 2026-08-27**

**Root cause: Julia soft scope silently destroying the baseline.** The two belief cells run

```julia
for m in 1:num_bins
    child_model = ConSavLaborCollege_AR1(...)   # <-- rebinds the GLOBAL
```

A top-level `for` in Julia uses *soft scope*, so assigning a name that already exists in
global scope **rebinds the global** rather than creating a loop-local. `child_model` already
held the simulated baseline from cell 21. After the belief loop it held the **last belief
bin's model** — solved, but never simulated, so its `sim_*` arrays were still `fill(NaN, …)`.
`baseline_sim` was therefore all-NaN by the time the counterfactual plots ran, and an all-NaN
series in Plots.jl **still produces a legend entry while drawing nothing** — precisely the
reported symptom: "(Base)" appeared in every legend with no line anywhere.

Confirmed from the executed notebook rather than by inspection: the baseline's own plot
(cell 23) renders correctly with human capital at 2.99 / 2.59, while the heterogeneous panel
shows only 2.885 / 2.66 and its y-axis spans just ~2.65–2.89. Had the baseline series been
present, the axis would have stretched to 2.59–2.99.

Fixed by renaming the loop-local to `belief_child` in all 5 belief cells (`child_models[]`,
the collection, is untouched).

Two contributing defects, both real and both fixed alongside:

1. `extract_simulation_by_path` used bare `mean`, so a **single NaN** in a group returned NaN
   for the whole period. Necessary but not sufficient here — the data was genuinely gone — and
   still correct, since the child arrays legitimately carry NaN (agents on the other path,
   periods before a branch opens). Now NaN-robust.
2. The baseline was drawn in the **same colour** as the counterfactual, dashed, **thinner**
   (1.0 vs 1.5) and **after** it, so wherever the two nearly coincided it read as one line.
   Now thicker and semi-transparent (`lw = 4.0, alpha = 0.30`), a faint band behind the solid
   series.

Also removed two hardcoded `ylims` that could clip a series out of view (`(0,30)` on human
capital — which is what made the axis evidence above legible; and the consumption y-range
ignored the baseline), and the assets panel now includes any negative region plus a zero line.
The below-`a_min` warning threshold is 1e-3, not float epsilon: assets run 0–150, so ~1e-4 is
interpolation slack at a binding constraint, and warning on it just trains you to ignore the
warning.

## ⚪ Grid caps — applied 2026-08-27, with one measured cost

Capped by instruction at 30 (assets, HC) and 5 (shock nodes) everywhere. Measured:

| config | college% | hc18 | assets | e_p17 | t_p17 |
|---|---|---|---|---|---|
| Na50 Nk50 Nt10 Np7 (was) | **57.70** | 2.79 | 19.7 | 7.49 | 0.227 |
| Na30 Nk30 Nt10 Np7 | **50.75** | 2.80 | 19.1 | 7.56 | 0.229 |
| Na30 Nk30 Nt10 Np5 | 51.50 | 2.80 | 19.1 | 7.55 | 0.229 |
| Na30 Nk30 Nt10 Np3 | 50.75 | 2.80 | 19.1 | 7.54 | 0.229 |
| Na30 Nk30 Nt5 Np5 | 52.10 | 2.80 | 19.1 | 7.55 | 0.229 |

`Np` (Rouwenhorst) and `Nt` (Gauss-Hermite) are **fully converged**: 7 → 5 → 3 and 10 → 5 leave
every moment flat to the third digit, so `Np = 3` would also be defensible. The child's
`Na`/`Nk` 50 → 30 is **not** free: it moves the **college share 57.7% → 50.8%** while leaving
every other moment unchanged. The college margin is a threshold choice, so its location moves
with child asset/HC resolution — worth remembering when reading the college-share results.

## 🟡 P10 — Parental leisure restored; the calibration tension it exposes is open

**Fixed.** `model.txt` specified `U_p = φ₁ log c + φ₂ log l_p + φ₃ log HC` with
`l_p = 1 − h_p − τ_p`. The code had `−φ₂ h_p^(1+η)/(1+η)` instead, which dropped `τ_p` out
of the parent's preferences entirely — `util_parent` returned the identical value at
`τ_p = 0.05` and `τ_p = 0.90`. Parental time with the child was free.

The parent's leisure is now `φ₂ · l_p^(1−η)/(1−η)`, CRRA rather than log, with
`l_p = 1 − h_p − τ_p`. `η` is repurposed from the Frisch curvature to the leisure curvature;
it had no other use. `φ₂` goes from **20.0 to 0.8**: it used to scale a labor disutility and
now weights a leisure CRRA, a completely different magnitude. 0.8 reproduces the old
simulated labor supply almost exactly — mean `h_p` **0.2860** against **0.2848** — which is
the one moment that can be held fixed while the term is restored.

What it fixed, measured:

| | before | after |
|---|---|---|
| `corr(τ_p, h_p)` at t=15 | **+0.603** | **−0.999** |
| mean `l_p` | 0.3667 | 0.7029 |
| min simulated `l_p` | **0.00000** (at the corner) | 0.6112 |
| `h_p + τ_p` at its bound | **35.29%** of states | never |

The sign is the point. Before, parents who worked *more* also spent *more* time with the
child, because the time constraint was binding and `τ_p = 1 − h_p` was mechanical. It is now
a genuine interior trade-off at every period.

### The open part: `τ_p` is not pinned by `φ₂`

Mean `τ_p` falls from 0.348 to **0.011** — about 16 minutes a day, which is too low
empirically. It cannot be fixed with `φ₂`: over `φ₂ ∈ [0.05, 3.0]` — a 60-fold range —
`τ_p` stays between 0.005 and 0.023, because the first-order condition

    φ₂ · l_p^(−η)  =  β · ∂V/∂HC · HC_next · σ₁ / τ_p

scales with `φ₂` on both sides. `τ_p` is set by `σ₁` and by how much the child's skill is
worth, not by the price of time.

Raising `σ₁₀` does lift `τ_p`, but it destroys `HC`:

| `σ₁₀` | −1.8 | −0.9 | 0.0 | +0.7 | +1.2 |
|---|---|---|---|---|---|
| mean `τ_p` | 0.011 | 0.025 | 0.069 | 0.102 | 0.001 |
| mean final `HC` | 2.55 | 1.64 | 0.41 | 0.35 | **0.00** |

That is Cobb-Douglas with inputs below one: `HC_next = exp(… + σ₁ log τ_p + …)` and
`log τ_p < 0`, so a larger elasticity *reduces* output. `τ_p` and `HC` cannot both be
matched by moving `σ₁` alone.

**What this needs is a calibration decision, and it is yours.** The levers that move `τ_p`
without collapsing `HC` are the units of the production inputs (`R_t` and the scale of
`τ_p`, `e_p`) and the weight on the child's skill (`φ₃`, `λ₂`, `ψ_terminal`). Which moments
you want to match — parental time use, the HC distribution, or both — determines the answer.

## 🟡 P5 — Linear continuation moves policies (parent side fixed)

**Parent solver: fixed.** `create_interp` now returns a `SmoothContinuation` — cubic in
`(a, hc)`, linearly blended in the binary `k`. This was the cause of the ragged policy
plots; see the commit for the ruled-out alternatives and the before/after reversal counts.

**Child solver: still open, and still measured as real.**

`Gridded(Linear())` makes the continuation C0 but not C1, so `Interpolations.gradient` is
piecewise-constant with a jump at every knot while SLSQP builds a BFGS quadratic model from
it. This was previously downgraded on a Bellman residual of `5.8e-13` and then restored to
High because that residual re-evaluates the *stored* policy — it detects an inconsistent
value, never a suboptimal one.

**It has now been tested directly and it is confirmed.** Two new diagnostics do it:

- `bellman_optimality_residual` re-optimizes sampled states from four starts against the
  same continuation and compares the maximum against the stored `V`.
- `continuation_interpolation_test` solves each sampled state twice — once against the
  linear continuation the solver uses, once against an interpolating cubic spline of the
  same value array — and compares the two optimal policies.

Measured across four grids (`Nt=6`, `a_max=100`, work path):

| grid | consistency residual | optimality residual (max) | states improved | max \|Δc\| | max \|Δh\| | mean \|Δh\| |
|---|---|---|---|---|---|---|
| 20×20 | 5.53e-13 | 1.98e-05 | 4.17% | 2.81 | 0.139 | 6.2e-03 |
| 30×30 | 5.58e-13 | 1.26e-05 | 3.33% | 1.18 | 0.135 | 4.7e-03 |
| 50×50 | 5.63e-13 | 1.50e-05 | 2.50% | 5.46 | 0.108 | 5.4e-03 |
| 80×80 | 5.65e-13 | 1.22e-04 | 3.33% | 1.45 | 0.170 | 3.5e-03 |

Three things to read off this:

1. **The consistency residual is blind to it.** It is 5.6e-13 at every grid — flat. It
   measures whether `V` matches the stored `(c,h)`, and it always will.
2. **The continuation choice moves labor supply by up to ~0.11–0.17** of the unit time
   endowment at some states — 11 to 17 percentage points. Consumption moves by up to
   several units.
3. **Refining the grid does not remove it.** Quadrupling the grid from 20×20 to 80×80
   leaves the maximum policy gap the same size.

The optimality residual itself is small (≈1e-5 relative, 2.5–4% of states improvable),
which is ordinary SLSQP slack. The continuation gap is the real finding.

**What this does *not* say.** The cubic spline is not "the right answer" — it is a second
reasonable approximation that disagrees with the first. Cubic splines overshoot and can
break the monotonicity and concavity of `V`, which matters here. The honest statement is
that **the numerical solution is not pinned down at those states**, not that linear is
wrong and cubic is right.

**Decision required.** Three options, in increasing cost:

- Accept it and report it, given the affected states are a minority and the moments used
  in the paper may be insensitive. *This needs a moments-level test that has not been run.*
- Switch to a **shape-preserving** scheme (monotone cubic / PCHIP), which is C1 without
  overshoot. This changes every result and requires re-running everything.
- Solve on a much finer grid where the two agree. The table above suggests this is
  expensive: 4× refinement bought nothing.

## 🟡 P7b — The `BothCollege` share is hardcoded

`parent_family.jl` draws `sim_k_init` from `Bernoulli(0.3)` — 30% of households have two
college-educated parents. The number is hardcoded and **still needs an empirical source**:
it should come from the estimation sample, as the wage coefficients in
`wage2_styled.do` do.

The other half of P7 — the `model.txt` claim that `(φ₁,φ₂,φ₃)` are "normalized to sum to
one" when the calibration is `(1, 20, 0.03)` — is **fixed**; the claim is dropped and the
reason stated.

**Fix.** Replace `0.3` with the share measured in the estimation sample, and record it in
`model.txt` alongside the wage estimates.

## ⚪ G3 — `create_focused_grid` silently builds an invalid grid on a narrow range

The focus point is hardcoded as `min + 3.0`:

```julia
a_grid = create_focused_grid(a_min, a_min + 3.0, a_max, Na, 0.3, 1.2)
hc_grid = create_focused_grid(hc_min, hc_min + 3.0, hc_max, Nhc, 0.8, 1.2)
```

If `max ≤ min + 3.0` the focus segment runs past the end of the grid and the result is not
monotone. Nothing checks it; the failure surfaces much later, inside Interpolations, as
`knot-vectors must be unique and sorted in increasing order`. Hit while testing
`hc_max = 3.0`, which is a value the coverage study says is worth trying.

**Fix.** Make the focus point a fraction of the range rather than a constant, or assert
`a_focus < a_max`.

## ✅ C2 — Psychic cost uses the wrong power — **superseded 2026-08-26**

`kappa/(HC+1)^4` in code against `kappa/(HC+1)^2` in `model.txt`. At `HC = 1, kappa = 5`:
0.31 vs 1.25. Deferred at the time, then **retired by replacing the power form
altogether**: the psychic cost is now `kappa_0 + kappa_theta*log(theta)
+ kappa_ParEd*BothCollege`, the log form both Colas (2021) and Daruich & Fernández
(2023) use. There is no exponent left to disagree about.

⚠️ Note for calibration: `kappa_0 + kappa_theta*log(theta)` crosses **zero at
theta = 3.86** while mean theta is 3.44, so a good part of the sample receives a
psychic *benefit* from college and the channel is doing little work. `kappa_0` is the
natural lever and belongs in the SMM parameter set. See
`docs/WAGE_PROCESS_IMPLEMENTED.md`.

## ✅ C8 — Unused `Nt` dimension — **fixed 2026-08-26 (in the LIVE file)**

The original finding was scoped to the superseded `child_lifecycle_ar1.jl`. The same
waste was present in the **live** `child_lifecycle.jl` and is now removed.

The taste shock `eps_0` enters at `t = 1` on the college branch only. Every other
solution slice was written with `.=` across the whole `Nt` dimension and read back at
index 1, so all six solution arrays carried `Nt` copies of identical data — about
**88% waste**, 306 MB of 350 MB at production grids. Three shapes replace one:

| array | shape | why |
|---|---|---|
| `sol_*_work` | `(T, Na, Nk, Np, 1)` | eps-free |
| `sol_*_grad` | `(T, Na, Nk, Np, 1)` | eps-free; **new** — the graduate's working life, split out of the college arrays |
| `sol_*_college` | `(t_college, Na, Nk, Np, Nt)` | the study years, the only place eps varies |

**Verified behaviour-preserving, not assumed.** Work-array sums fell by a factor of
exactly `Nt` (5.000000 on the test model), which is pure duplicate removal, while
`sol_tr_work`, `sol_tr_college`, the college share and every `sim_*` array came back
bit-identical. Measured **18.36 MB → 2.56 MB**, a 7.2× cut, and 42.6 MB for nine arrays
at production grids.

This matters for estimation: `smm.jl` caps its caches (`TIER0_CAP = 32`) *because* of
the per-child footprint, having once reached 5.5 GB on an 8 GB machine. That cap can now
be raised several-fold.

⚠️ One consequence: `sol_*_grad` is NaN for `t <= t_college` by construction, since a
graduate has no working life before then. Rather than blanket-allow NaN there and lose
the ability to detect a real solver failure, **`check_grad_mask` asserts the pattern**,
the way `check_feasibility_mask` does for the college arrays. It reports
`0 NaN where solved, 0 finite where unsolved (of 102000)`.

## 🟠 SMM readiness after the wage respecification — **assessed 2026-08-26**

`smm.jl` **runs**: a smoke run (`--quick --sobol 4 --restarts 1 --localmax 3
--polishmax 3`) completes with exit 0 through Sobol, local and polish stages, and writes
`smm_estimates.toml`. Mechanically the pipeline is estimation-ready. Statistically there
are three problems, in order of severity.

**1. The model is under-identified: 14 parameters, 12 moments.**

| | count |
|---|---|
| estimated parameters | `sigma_1_0/1_1/2_0/2_1/3_0/4_0/4_1`, `phi_2_0`, `phi_3_0`, `R_0`, `omega`, `kappa_terminal`, `college_cost`, `r` = **14** |
| targeted moments | `e_p`×2, `tau_p`×2, `i_c`×2, `h_p`×2, terminal assets, `c_p`×2, college share = **12** |

SMM needs at least as many moments as parameters. With 14 > 12 the objective generically
has a two-dimensional set of exact minimisers, so the reported optimum is one point on a
manifold and its standard errors are not defined. **No amount of tuning fixes this — it
needs moments, not effort.** Adding `kappa_0` (recommended on other grounds) makes it
15 vs 12 and worse; add moments first.

**2. The incumbent calibration is now far from the data.** The baseline objective is
**Q = 111.80**, against the roughly Q ≈ 3 recorded in `smm.jl`'s own comments before the
respecification. Four Sobol draws already reach Q = 26.4. Re-estimation is not optional
tidying; the current parameter vector is simply no longer a good fit.

**3. ⚠️ A latent cache trap.** `tier0` keys on `(college_cost, r)` only, because those
were the only estimated parameters entering `solve_model_work!`/`solve_model_college!`.
The wage parameters and the psychic-cost parameters now also enter those solves. They are
currently constructor defaults and never varied, so the cache is correct **today** — but
**adding any of them to `PARS` without extending `ccost_key`/`r_key` would silently
return a child solved at the wrong parameters.** This is the kind of failure that
produces plausible, wrong estimates rather than an error, so it should be fixed
pre-emptively if `kappa_0` or `alpha_theta` is ever estimated.

Also unchanged from before: no standard errors, no weight-matrix estimation, no
over-identification test, and hand-set weights (1.0–3.0). `SMM.md` already records these.

## ✅ Child `Np` convergence — **tested 2026-08-26, converged**

`GUIDE.md` records that raising the **parent's** `Np` from 3 to 7 moved the college
share **17.85% → 22.40%**, making the shock grid the most consequential numerical choice
in the model. The equivalent study had never been run for the **child**, which has always
run at `Np = 5` (`run_all.jl` overrides `Na`, `Nk` and `Nt` but not `Np`). That was the
largest untested numerical risk in the codebase. It is now measured, at production grids,
on the full family run:

| `Np` | college share | mean θ | mean wage | child solve | memory |
|---|---|---|---|---|---|
| 3 | 0.2002 | 3.4477 | 50.554 | 23.2 s | 25.6 MB |
| 5 | 0.2000 | 3.4464 | 50.795 | 41.4 s | 42.6 MB |
| 7 | 0.2002 | 3.4458 | 50.720 | 55.9 s | 59.6 MB |
| 9 | 0.1998 | 3.4457 | 50.737 | 71.5 s | 76.7 MB |

**The child's shock grid is flat.** The college share spans 0.04pp across `Np ∈ {3,…,9}`,
mean θ 0.06%, mean wage 0.5%. `Np = 5` is safely converged and even `Np = 3` would do.

This is the **opposite** of the parent's result, and the contrast is the point: the
parent's investment policy is sharply nonlinear in its shock, whereas the child's
consumption-savings-labour problem has a value function smooth in `z`, which Rouwenhorst
integrates accurately at any `N` since it matches the first two moments exactly.

**Opportunity:** dropping the child to `Np = 3` nearly halves the child solve, 41.4 s →
23.2 s, for a 0.02pp change in the college share. Since `smm.jl` pays the child solve
once per Tier-0 cache entry, that is close to a free 1.8× on the dominant cost of an
estimation run.

---

---

## Grid design: what the refinement study found

Run one-at-a-time from the production configuration, 2,000 agents.

**The state grids are converged. Doubling any of them does nothing:**

| change | college % | Δ college |
|---|---|---|
| baseline | 17.85 | — |
| child `Na` 50→100 | 18.00 | +0.15 |
| child `Nk` 50→100 | 17.95 | +0.10 |
| parent `Na` 30→60 | 17.95 | +0.10 |
| parent `Nhc` 30→60 | 17.85 | **+0.00** |
| **halve all four** | **7.45** | **−10.40** |

Halving breaks it badly, so the current sizes sit right at the convergence point — well
chosen, not wasteful. Do not spend nodes here.

**The shock grid was the binding approximation.** Parent `Np` 3 → 7 moved the college share
by **+4.55pp** and mean terminal parental assets by **+8.9%** — thirty times more than
doubling any state grid. It converges by 5–7 (Np=5: 22.00%, 7: 22.40%, 9: 22.30%, 13:
21.80%). The reason is that Tauchen at N=3 is not a mildly coarse version of the process,
it is a different one:

| ρ=0.9, σ=0.1 | Tauchen sd err | Tauchen persistence err | Rouwenhorst |
|---|---|---|---|
| N=3 | **+21.5%** | **+10.8%** | exact |
| N=7 | +17.1% | +0.18% | exact |
| N=11 | +6.6% | −0.17% | exact |

Both modules now use Rouwenhorst, and the parent uses `Np = 7`.

**Node placement is still poor, but it is headroom rather than error.** Measured against
where the simulation actually goes: parent `hc_max = 6.0` against a data p99 of 2.46 (only
3 of 30 nodes inside the IQR); `ap_grid` puts 39 of 50 nodes outside the p1–p99 range of the
transfers it indexes; child `a_grid` has 17 of 50 nodes above the data's p99. Since doubling
these grids changes nothing, reclaiming the waste buys accuracy the model does not currently
need — it is tidiness, not correctness. **Left undone deliberately.**

⚠️ **A latent bug blocks that clean-up.** `create_focused_grid` hardcodes the focus point at
`min + 3.0`, so any `hc_max ≤ 3.001` (or `a_max ≤ a_min + 3.0`) silently produces a
non-monotone grid and dies inside Interpolations with `knot-vectors must be unique and
sorted in increasing order`. Anyone narrowing a grid range will hit this first.

---

## Improvements to add

Ordered by priority. Improvement 1 is now **done** — it is what settled P5.

| Priority | Improvement | Status |
|---|---|---|
| ~~8.0~~ | True maximized-RHS Bellman residual, re-optimizing sampled states independently | **done** — `bellman_optimality_residual` |
| ~~7.5~~ | Grid refinement on real outcomes | **done** — and it found that the *state* grids are converged while the *shock* grid was not. See below. |
| ~~7.0~~ | Tauchen vs Rouwenhorst in the solved model | **done** — switched to Rouwenhorst in both modules. See below. |
| **6.5** | **Paired bootstrap** for counterfactual differences. Common draws are now stored and shared (N15), so the pairing is available; the bootstrap is not written. | open |
| **6.0** | **Explicit tests around the college-feasibility threshold**: just below, exactly at, and just above. The dead band between them is now zero wide (N13/C14), so this is a regression test for that. | open |
| **6.0** | **Standardize monetary units** across simulation arrays and plots. Tables are done — `ASSET_RESCALE` is defined once and both asset tables use it — but parent `sim_wage` still stores `2 ×` the mean parental wage while labelled simply "wage". | partial |
| **5.5** | **Require zero `MAXEVAL_REACHED`** for final estimation runs. The current 95% floor permits 5% un-converged policies to be stored; runs currently report 100% converged, so tightening the floor costs nothing today. | open |
| **7.0** | **Add moments to the SMM.** 14 parameters against 12 targeted moments; the model is under-identified. Wage/earnings moments would also let the eight wage parameters be estimated rather than calibrated. See the SMM readiness assessment above. | open |
| **6.0** | **Add `kappa_0` to `PARS`**, and extend the Tier-0 cache key when doing so. It is the only free lever on the college margin besides `college_cost`, and the psychic cost is currently near zero at mean theta. Do this *after* adding moments, not before. | open |
| **5.0** | **Centre the shock grid** so `E[z] = 1`. `p_grid = exp.(mc.state_values)` gives `E[z] = 1.2235` by Jensen, a 22% level shift silently absorbed into `lnw0`. Harmless now, but it moves if `sigma_p` or `p_ar1` are ever changed. One line: `exp.(mc.state_values .- sigma_y^2/2)`, then re-derive `lnw0`. | open |
| **4.5** | **Add timeouts** to the notebook and PDF validation commands. `run_all.jl` can wait indefinitely on `pdflatex`. | open |
| ~~4.0~~ | **done 2026-08-26.** Replaced by a log-linear wage with education and childhood HC separated; see `docs/WAGE_PROCESS_IMPLEMENTED.md`. Original note: **Reconsider how the child's HC enters the wage.** It is `w(1 + αk)` with `α = 0.08`, so the *proportional* return to skill decays as `α/(1+αk)` — 7.3% at `k = 3`, 3.1% at `k = 20`. The estimated *parental* wage equation puts education in logs. If the child's skill is measured on a comparable scale, `w·exp(αk)` would hold the proportional return constant. This is a specification question, and it is the main reason the marginal value of HC falls away faster than the marginal value of a transfer. | **done** |

---

## Remaining work, in order

1. **P10 (calibration half)** — `τ_p` = 0.011 is too low and neither `φ₂` nor `σ₁` fixes it.
   The levers are the units of the HC production inputs and the weight on the child's skill.
   This is the only open item that changes the economics.
2. **P5 (child side)** — apply the same shape-preserving continuation to
   `child_lifecycle.jl`, now that the parent side shows it is safe: `V` stayed monotone at
   59,160 of 59,160 adjacent pairs and the Bellman residual went to exactly zero.
3. **P7b** — get the `BothCollege` share from the estimation sample.
4. **G3** — two-line guard, needed before any grid range can be narrowed.
5. **Regenerate every table and figure.** The headline numbers moved a long way across this
   session (college share 13.1% → 19.1%); anything already drafted is stale.
3. **Improvement 7.0** — Tauchen vs Rouwenhorst in the solved model, not just in the
   discretization report.
4. **Improvement 7.5 / 6.5** — grid refinement on real outcomes, and the paired bootstrap.

---

## Resolved

One line each. Full detail is in the git history — every fix is in a commit whose message
explains it, on branch `fix/remove-retirement`.

| # | Issue | Closed in |
|---|---|---|
| C3 | Retirement not in the model | Phase 0 |
| N12 | `ā^P` placeholder | Phase 0 |
| N11 | Notebook cannot run top-to-bottom | Phase 1 |
| N2 | Diagnostics suppressed | Phase 1 |
| P2 | Unseeded RNG | Phase 1 |
| X1 | No accuracy diagnostics | Phase 1+4 |
| C11 | Transfer/simulation extrapolation | Phase 2 |
| C9 | Child simulation never clamps | Phase 2 |
| P3 | Parent states unclamped | Phase 2 |
| C6 | Stationary solve vs median simulation | Phase 3 |
| N1 | College choice outside the ε expectation | Phase 3 |
| N10 | Heterogeneous arms mismatched `y` | Phase 3 |
| P1 | Spurious `∂V/∂k` | Phase 3 |
| N3 | Invalid CEV (removed; welfare gaps + bootstrap SE instead) | Phase 5 |
| N4 | θ-experiment ω mismatch | Phase 5 |
| N8 | Model/label order swapped | Phase 5 |
| N9 | Stale τ labels | Phase 5 |
| T1 | `fmt_num` emitted scientific notation ≥1e6 | audit |
| T2 | `build_tables_pdf` could `\input` a stale copy of itself | audit |
| T3 | Dead `safe_maximum` / `AMIN` in `parent_family.jl` | audit |
| T4 | `bellman_residual` rebuilt interpolators per sample | audit |
| T5 | `simulate_model_child!` missing `a_min` in `@unpack` | notebook run |
| T6 | Cell 11 broke on the new terminal-value API | notebook run |
| T7 | `plot_family_counterfactuals` used before defined | notebook run |
| T8 | `simulate_model_hetero!` `@inbounds` over unvalidated `belief_type` | notebook run |
| C1 | `-Inf` sentinel → 30% NaN | pre-Phase 0 |
| C10 | Return codes never inspected | pre-Phase 0 |
| C4 | Asymmetric transfer optimization | pre-Phase 0 |
| **X3** | `check_simulation` dropped non-finite states before computing off-grid shares, so a 96%-NaN run reported 0.00% outside | **final sweep** |
| **X4** | `check_solution` omitted `sol_h_college` and allowed NaN blanket-wide; `check_feasibility_mask` now checks both theoretical masks | **final sweep** |
| **P4** | `obj_last_period_full` returned a finite `-1e12` sentinel that every downstream check accepted | **final sweep** |
| **P6** | The parent-only loop stored NLopt results unchecked | **final sweep** |
| **P9** | Heterogeneous parent sim took wage and tax from the base model, policies from the belief model | **final sweep** |
| **P7a** | `model.txt` claimed the φ weights sum to one; claim dropped, reason stated | **final sweep** |
| **C7** | `clamp(nothing, …)` is a `MethodError`, so neither guard guarded; weights now normalized to sum to exactly 1 | **final sweep** |
| **C12** | `sim_a[:, T+1]` was never written; the terminal period leaves no bequest, so it is 0 | **final sweep** |
| **C14** | Interpolants built across both NaN masks; the measured 2.39-wide college dead band is now 0.00 | **final sweep** |
| **C15** | Heterogeneous family simulator used `max(a_next, a_min)`, rewriting the budget law, with no upper guard | **final sweep** |
| **C16** | No upper domain constraint; 3.59% of asset and 5.00% of HC transitions left the grid. Now bounded — residual excursions are 9.99e-07 (NLopt constraint tolerance) and 1.00e-03 (labor lower bound) | **final sweep** |
| **C17** | Work bounds hardcoded `0.01` instead of `model.c_floor` | **final sweep** |
| **D1** | `model.txt` placed the max outside `E_ε`; all eight spec edits applied | **final sweep** |
| **M1** | Notebook wrote no tables, and `table_belief_groups` printed raw units under an `×10³` caption | **final sweep** |
| **M2** | Notebook ran every counterfactual on the known-inadequate `a_max = 50` | **final sweep** |
| **N13** | Parent and child shared one asset grid, forcing `a_min = 0` into the parental dimension where the terminal value diverges | **final sweep** |
| **N15** | Unseeded belief draws; taste-shock seeds 123 vs 2222 across simulators | **final sweep** |
| **T9** | Leisure-floor gradient cliff of 1.28e7 broke the parent solve; the log is now linearized below the floor (C1, bounded) | **final sweep** |
| **T10** | Parent-only loop floored `t_p`/`h_p` at 1e-6 against the full loop's 1e-4, letting `HC_next·σ₁/t_p` reach 1e6 | policy sweep |
| **T11** | Neither backward-induction loop clamped its warm start from `t+1` into the current box | policy sweep |
| **T12** | Parent `c` and `e` were boxed at a constant 100 regardless of the budget; SLSQP reached `a_next = −174` and the continuation was extrapolated 174 units past its edge | policy sweep |
| **T13** | Production logs gave `∂HC_next/∂x = HC_next·σ_j/x`, unbounded at the 1e-4 time floors; `TIME_FLOOR = 1e-3` is strictly slack and caps it 10× | policy sweep |
| **P5a** | Parent continuation was `Gridded(Linear())`, so `∂V/∂hc` was a step function and `τ_p` inherited the steps — the ragged policy plots. Now PCHIP | policy sweep |
| **P10a** | Parent's utility omitted `φ₂ log l_p`, so `τ_p` was free and `corr(τ_p,h_p)` came out **+0.60**. Leisure restored as CRRA, `φ₂` 20.0 → 0.8, correlation now **−0.999** | policy sweep |
| **I7.0** | Tauchen → Rouwenhorst in both modules; exact sd and persistence at every `N` | grid study |
| **I7.5** | Grid refinement on real outcomes; found the state grids converged and the shock grid binding | grid study |
| **G1** | Parent `Np` 3 → 7 — the binding approximation in the model (college share +4.55pp) | grid study |
| **G2** | Parent `a_max` 50 → 100; simulated assets reached 281.5, off-grid 0.43% → 0.10% | grid study |

### Tested and rejected

Alternatives that were measured and **deliberately not adopted**. Each one is a real
experiment with a number attached, not an untried idea.

| Tried | Measured | Why rejected |
|---|---|---|
| **Interpolating cubic** for the parent's continuation | Fixes the raggedness as well as PCHIP does | Overshoots: pushed `∂V/∂hc` to 13.1 at the low-`hc` edge, which times `HC_next·σ₁/τ_p` gave gradients of 2e4 and broke the `σ₃₁ × 1.5` arm with a NaN iterate. PCHIP is C1 *and* bounded by the neighbouring secants. |
| **Tighter solver** — `xtol_rel` 1e-4 → 1e-10, `ftol_rel` 1e-13, `maxeval` 40× | Reversal counts changed by **at most 1** | The raggedness was never solver noise. Six orders of magnitude bought nothing. |
| **Finer `Nhc`** — 20 → 40 → 80 | Reversals at t=15 went **5 → 10 → 15**, range shrank 0.013 → 0.011 | Made it *worse*. Cell-boundary artefact, not coarseness — more cells, more steps. |
| **Halving all state grids** (child `Na`,`Nk` 50→25; parent `Na`,`Nhc` 30→15) | College share **17.85% → 7.45%** | −10.4pp. The current sizes are at the convergence point, not above it. |
| **Doubling any state grid** | College share moved ≤ 0.15pp; parent `Nhc` 30→60 moved it **0.00** | No accuracy to buy. Compute belongs in `Np`, not here. |
| **Parent `a_max` 200** instead of 100 | Off-grid 0.10% → 0.05%; mean terminal assets 22.13 → 21.77 | Not worth the lost range resolution. 100 is enough. |
| **Raising `σ₁₀`** to lift `τ_p` toward its old level | `τ_p` 0.011 → 0.102, but mean final `HC` **2.55 → 0.35 → 0.00** | Cobb-Douglas with inputs below 1: `log τ_p < 0`, so a larger elasticity *reduces* output. `τ_p` and `HC` cannot both be matched this way. |
| **Re-ranging `hc_max`, `ap_grid`** to the ergodic distribution | ~⅓ of nodes sit outside the data's p1–p99 | Real waste, but since doubling the same grids changes nothing, reclaiming it is tidiness rather than correctness. Also blocked by the `create_focused_grid` bug below. |
| **`τ_p` priced at weight 0.5** as a diagnostic before the full P10 fix | `corr(τ_p,h_p)` went negative at every t, but raggedness unchanged | Confirmed the sign problem and the continuation problem were independent. Superseded by the real fix. |

### Withdrawn or not errors

| # | Why |
|---|---|
| N5 | `psi_terminal_belief_bin` unused — **deliberate modelling choice**, confirmed by the author: `ψ_term` is held common across belief types. **Dead code now removed (2026-08-26).** The `b_min`/`b_anchor`/`ψ_anchor`/`psi_from_belief_linear` block computed `psi_terminal_belief_bin` in notebook cells 40, 78, 79 and 80 and never used it — every model was built with `psi_terminal = 4.0`. It was also anchored on the `college_boost` scale, which no longer exists now that beliefs concern `β_E`, and it shadowed `m` immediately before `for m in 1:num_bins`. |
| N6 | Belief correction — **the original finding was wrong.** It cancels exactly to `k₀ + 4b*` for every belief; the claim mistakenly applied the correction to `k₀` rather than to `k_{t_college}`. |
| N7 | Res-vs-Exp arms asymmetric (child `y=1.08`, parent `y=1.2`) — **deliberate**, confirmed by the author. |
| P8 | `Age` units — **code is correct.** `wage2_styled.do:132` re-indexes age 26 → model period 1, so `β_age * t` is right. |
| C5 | Shock discretization — **resolved as a documented approximation** (Phase 0.7). The estimated process is permanent-plus-transitory; the model uses a stationary AR(1) deliberately. See the open Tauchen/Rouwenhorst decision above. |

### Phase log

| Phase | What it closed |
|---|---|
| **0** — freeze the spec | N6, P8, P7 (wage half), C3, N1 timing, N12, C6 decision, `T`, C5, φ, college length |
| **1** — deterministic and observable | N11, P2, P6, N2, X1 (minimal) |
| **2** — numerical domain | C11, C9, P3 |
| **3** — central equations | P1, N1, C6, N10, P4 |
| **4** — numerical accuracy | X1 (full), P5 downgraded |
| **5** — counterfactual design | N4, N3, N8, N9, M1 (partial), experiment definitions |
| **audit** — deep sweep | T1–T4 |
| **notebook execution** | T5–T8 |
| **final sweep** — everything but C2/C8 | X3, X4, P4, P6, P9, P7a, C7, C12, C14, C15, C16, C17, D1, M1, M2, N13, N15, T9; P5 established and measured |
