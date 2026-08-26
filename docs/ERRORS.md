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

## 🟡 P12 — One counterfactual arm no longer solves

`σ₂₁ × 1.5` (education-expenditure slope) fails under the new HC block; the other 24 arms
in notebook cell 29 solve. Not yet diagnosed. Likely the same class as the earlier solver
failures — a state where the recalibrated technology drives an input to its bound.

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

## ⏸️ C2 — Psychic cost uses the wrong power

`kappa/(HC+1)^4` in code against `kappa/(HC+1)^2` in `model.txt`. At `HC = 1, kappa = 5`:
0.31 vs 1.25. **Deferred out of scope by instruction.**

## ⏸️ C8 — Duplicate `discrete_draw`; unused `Nt` dimension

In the **superseded** `child_lifecycle_ar1.jl` only. Harmless while it stays
reference-only. **Deferred out of scope by instruction.**

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
| **4.5** | **Add timeouts** to the notebook and PDF validation commands. `run_all.jl` can wait indefinitely on `pdflatex`. | open |
| **4.0** | **Reconsider how the child's HC enters the wage.** It is `w(1 + αk)` with `α = 0.08`, so the *proportional* return to skill decays as `α/(1+αk)` — 7.3% at `k = 3`, 3.1% at `k = 20`. The estimated *parental* wage equation puts education in logs. If the child's skill is measured on a comparable scale, `w·exp(αk)` would hold the proportional return constant. This is a specification question, and it is the main reason the marginal value of HC falls away faster than the marginal value of a transfer. | open |

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
