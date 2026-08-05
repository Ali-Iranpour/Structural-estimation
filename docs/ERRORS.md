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
python3 tools/nb_smoketest.py                 # notebook, all 64 cells, shrunken grids
```

⚠️ **The diagnostics currently understate the problem** — see X3 and C16. A green run does
not yet mean a sound solution.

---

## Open findings

| # | Issue | File | Severity |
|---|---|---|---|
| M2 | Notebook uses the known-inadequate child `a_max = 50` | notebook | 🟠 |
| C16 | Work solver has no upper domain constraint; **3.59% / 5.00%** of stored transitions leave the grid | child_lifecycle | 🟠 |
| X3 | `check_simulation` drops non-finite states before computing off-grid shares | diagnostics | 🟠 |
| P6 | Parent-only loop still stores unchecked NLopt results | parent_family | 🟠 |
| C15 | Heterogeneous child simulator silently clamps, no upper guard | parent_family | 🟠 |
| D1 | `model.txt` still places the max outside `E_eps` | docs | 🟠 |
| P5 | Linear continuation under SLSQP — **unverified**, not established benign | all | 🟠 |
| C14 | Interpolators built over arrays containing infeasible NaN rows | both | 🟡 |
| N15 | Unseeded belief draws; mismatched taste-shock seeds | notebook + src | 🟡 |
| P4 | `obj_last_period_full` still returns a finite `-1e12` sentinel | parent_family | 🟡 |
| X4 | `check_solution` allows blanket NaN and omits `sol_h_college` | diagnostics | 🟡 |
| P9 | Heterogeneous parent sim takes wage/tax from the base model | parent_family | 🟡 |
| N13 | Parent and child share one asset grid | child_lifecycle | 🟡 |
| M1 | Notebook writes no tables; table asset units inconsistent | notebook + tables | 🟡 |
| P7 | phi normalization; `BothCollege` share hardcoded | parent_family | 🟡 |
| C17 | Work bounds hardcode `0.01` instead of `model.c_floor` | child_lifecycle | 🟡 |
| C7 | `findfirst` / `discrete_draw` can return `nothing` | both | ⚪ |
| C8 | Duplicate `discrete_draw`; unused `Nt` dimension | child_lifecycle_ar1 | ⚪ |
| C12 | `sim_a[:, T+1]` never written | both | ⚪ |
| C2 | Psychic cost uses `^4`, model says `^2` | child_lifecycle | ⏸️ |

---

## 🟠 M2 — The notebook uses the known-inadequate child grid

All **17** `ConSavLaborCollege_AR1(...)` constructions in the notebook pass `a_max = 50.0`.
`run_all.jl` uses **100.0**, and its own comment records why: at 50, `check_simulation`
reported 2.87% of simulated child assets above the grid.

The notebook's baseline *and every counterfactual* therefore run on the grid already
measured as too small, while only the baseline script uses the corrected one.

**Fix.** Set `a_max = 100.0` at all 17 sites, or define the grid once and reuse it.

## 🟠 C16 — The work solver can leave the solved grid

`asset_constraint_work` constrains only `a' >= a_min`. There is **no** `a' <= a_max`, and
nothing constrains `k' = k + h <= k_max`. The continuation value is then evaluated
off-grid by `Line()` extrapolation.

Measured over 100,000 stored work transitions:

| | share leaving the grid |
|---|---|
| assets | **3.59%** |
| human capital | **5.00%** |

Forward simulation reports 0.00%, so this is invisible to the current diagnostics: the
*solution* leaves the domain even where the *simulation* does not.

**Fix.** Add explicit upper constraints, or widen the grids until stored transitions stay
inside — and add this solver-side measurement to `diagnostics.jl`, which does not make it.

## 🟠 X3 — `check_simulation` can report 0% on an all-NaN simulation

`diagnostics.jl:89` filters non-finite values *before* computing shares:

```julia
a, k = filter(isfinite, a), filter(isfinite, k)
below_a = count(<(m.a_min), a) / max(length(a), 1)
```

Demonstrated: a `sim_a` that is **96% NaN** reports `above_a = 0.00%` and passes. This
masks C12 (the entirely-NaN terminal asset column) and would mask any failure producing
NaN rather than an out-of-range number.

**Fix.** Count non-finite entries as violations, or report them as their own column, and
fail on them.

## 🟠 P6 — Parent-only loop still stores unchecked results

The terminal and adolescence loops validate `x_opt`/`minf` before storing. The
**parent-only** loop (`t = 7..1`, `parent_family.jl:555`) does not — it goes straight from
`optimize(...)` to `push!(itercounts, ...)` and storage. P6 was only partially fixed.

**Fix.** Apply the same finiteness check as the other two loops.

## 🟠 C15 — Heterogeneous child simulator clamps silently

`parent_family.jl:1396`, in `simulate_model_family_hetero!`:

```julia
sim_a[i, t+1] = max(a_next, a_min)
```

Pre-Phase-2 behaviour: it **rewrites the budget law** by replacing a genuinely negative
asset with `a_min`, and applies no upper guard on assets or human capital. It contradicts
the `snap` convention adopted in Phase 2 — correct only float-sized violations, leave real
excursions visible.

**Fix.** Use `snap` here too.

## 🟠 D1 — The paper still describes commitment timing

`model.txt:185` still has the max **outside** `E_eps`, the commitment timing Phase 0.5
rejected and the code no longer implements. The code computes
`E_eps[ max_{d,tr} E_z[W] ]`.

**Fix.** Replace with `V^CD_{T_L-1} = E_eps[ V^CD_{T_L}(., eps) ]`. This is edit 6 of the
eight in [`SPEC_DECISIONS.md`](SPEC_DECISIONS.md), none of which have been applied.

## 🟠 P5 — Linear continuation value under SLSQP (unverified)

`Gridded(Linear())` makes the continuation C0 but not C1, so `Interpolations.gradient` is
piecewise-constant with jumps at every knot while SLSQP builds a BFGS quadratic model from
it. `interp_vec = Vector{Any}` also forces dynamic dispatch in the innermost objective.

**Restored to High.** It was downgraded on a Bellman residual of `5.8e-13`, but that
residual re-evaluates the *stored* policy — it detects an inconsistent value, not a
suboptimal policy. Nothing has established that linear interpolation gives negligible
policy and moment differences.

**Fix.** Establish or refute it via Improvement 1, and compare against shape-preserving and
smooth interpolation.

## 🟡 C14 — Interpolators built over arrays containing NaN rows

`simulate_model_family!` (`child_lifecycle.jl:1108`) and especially
`simulate_model_family_hetero!` (`parent_family.jl:1293`) build interpolants from whole
transfer/college arrays, including infeasible rows holding `NaN`.

Measured: the transfer-stage feasibility threshold is **2.2774**, the first feasible asset
grid node is **4.6661** — a **2.39-wide band** where a state is economically feasible but
interpolates against NaN.

Latent today (current terminal assets do not land in the band) but structurally wrong.

**Fix.** Build every college/transfer interpolant over the feasible slice, consistently.

## 🟡 N15 — Unseeded belief draws and mismatched shock seeds

- **4** unseeded `rand(Beta(alpha, beta), simN)` calls in the notebook.
- Homogeneous simulators use `MersenneTwister(123)` (`child_lifecycle.jl:680, 1130`); the
  heterogeneous one uses `MersenneTwister(2222)` (`parent_family.jl:1312`).

With an otherwise identical one-bin model at `N = 500`, the seed difference alone moved the
college count from 21 to 23. Counterfactual differences still carry resampling noise, which
P2 was supposed to remove.

**Fix.** Seed the Beta draws from the model seed; use one stored set of taste-shock draws
everywhere.

## 🟡 P4 — A finite sentinel survives in `obj_last_period_full`

`parent_family.jl:634`:

```julia
if any(isnan, grad)
    grad .= -1e12
    return -1e12
end
```

The `-1e8` penalties were removed from the utility and HC functions; this one remains.
Because `-1e12` is *finite*, every downstream check accepts it. The finite-sentinel problem
is not fully removed.

**Fix.** Let the NaN propagate and fail, or floor the underlying quantity consistently in
value and gradient, as `LEISURE_FLOOR` does.

## 🟡 X4 — `check_solution` is too permissive

Two problems at `diagnostics.jl:58`:

1. `sol_h_college` appears in `allow_nan` but is **not among the arrays actually
   inspected**, so it is never checked.
2. NaN is allowed blanket-wide per array. Nothing verifies the NaN pattern coincides with
   the theoretical feasibility mask, so a solver failure *inside* the feasible region would
   pass silently.

**Fix.** Inspect `sol_h_college`; compare the NaN mask against `a_req` / `first_feasible_a`.

## 🟡 P9 — Heterogeneous parent sim mixes models

`parent_family.jl:1208` selects policies from the belief-specific `pm` but computes wage
and tax from `model` (= `parent_models[1]`):

```julia
wage      = wage_func(model, k, t, p_shock)
after_tax = model.tax_lambda * labor_pre ^ (1 - model.tau)
```

Harmless only while every belief-specific parent model shares wage and tax parameters —
true today, silently wrong the moment it is not.

**Fix.** Use `pm` throughout.

## 🟡 N13 — Parent and child share one asset grid

The transfer arrays are indexed on the **child's** asset grid but hold **parental** assets.
Two consequences:

1. `a_min = 0` is needed so the work branch's `tr = 0` is on-grid, but at `a = 0` the parent
   cannot retain `delta_P`, `a_term -> 0` and `kappa_term*log(a_term)` diverges — a genuine
   model singularity, handled by dropping that row from the terminal spline.
2. **Between `delta_P` and the first valid spline node** the terminal value is obtained by
   *extrapolation* rather than interpolation, because `terminal_value_spline` fits only over
   `valid_rows`. Same dead band as C14.

**Fix.** Separate the parental-asset grid from the child-asset grid; start the parent grid
above `delta_P` so neither the singularity nor the dead band exists.

## 🟡 M1 — Notebook writes no tables, and table units are inconsistent

`run_all.jl` writes three tables with provenance; the notebook writes **none** — zero
`table_*` and zero `write_manifest` calls. `belief_df` and the `@sprintf` belief summary
exist only as notebook output, which is stripped on commit.

**Fix the unit inconsistency first:** `table_outcomes` multiplies assets by `rescale = 10`;
`table_belief_groups` does not rescale at all. Wiring them in as-is would put assets in two
different units in the same paper.

## 🟡 P7 — Parameters that do not match the spec

- `phi_2_0 = 20.0` with `phi_1_0 = 1.0`, `phi_3_0 = 0.03` — sum 21.03 against `model.txt`'s
  "normalized to sum to one". **Decided (Phase 0.8): drop the claim**; the `model.txt` edit
  is in `SPEC_DECISIONS.md`, not yet applied.
- `Bernoulli(0.3)` for the `BothCollege` share is hardcoded and **still needs an empirical
  source** — it should come from the estimation sample, as the wage coefficients do.

## 🟡 C17 — Work bounds hardcode the consumption floor

`child_lifecycle.jl:294` and `:301` use a literal `0.01` for the consumption lower bound and
the initial guess rather than `model.c_floor`. `c_floor` defaults to `0.01`, so they agree
today — but changing the configured floor would silently re-introduce exactly the
inconsistency C1 was about.

**Fix.** Use `model.c_floor` at both sites.

## ⚪ C7 — `findfirst` / `discrete_draw` can return `nothing`

Broader than previously recorded. Three sites:

- `eps_indices = [findfirst(w -> w >= rand(rng), cum_weights) ...]` — `t_weight` sums to 1
  only up to floating-point error.
- `discrete_draw` has the same structure and failure mode.
- The heterogeneous simulator's `clamp(findfirst(...), 1, Nt)` **also fails** when the
  result is `nothing`: `clamp(nothing, ...)` is a `MethodError`, so the guard does not guard.

**Fix.** `something(findfirst(...), Nt)` before clamping; normalize the weight vectors to
sum exactly to 1.

## ⚪ C8 — Duplicate `discrete_draw`; unused `Nt` dimension

In the **superseded** `child_lifecycle_ar1.jl` only. Harmless while it stays
reference-only.

## ⚪ C12 — `sim_a[:, T+1]` never written

Both simulators guard the transition on `if t < T`, so the final column of a `T+1`-column
array stays `NaN`. The child consumes everything at `T`, so the correct value is `0`.
Currently masked by X3.

## ⏸️ C2 — Psychic cost uses the wrong power

`kappa/(HC+1)^4` in code against `kappa/(HC+1)^2` in `model.txt`. At `HC = 1, kappa = 5`:
0.31 vs 1.25. **Deferred out of scope by instruction.**

---

## Improvements to add

None implemented. Ordered by priority.

| Priority | Improvement |
|---|---|
| **8.0** | Replace the Bellman residual with a **true maximized-RHS residual**: re-optimize sampled states independently and compare the maximum against stored `V`. The present check re-evaluates the stored policy, so it detects inconsistency but not suboptimality. Also report Euler/FOC, complementary-slackness and constraint residuals. |
| **7.5** | **Grid refinement over at least three** asset/HC grids, reporting college share, transfer distribution, terminal parental assets and selected policy functions — not one scalar. |
| **7.0** | **Compare Tauchen and Rouwenhorst in the solved model.** At the child's own parameters Tauchen overstates the unconditional shock sd by **31.4%**. Phase 0.7 kept the AR(1) *process* as a documented approximation; the *discretizer* is a separate, still-open choice. |
| **6.5** | **Common stored preference- and wage-shock draws across every counterfactual**, then a **paired bootstrap** for differences. |
| **6.0** | **Explicit tests around the college-feasibility threshold**: just below, exactly at, inside the band between threshold and first grid node, and just above. |
| **6.0** | **Standardize monetary units** across simulation arrays, plots and tables. Parent `sim_wage` stores `2 x` the mean parental wage while labelled simply "wage". |
| **5.5** | **Require zero `MAXEVAL_REACHED`** for final estimation runs, or verify those solutions independently. The current 95% floor permits 5% un-converged policies to be stored. |
| **4.5** | **Add timeouts** to the notebook and PDF validation commands. `run_all.jl` can wait indefinitely on `pdflatex`; the smoke test can hang in plotting-library initialization. |

---

## Remaining work, in order

1. **X3, C16** — fix the diagnostics before trusting any other green result.
2. **M2** — put the notebook on the corrected grid.
3. **P6, C15, P4, C17** — finish the partially-applied fixes.
4. **N15** — complete P2 (seeded beliefs, one shared taste-shock draw set).
5. **C14, N13** — the feasibility dead band.
6. **D1 and the other seven `model.txt` edits** in `SPEC_DECISIONS.md`.
7. **M1** (after the unit fix), **X4**, **P9**, **P7**, **C7**.
8. **P5** — establish or refute via Improvement 1.
9. **Run the notebook at production grids.** The smoke test only proves it executes at
   `Na=12, Nk=12, Nt=4, simN=200` with 3 belief bins.

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
| P6 | NaN guard unreachable | Phase 1 |
| X1 | No accuracy diagnostics | Phase 1+4 |
| C11 | Transfer/simulation extrapolation | Phase 2 |
| C9 | Child simulation never clamps | Phase 2 |
| P3 | Parent states unclamped | Phase 2 |
| C6 | Stationary solve vs median simulation | Phase 3 |
| N1 | College choice outside the ε expectation | Phase 3 |
| N10 | Heterogeneous arms mismatched `y` | Phase 3 |
| P1 | Spurious `∂V/∂k` | Phase 3 |
| P4 | Objective/gradient inconsistent | Phase 3 |
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

### Withdrawn or not errors

| # | Why |
|---|---|
| N5 | `psi_terminal_belief_bin` unused — **deliberate modelling choice**, confirmed by the author. Beliefs shift the perceived college boost only; `ψ_term` is held common across belief types. |
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
