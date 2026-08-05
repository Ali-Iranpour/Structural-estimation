# Known Errors

Audit of the model code against [`model.txt`](model.txt), and the record of what was fixed.

**Open findings carry full detail below. Everything already closed is one line each in
[Resolved](#resolved) at the end of this file.**

Last updated **2026-08-05**.

| Severity | Meaning |
|---|---|
| 🟡 **Medium** | Affects interpretation, robustness, or reproducibility. |
| ⚪ **Low** | Cosmetic, maintainability, or latent. |
| ⏸️ **Deferred** | Real, but out of scope by instruction. |

No 🔴 Critical or 🟠 High findings remain open.

### Which files are live

| File | Status |
|---|---|
| `code/run_all.jl` | **LIVE** — one reproducible end-to-end run |
| `code/src/parent_family.jl` | **LIVE** — parent problem |
| `code/src/child_lifecycle.jl` | **LIVE, canonical** — child module. All child-side fixes go here. |
| `code/src/{paths,manifest,diagnostics,tables}.jl` | **LIVE** — infrastructure |
| `code/transfer_CRRA_wage.ipynb` | **LIVE** — interactive driver, counterfactuals |
| `code/src/child_lifecycle_ret.jl` | **SUPERSEDED** — reference only. Do not fix. |
| `code/src/child_lifecycle_ar1.jl` | **SUPERSEDED** — reference only. Do not fix. |

### Verification in place

```bash
cd code && julia --project=.. run_all.jl      # baseline path + diagnostics + tables + PDF
python3 tools/nb_smoketest.py                 # the notebook, all 64 cells, shrunken grids
```

Latest full run: parent converged share **1.0000**, Bellman residual **5.8e-13**,
**0.00%** of simulated states outside the grid, no NaN or Inf in any array.

---

## Open findings

| # | Issue | File | Severity |
|---|---|---|---|
| M1 | Notebook writes no tables | notebook | 🟡 |
| N13 | Parent and child share one asset grid; `a = 0` is a model singularity | child_lifecycle | 🟡 |
| P5 | Piecewise-linear continuation value under SLSQP | all | 🟡 |
| P7 | φ weights not normalized; `BothCollege` share hardcoded | parent_family | 🟡 |
| C7 | `findfirst` can return `nothing` | child modules | ⚪ |
| C8 | Duplicate `discrete_draw`; unused `Nt` dimension | child_lifecycle_ar1 | ⚪ |
| C12 | `sim_a[:, T+1]` never written | child modules | ⚪ |
| C2 | Psychic cost uses `^4`, model says `^2` | child_lifecycle | ⏸️ |

---

## 🟡 M1 — The notebook writes no tables

`run_all.jl` writes three tables with provenance. The **notebook** writes none: zero
`table_*` and zero `write_manifest` calls in code. The counterfactuals and the
subjective-expectations build live there, so their results reach the paper only as figures.

`belief_df` and the `@sprintf` belief summary exist **only as notebook output**, which is
stripped on commit — so they are not persisted anywhere.

**Fix.** Call the writers already in `code/src/tables.jl`:

```julia
table_belief_groups(belief_type, college_boost_belief_bin, model_parent_het.sim_a_init,
                    final_assets_het, final_hc_het, "belief_groups")
table_college_work(path_choice_hetro, "se_college_work_choice"; caption = "…")
write_manifest(tabpath(); experiment = "subjective expectations", seed = 1234)
```

---

## 🟡 N13 — Parent and child share one asset grid

The transfer arrays are indexed on the **child's** asset grid but hold **parental** assets
at separation. `a_min = 0` is required so the work branch's `tr = 0` is on-grid, but at
`a = 0` the parent cannot retain `δ_P`, `a_term → 0`, and `κ_term·log(a_term)` diverges.
That is a genuine singularity of the model, not a numerical artefact.

Currently handled by dropping the singular row: `terminal_value_surface` marks it `NaN`
and `terminal_value_spline` fits over `valid_rows` only.

**Fix.** Separate the parental-asset grid from the child-asset grid — parental resources
and the child's transfer need not share a support. The parent grid would start above `δ_P`
and never contain the singularity.

---

## 🟡 P5 — Piecewise-linear continuation value under SLSQP

`create_interp` (`parent_family.jl`) and `create_interpolator` (child) use
`Gridded(Linear())`, so `V_{t+1}` is C⁰ but not C¹ and `Interpolations.gradient` is
piecewise-constant with jumps at every knot, while SLSQP builds a BFGS quadratic model
from it. `interp_vec = Vector{Any}` also forces dynamic dispatch in the innermost
objective.

**Not currently binding**: the measured Bellman residual is `5.8e-13`, so at this grid the
solution is consistent with its own policy. Downgraded from High on that evidence.

**Fix, when convenient.** Compare rather than assume: linear + derivative-free optimizer,
shape-preserving (Schumaker), and smooth + monotonicity checks. Type the interpolator
vector concretely.

---

## 🟡 P7 — Parameters that do not match the spec

- **`phi_2_0 = 20.0`** with `phi_1_0 = 1.0`, `phi_3_0 = 0.03` — sum 21.03, while
  `model.txt` says `(φ₁,φ₂,φ₃)` are "normalized to sum to one". **Decided (Phase 0.8):
  drop the claim** — under CRRA with `η`, `φ₂` is a labor-disutility scale, not a share.
  The `model.txt` edit is listed in [`SPEC_DECISIONS.md`](SPEC_DECISIONS.md) and not yet
  applied.
- **`Bernoulli(0.3)`** — the `BothCollege` share is hardcoded and should be sourced to the
  estimation sample.

---

## ⚪ C7 — `findfirst` can return `nothing`

`eps_indices = [findfirst(w -> w ≥ rand(rng), cum_weights) …]`. `t_weight` sums to 1 only
up to floating-point error; if `cum_weights[end] = 1 − ε`, a draw above it yields
`nothing`, which then indexes an array.

**Fix.** `clamp(something(findfirst(...), Nt), 1, Nt)`, or normalize `t_weight` to sum
exactly to 1.

---

## ⚪ C8 — Duplicate `discrete_draw`; unused `Nt` dimension

In the **superseded** `child_lifecycle_ar1.jl` only: `discrete_draw` is defined twice
identically, and `sol_shape = (T, Na, Nk, Np, Nt)` carries an `Nt` dimension the work path
never uses (10× the memory). Harmless while that module stays reference-only.

---

## ⚪ C12 — `sim_a[:, T+1]` is never written

Both simulators guard the transition on `if t < T`, so `sim_a` is filled for columns
`1..T` while the array is allocated with `T+1` columns. The final column stays `NaN`
(exactly `1/(T+1)` of the array). The child consumes everything at `T`, so the correct
value is `0`, but `sim_a[:, end]` returns `NaN` to anything that reads it.

---

## ⏸️ C2 — Psychic cost uses the wrong power

`child_lifecycle.jl`: `psychic_cost = model.kappa / (k + 1.0)^4`. `model.txt` specifies
`κ_X = κ/(HC+1)²`.

At `HC = 1, κ = 5`: the model gives 1.25, the code 0.31 — 4× too small and decaying twice
as fast in `HC`. This term generates the human-capital gradient in college enrolment, i.e.
the link from childhood investment to the college margin.

**Deferred out of scope by instruction.** One-character fix when wanted.

---

## Remaining work

1. **M1** — wire the table writers into the notebook's counterfactual cells.
2. **Apply the eight `model.txt` edits** in [`SPEC_DECISIONS.md`](SPEC_DECISIONS.md).
   They are paper prose and were left for the author.
3. **Open decision — Tauchen vs Rouwenhorst.** At the model's own `ρ = 0.95, σ = 0.2,
   N = 5`, Tauchen overstates the unconditional sd by **31.4%** and persistence by 4.0%;
   Rouwenhorst is exact on both. Phase 0.7 kept the stationary AR(1) *process* as a
   documented approximation — that stands — but the *discretizer* is a separate choice and
   was not switched, since it changes results.
4. **N13** — separate the parent and child asset grids.
5. **P5, P7, C7, C12** as convenient; **C2** when taken back into scope.
6. **Run the notebook at production grids.** `nb_smoketest.py` proves it executes at
   `Na=12, Nk=12, Nt=4, simN=200` with 3 belief bins; the 20-bin build and the full grids
   have not been run end-to-end. One arm reports 0% college — pre-existing, worth checking
   at full scale.

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
