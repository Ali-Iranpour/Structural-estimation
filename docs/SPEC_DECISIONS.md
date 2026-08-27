# Specification Decisions (Phase 0 freeze)

Decisions taken before repairing code, so that no work is done on a specification that is
later discarded. Each is closed — implement against these, not against earlier drafts.

Frozen **2026-08-05**.

| # | Decision | Resolution |
|---|---|---|
| 0.1 | Belief correction (N6) | **Not an error.** Cancels exactly to `k₀ + 4b*`. Withdrawn. |
| 0.2 | Wage-equation `Age` units (P8) | **Code correct.** Stata re-indexes age 26 → model period 1. |
| 0.3 | `2 ×` on the parental wage (P7a) | **Intentional.** Regression is on the mean; `2 ×` is household earnings. |
| 0.4 | Retirement (C3) | **Removed.** `code/src/child_lifecycle.jl` is canonical. |
| 0.5 | ε timing (N1) | **ε observed before the transfer**; `E_ε` outermost. |
| 0.5b | `ā^P` / `δ_P` (N12) | **`δ_P = c_floor = 0.01`.** |
| 0.5c | `z₀` at separation (C6) | **Drawn from the stationary distribution.** |
| 0.6 | Child horizon `T` | **51** — ages 18–68 inclusive. |
| 0.7 | Wage shock process (C5) | **Keep the stationary AR(1) as a documented approximation.** |
| 0.8 | φ normalization (P7b) | **Drop the normalization claim.** `φ₂` is a scale, not a share. |
| 0.9 | College length | **Four years**, ages 18–21, work at 22. Code is right; the paper display is off by one. |
| — | N5, N7 | **Deliberate modelling choices, not errors.** See ERRORS.md. |
| — | C2 (psychic-cost exponent) | **Out of scope** by instruction. |

---

## 0.5 — ε timing

```
E_{ε₀} [ max_{d,tr} E_{z₀} [ W_d(tr; ε₀, z₀) ] ]
```

Nested in that order. `ε₀` is observed at the half period, `z₀` is not, so enrolment and
the transfer condition on `ε₀` but not on realized `z₀`. It is **not**
`max_{d,tr} E_{ε₀,z₀}[W_d]`, which would select the transfer before the shock is seen.

## 0.7 — Wage shock: approximation, and what it approximates

`wage2_styled.do` estimates (line 5 of §3):

```
u_t = eps_t (transitory) + sum iota (permanent random walk)
```

with `σ_ε = 0.1335` (transitory), `σ_ι = 0.1893` (permanent innovation), initial-shock
variance `0.2357`.

The model implements a **single stationary AR(1)**, `ρ = 0.95`, `σ_p = 0.2`, Tauchen-
discretized — no transitory component, and the persistent part is stationary rather than a
unit root. `σ_p = 0.2` is close to the estimated `σ_ι = 0.189`, so the model approximates
the permanent component and omits the transitory one.

**Decision: keep it, and say so.** This is a deliberate approximation, not an oversight.
It must be stated in the paper; the current text claims a random walk, which the code does
not implement and Tauchen cannot discretize.

---

## Required `model.txt` edits

> **Status: all eight applied to [`model.txt`](model.txt) on 2026-08-06.** Edits 6 (D1) and
> 2 (P7) were tracked as open findings; the other six were listed here but left `model.txt`
> describing a model the code does not implement, so they went in with them.

These are paper-prose changes and the wording below is a suggestion, not a requirement —
**reword freely.** What must not change is the content: each one corrects a place where
`model.txt` contradicted the code.

**1. Wage shock — replace the random-walk sentence** (§Time and Budget Constraints):

> Following the empirical wage literature, we assume these shocks follow a random walk

with something like:

> The estimated residual process is a permanent random walk plus a transitory component,
> with `σ_ι = 0.189` and `σ_ε = 0.134`. For tractability the model approximates this with a
> single persistent AR(1), `z_t = ρ z_{t-1} + ε_t`, `ρ = 0.95`, `σ = 0.2`, discretized by
> Tauchen; the transitory component is omitted.

**2. φ normalization** — delete "and normalized to sum to one" from the `(φ₁,φ₂,φ₃)`
sentence, and add: `φ₂` scales the disutility of labor and is not a budget share.

**3. `Age` units** — in eq. (wage_log), state that `Age_{it}` is **model time**,
`= biological age − 25`, so period 1 is parental age 26.

**4. Hours per parent** — state that `h_{p,t}` is hours *per parent* and household labor
income is `2 · w̄_t · h_{p,t}`, `w̄` being the mean parental wage.

**5. Transfer bound** — replace `0 ≤ tr ≤ a_{T_L}` with `0 ≤ tr ≤ a_{T_L} − ā^P`, and
define `ā^P` as the minimum asset the parent retains (set to one period's minimum
consumption).

**6. `V^{CD}_{T_L-1}`** — replace

> `V^{CD}_{T_L-1}(a,HC,m) = max_{tr} E_{ε₀,z}[·]`

with

> `V^{CD}_{T_L-1}(a,HC,m) = E_{ε₀}[ V^{CD}_{T_L}(a,HC,ε₀,m) ]`

and let eq. (T_L−1) take `E_z` only. The current pair double-counts the ε expectation and
implies commitment, contradicting the prose.

**7. College length** — the third case of `V^E_t` should be `t = 21`, not `t = 22`; the
continuation into `V^W` happens at 22.

**8. Retirement** — no change needed; the code now matches "no retirement stage".
Confirm the child horizon reads as ages 18–68 (51 periods).
