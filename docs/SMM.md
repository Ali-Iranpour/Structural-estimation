# Estimation: SMM by TikTak

How the parameters are estimated, which moments are targeted, and — the part that
matters for the paper — **every decision the published algorithm leaves open, and
what we chose**.

```bash
cd code && julia --project=.. smm.jl                 # full run, ~2.3 h
cd code && julia --project=.. smm.jl --quick         # smoke test, minutes
cd code && julia --project=.. smm.jl --warm          # resume from the last estimates
```

Flags: `--sobol N`, `--restarts N*`, `--localmax`, `--polishmax`, `--warm`, `--quick`.

Results go to `output/data/smm_estimates.toml` with the git commit, the budget, the
estimated parameters and the moment fit.

---

## The optimizer

**TikTak**, from Arnoud, Guvenen & Kleineberg (2022), *Benchmarking Global
Optimizers* — the paper finds it the strongest performer both on test functions and
on their economic application. Reference implementation:
<https://github.com/serdarozkan/TikTak>. Ours is [`code/src/tiktak.jl`](../code/src/tiktak.jl),
standalone and reusable.

Two stages:

| Stage | What it does |
|---|---|
| **Global (pre-testing)** | Evaluate `f` at `N` Sobol' points covering the box, sort ascending, keep the best `N*` as seeds `s₁ … s_{N*}`. The rest are discarded. |
| **Local** | `N*` local searches. The `j`-th starts at `s̃ⱼ = (1 − θⱼ)·sⱼ + θⱼ·Z*`, where `Z*` is the best minimiser found so far. `θⱼ` rises with `j`, so early restarts explore and later ones exploit. Then one polishing search from the winner. |

Why a global method at all: the objective has failure regions (parameter draws where
the model does not solve) and the earlier BOBYQA multistart showed that a uniform
random start in 14 dimensions usually lands in one. A pure local method from a single
start cannot be trusted here.

---

## Decisions the paper leaves open

The paper specifies `θⱼ` only as "very small, possibly zero" early and "gradually
increased". It gives **no formula**, and the repo README does not either. These were
settled deliberately on **2026-08-07** rather than silently defaulted.

### 1. The mixing schedule `θⱼ`

```
θⱼ = clamp((j / N*)^0.5, 0.1, 0.995)
```

Square-root growth, the form used in the circulated TikTak implementations: `0.32` at
10% through the local stage, `0.71` at 50%. The `0.995` cap stops the seed point from
vanishing entirely, so even the last restart carries some new information.

Exposed as `(theta_p, theta_lo, theta_hi)` so the schedule can be changed without
touching the algorithm.

*Alternative considered:* linear `clamp(j/N*, 0.1, 0.995)`, which explores longer for
the same `N*`. Rejected as the less standard choice, not on evidence — we have not
benchmarked the two against each other on this objective.

### 2. The local optimizer

**Nelder-Mead at `ftol_rel = 1e-3`** — the paper's `TikTak-nm3` variant.

The paper's other variants use **DFNLS**, which has no maintained Julia binding, so
the `nm` variants are the faithful options available. Footnote 14 defines the
Nelder-Mead tolerance as the spread of function values across the simplex; NLopt's
`ftol_rel` is the closest available stopping rule, and that is what we use.

### 3. Polishing

**BOBYQA at `ftol_rel = 1e-10`**, capped at 500 evaluations. The paper applies a
polishing search with a stringent criterion after every global optimisation
(Section 3.3) and names DFNLS and/or BOBYQA for it.

### 4. Seeding — a deliberate deviation

Pure TikTak seeds **only** from Sobol points. We additionally force **the incumbent
calibration** into the pre-testing pool, where it competes on function value like any
Sobol point.

This is a departure from the published algorithm and is recorded as such. The reason
is practical: the incumbent scores `Q ≈ 3` while a random Sobol draw is usually far
worse or fails to solve outright, so without it a fixed budget can return something
worse than the calibration we already had. With it, the estimate is weakly better than
the starting point by construction.

*If you want the published algorithm exactly*, pass `extra_seeds = []` in
`estimate()`.

### 5. Budget — measured, not guessed

Nelder-Mead at `ftol 1e-3` in 14 dimensions **converges in ~295 evaluations**
(measured). The local stage therefore dominates:

| | evaluations | wall clock |
|---|---|---|
| `N = 1000` Sobol | 1,000 | ~22 min |
| `N* = 50` restarts × 295 | 14,750 | **~5.3 h** |
| `N* = 25`, capped at 200 | 5,000 | ~1.8 h |

Defaults are `N = 1000`, `N* = 25`, local searches capped at 200 and the polish at
500 — about 6,500 evaluations, ~2.3 h. `N* = 25` is 2.5% of `N`, inside the paper's
1–10% guidance. Raise `--restarts` for a more reliable, longer run.

---

## Validation

`tiktak_selftest()` runs three checks, each of which can only pass if a *different*
part of the implementation is right:

| Check | Result |
|---|---|
| Sphere, d=10 — must hit the minimum to machine precision | `f = 4.4e-47` |
| Rastrigin, d=3 at `N = 2000` — dense lattice of local minima, solvable at this budget | `f = 0.0` exactly |
| Rastrigin, d=4 — TikTak vs the same code with `θ ≡ 0`, which *is* plain multistart | **0.995 vs 2.985** |

The third is the only one that tests the distinguishing feature — the mixing of each
seed with the incumbent — and TikTak wins by 3×.

Budgets matter more than they look: Rastrigin in 6d has on the order of `11⁶` local
minima in the box, so a small budget failing there is the function being hard, not the
optimizer being wrong.

---

## The objective

Weighted relative distance, `Q(θ) = Σⱼ wⱼ ((mⱼ(θ) − m̂ⱼ) / sⱼ)²` with
`sⱼ = max(|m̂ⱼ|, 0.05)`, so every moment contributes on a comparable scale regardless
of units. A failed solve returns a large **finite** penalty rather than `Inf` or an
exception, so a local search can still form a descent direction away from it.

**Common random numbers** throughout: every model is constructed with the same `seed`,
so the initial draws and shock paths are identical across evaluations. Without this
the objective is a step function of simulation noise and no derivative-free method
converges.

### Targets

Money targets are in **model units**; the display column is `×10` (thousands of
dollars), matching `ASSET_RESCALE` in `tables.jl`.

| Moment | Target (display) | Weight |
|---|---|---|
| `e_p` at t=1 / t=17 | 2.5 / 7.5 | 1.0 |
| `τ_p` at t=1 / t=17 | 0.475 / 0.275 | 2.0 |
| `i_c` first bargaining period / t=17 | 0.05 / 0.15 | 1.0 |
| `h_p` at t=1 / t=17 | 0.35 / 0.35 | 1.5 |
| Terminal parental assets | 250 | 2.0 |
| `c_p` at t=1 / t=17 | 37.5 / 52.5 | 1.5 |
| College share | 25% | 3.0 |

### Parameters

14 estimated. Bounded ones are searched on a linked scale (`log` for positive levels,
`logit` for shares) so the optimizer cannot leave the economically meaningful region.

`σ₁₀ σ₁₁ σ₂₀ σ₂₁ σ₃₀ σ₄₀ σ₄₁` (HC technology), `φ₂₀` (leisure), `φ₃₀ = λ₂₀` (weight on
log HC), `R₀` (TFP), `ω` (altruism), `κ_term` (taste for retained assets),
`college_cost`, `r`.

### β is fixed at 0.98, and `r` is estimated

**Decided 2026-08-07.** The Euler equation pins the consumption slope:

```
c_{t+1}/c_t = (β(1+r))^(1/ρ)
```

so the requested rising consumption profile is not a free moment — it is an arithmetic
consequence of `β`, `r` and `ρ`. At `ρ = 1.5`:

| `β` | `r` | `c_p` 37.5 becomes |
|---|---|---|
| 0.96 | 0.03 | **33.3** (falls) |
| 0.98 | 0.015 | 35.4 |
| 0.98 | 0.020 | 37.3 (flat) |
| 0.98 | 0.030 | 41.4 |
| 0.98 | 0.050 | **50.9** (hits the 52.5 target) |

β is held at **0.98** by instruction — it should never approach 1 — so `r` is the only
remaining lever and is estimated over `[0.010, 0.055]`. **A lower `r` and a rising
consumption profile are in direct conflict**; the search resolves that against the
other 11 moments, and the report prints the implied Euler growth factor so the
trade-off is visible in the output.

---

## Why it is fast enough

A naive SMM loop re-solves the whole model per draw. The parameters split into three
tiers by what they actually touch, and the code caches accordingly:

| Tier | Cost | Depends on | Cached on |
|---|---|---|---|
| **0** — child work + college value functions | ~2 s | `college_cost`, `r` only | those two, rounded to 0.1 and 0.005 |
| **1** — child transfer stage + terminal spline | ~1 s | adds `ω`, `κ_term` | all four |
| **2** — parent solve + both simulations | ~1 s | everything | not cached |

`ω`, `ψ_term` and `κ_term` enter `obj_transfer_*` and `terminal_value` and **nothing
else**, which is what makes Tier 1 separable. The typical draw pays Tier 2 alone —
roughly 6× cheaper than the naive version, which is what makes a few thousand
evaluations affordable.

Rounding the Tier-0 keys bounds that cache at ~180 entries over the whole box instead
of one solve per draw. `college_cost` has to be estimated despite being Tier-0,
because it sets `a_req[1]` and is the only direct lever on the college share.
