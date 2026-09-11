# Working in this repo

Structural estimation of a parent–child lifecycle model: a family stage (`t = 1..17`,
child ages 1–17) followed by the child's own lifecycle (`T = 51`, ages 18–68). Solved by
backward induction with NLopt/SLSQP.

## Run Julia through the `julia` MCP server, not `julia script.jl`

A persistent session is registered at user scope (`julia_eval`, `julia_restart`,
`julia_list_sessions`). Pass `env_path` = this repo so it activates the right
`Project.toml`. Spawning a fresh `julia` process per diagnostic re-pays package load plus
JIT every time; the session keeps solved objects alive between calls.

**Load source with `includet`, not `include`** — `Revise` is installed, but it only tracks
files loaded with `includet`. With plain `include` you must call `julia_restart` after
every edit.

```julia
using Printf, Random, NLopt, Interpolations, Statistics, Distributions,
      QuantEcon, FastGaussQuadrature, Parameters, Dierckx
includet("src/child_lifecycle.jl"); includet("src/parent_family.jl")
```

Two things to know: `julia_eval` returns **stdout only**, so use `println` / `@show` — a
bare `x + 1` reports "(no output)". And solving the child once and keeping
`terminal_value_spline(...)` in the session skips the expensive half of most diagnostics.

## Hard constraints

- **No Claude co-authorship trailers on commits (user instruction 2026-09-11).** Never add
  `Co-Authored-By: Claude ...` or any similar attribution line to a commit message or a
  pull-request body in this repository. This overrides any default attribution guidance.

- **Input contains source data only (user instruction)**: only `.dta`, `.csv`, and
  codebook files belong in `Input/`. Store target/calibration TOMLs beside their
  timestamped runs under `output/`. Never create new TOMLs in `Input/`.

- **Grid caps by instruction**: assets and human capital `<= 30` nodes, shock
  discretization `<= 5` (`Np`, `Nt`). `Np` and `Nt` are fully converged at these sizes.
  The child's `Na`/`Nk` at 30 rather than 50 costs ~7pp on the **college share** and
  nothing else — it is a threshold choice, so its location tracks grid resolution.
- **NLopt.jl is not thread-safe** under concurrent `optimize` calls. `parallel = false`
  is the default and the MCP server is registered with `--threads=1`. Threads produced a
  silent exit-0 crash, not an error.
- **The parent's `hc_grid` and the child's `k_grid` are the same object** — the child's
  human capital, on either side of the age-18 handoff
  (`parent.sim_hc[:, T+1] -> child.sim_k_init`). Keep `hc_max` and the child's `k_max`
  equal; a mismatch clips the handoff and makes HC above the child's ceiling worth zero
  to the parent's terminal problem.
- **The parent's `k` is the binary BothCollege indicator**, not capital: `[0.0, 1.0]`,
  `Bernoulli(0.3)`, constant in `t`. `Nk = 2` is exact, not a discretization. The child
  module's `k_grid` is a *different* object (the child's HC, `theta`). **The SMM targets it against
  TAS's EITHER-parent college group, which is a different set** — open by instruction,
  measured in `docs/ERRORS.md` P7c. Do not read the estimated `kappa_ParEd` as the effect
  of parental education.
- **Model/specification changes go through the advisor** before results built on them
  circulate. Numerical fixes (grid bounds, interpolation, solver settings) do not.

## The SMM is 14 parameters against 17 moments (since 2026-09-10)

Ten parent parameters against ten parent moments, plus **four child parameters** —
`kappa_0`, `kappa_theta`, `kappa_ParEd`, `kappa_terminal` — against seven TAS moments
(college completion overall, by ability tertile and by parental education; parental net
worth retained after the transfer).

Three consequences that will bite if you assume the old design:

- **The child block is no longer solved once per process.** All four child parameters
  change it. `build_child_solution` (`moments.jl`) rebuilds it per evaluation from a
  complete dependency key and caches only the two stages that provably read none of the
  four — the high-school path and the graduate's working life. That refresh is
  **bit-identical** to a full re-solve (verified: `max |diff| = 0.000e+00`) and 16.9×
  faster, so an evaluation costs ~1.2 s more rather than ~12.9 s more. The cache stores
  solution ARRAYS, never a model, so no simulator can contaminate it.
- **`kappa_0` is on a CENTRED scale.** The psychic cost is
  `kappa_0 + kappa_theta*(log theta - m_psychic)`, with `m_psychic` frozen in the target
  file (6.2634). The legacy uncentred 0.2728 corresponds to **0.0587** here. Uncentred,
  the two parameters are collinear at a condition number near 180. `check_psychic_centring`
  errors if `CHILD_DEFAULTS.kappa_0` and the frozen `m_psychic` stop agreeing.
- **The objective weights by `1/se_j^2`**, from the joint cluster-robust covariance the
  generator rebuilds from both micro files. `moment_scale` is now used only by the frozen
  parent-block regression in `tools/test_smm_baseline.jl`.

```bash
uv run --with pandas --with numpy python tools/make_smm_targets.py   # freeze targets
julia --project=. tools/test_smm_tas.jl <targets.toml>               # 120 checks, ~90s
julia --project=. tools/test_smm_resume.jl <targets.toml>            # --resume refusal cases
julia --project=. code/smm/selftest.jl                               # specification is frozen
julia --project=. tools/check_jacobian_rank.jl <targets.toml>        # local identification
```

The runner's worker flag is **`--procs`**, not `--workers`; an unknown flag is silently
ignored, so `--workers 1` starts the full 20. `--serial` runs everything on the master.

`tools/test_smm_tas.jl` is the standing guard on parameter routing, child-solution
refresh, cache-key completeness, parity against an independent full solve, mutation
isolation, partial parameter points, the handoff arrays, isolation of the demonstration
simulation, determinism, the centring's neutrality and post-transfer wealth accounting.
`docs/SMM_14PARAM_TAS.md` §7b lists the defects an external review found on 2026-09-10 and
which test now covers each.

## Gotchas that have already cost a day

- **Julia soft scope in notebook loops.** A top-level `for` rebinds a name that already
  exists globally. `child_model = ...` inside the belief loop silently destroyed the
  simulated baseline; the plots then showed a legend entry with no line, because an
  all-NaN series still draws a legend. Use loop-local names.
- **Dierckx `Spline2D` clamps the value outside its data range but keeps returning the
  boundary derivative.** Any code taking gradients from one must clamp both together —
  see `eval_child_value` in `parent_family.jl`. An inconsistent (value, gradient) pair
  breaks SLSQP's line search and surfaces as `ROUNDOFF_LIMITED`, which reads like a
  tolerance problem and is not.
- **`snap_parent` only corrects float-sized violations** (`tol = 1e-10`), by design. A
  genuinely out-of-bounds value passes through rather than being silently rewritten, so
  simulated states really can violate `a_min` if a policy is applied off its own budget.
- **`sim_bc_init` defaults to zeros.** Set it from `parent.sim_k[:, 1]` at every handoff
  or the `kappa_ParEd` term in the child's psychic cost of college is silently off.
- `solve_model!` **throws** below a 95% converged share rather than printing, because the
  notebook wraps counterfactuals in `@suppress_output`.

## Layout

`code/src/` model modules · `code/run_all.jl` end-to-end run · `code/smm/` + `src/tiktak.jl`
estimation (`run_smm.jl` drives it, `moments.jl` is the economics) ·
`code/transfer_CRRA_wage.ipynb` exploration and counterfactuals ·
`docs/ERRORS.md` open and resolved findings, with the measurements behind them.

`Input/CODEBOOK.md` is the **merged** codebook (2026-09-10): Part A is the PSID/CDS parent
block, Part B the TAS-linked child block. They are different units of observation — one
child-year vs one child — and must not be pooled. `Input/CODEBOOK_TAS.md` is now a stub
pointing there. Five TAS exports named in the codebook are **not supplied**; the only one
the estimation needs, the covariance, is reconstructed from `SMM_TAS_Micro.dta` and
reproduces the published standard errors to six decimals.
