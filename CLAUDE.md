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

## The SMM is 16 parameters against 17 moments (since 2026-09-11; 14 / 17 from 2026-09-10)

**The baseline is the exp16b fit (promoted 2026-09-12).** `PARENT_DEFAULTS` and
`CHILD_DEFAULTS` carry run `2026-09-11_182836_exp16b` at full precision; the notebook and
`run_all.jl` build the child with `ConSavLaborCollege_AR1(; ..., CHILD_DEFAULTS...)`. The
next run's boxes: `kappa_0 [-3, 1]`, `kappa_ParEd [-1, 0.5]`, `sigma_4_1 [-0.05, 0.30]`
(`docs/SMM.md`, "The sixteen parameters").

Ten parent parameters against ten parent moments, plus **five child parameters** —
`kappa_0`, `kappa_theta`, `kappa_ParEd`, `kappa_terminal`, `sigma_eps` — and one more
parent parameter, `sigma_eta`, against seven TAS moments (college completion overall and
by parental education; the mean log-ability gap between completers and non-completers;
parental net worth retained after the transfer; the completion–wealth gap; the SD of log
HC at 17). **The 2026-09-11 additions are a preliminary experiment that has not been
through the advisor** — `docs/SMM.md` (the seven TAS moments, and the appendix) and
`docs/ERRORS.md` P13.

- `sigma_eta` is the SD of an i.i.d. zero-mean log shock in the HC technology
  (`hc_apply_shock` in `parent_family.jl`), integrated in the parent's continuation once
  per period (`eta_expected_interp`) and by quadrature on the age-18 handoff
  (`eval_child_value_eta`). `sigma_eta = 0.0` is the deterministic technology,
  bit-identical to the pre-shock solver; since 2026-09-12 `PARENT_DEFAULTS.sigma_eta` is
  the fitted 0.0315 and the search starts there (`SMM_START` is empty). `R_1` stays fixed at 0.
- `sigma_eps` is the SD of the college taste shock, now a struct field of the child model
  and part of `CHILD_ESTIMATED`; the cached work and graduate blocks are eps-free.
- `run_smm.jl` validates every flag, takes `--init-from <estimates.toml>` (warm start by
  name) and `--skip-polish` (an explicit bypass; `--polish-evals 0` is refused because
  NLopt reads it as no limit). Spec version `smm16_tas7_gap_v1`.

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
  file (6.2634). Uncentred, the two parameters are collinear at a condition number near
  180. `CHILD_DEFAULTS` (in `child_lifecycle.jl` since 2026-09-12, fitted at exp16b)
  records the `m_psychic` its `kappa_0` belongs to; `check_psychic_centring` errors if a
  target file carries a different one.
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

The runner's worker flag is **`--procs`**, not `--workers`; an unknown flag is now an
error before anything is written (it used to be silently ignored, so `--workers 1` started
the full 20). `--serial` runs everything on the master.

`tools/test_smm_tas.jl` is the standing guard on parameter routing, child-solution
refresh, cache-key completeness, parity against an independent full solve, mutation
isolation, partial parameter points, the handoff arrays, isolation of the demonstration
simulation, determinism, the centring's neutrality and post-transfer wealth accounting.
`docs/SMM.md`, "Fixes from external review (2026-09-10)", lists the defects an external
review found and which test now covers each.

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

`Input/CODEBOOK.md` is the **merged** codebook (regenerated 2026-09-11): Part A is the
PSID/CDS parent block, Part B the TAS-linked child block. They are different units of
observation — one child-year vs one child — and must not be pooled. `CODEBOOK_TAS.md` no
longer exists. The 2026-09-11 Stata rerun supplies `SMM_TAS_VCov.dta`, `_Funnel`,
`_TermWealth` and `_Weighted`, and moved parental net worth out of the by-age files into
`SMM_Assets_ByChildAge.dta` in **two-year bins** (`age_bin` = lower edge). The generator
reconstructs every TAS moment and the joint covariance from `SMM_TAS_Micro.dta` and
checks them against the supplied files to six decimals (`tools/test_smm_target_generator.py`).
