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
includet("code/src/child_lifecycle.jl"); includet("code/src/parent_family.jl")
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
  drawn once from `Bernoulli(p_bothcollege)`, constant in `t`. `Nk = 2` is exact, not a
  discretization. The child module's `k_grid` is a *different* object (the child's HC, `theta`).
  `p_bothcollege` is the target file's `bc_share_children_skill` = 0.2588 (the children the skill
  moments and `m_BC` are built on; Ali, 2026-10-02), required, no fallback (it was a hard-coded
  0.3). Since memo 19 the college moments split by BothCollege as the data define it (both
  parents with 16+ years).
- **Model/specification changes go through the advisor** before results built on them
  circulate. Numerical fixes (grid bounds, interpolation, solver settings) do not.

## Where things are decided (Ali, standing instructions)

- **Ask before deciding anything open.** Whenever unsure -- a parameter value, a moment, which
  run or file is current, how to word a rule -- ask Ali first; never write a guess into code or
  docs.
- **Never change a parameter value or a calibrated constant on your own**, and in a merge never
  resolve a conflict over a parameter value, or over anything Ali edited earlier, by a rule such
  as "the newer side wins": list each item with both values and ask.
- Specification changes (technology, moments, what is estimated) also go through the advisor
  before results built on them circulate.

## Two repositories

- **Child_Time_Study** (`/srv/project/speech/apps/Child_Time_Study`, GitHub
  `Ali-Iranpour/child-time-study`, branch `feat/stylized-facts-enhancements`) builds the data side
  in Stata: `Code/28_smm_moments.do` writes the moment vector, its joint covariance and the
  calibrated constants to `Output/Data/SMM/SMM_Moments.csv`, `SMM_VCov.csv`, `SMM_Constants.csv`
  (initial assets in `29_initial_assets.do`, the wage process in `30_wage_process.do`). **Stata runs
  on Ali's Mac** (the do-files carry Mac/Windows paths), so a server session cannot rerun it.
- **This repository** consumes them: copy the three CSVs into `Input/`, run
  `tools/make_smm_targets.py`, use the new `output/smm_runs/<stamp>_targets/targets.toml`.
- In Child_Time_Study, if a pull would touch locally modified files, **stop and ask**; never stash,
  reset, check out or force. Do not pull, fetch or reset **this** repository on the server without
  asking: the server copy can hold work that is not on GitHub.

## Handover

**`docs/HANDOVER.md`** is the single handover: current branch and worktrees, what is running,
decisions taken, open decisions, next actions. Read it first; update it at the end of each phase
(move superseded state into its History section).

## Long-running computation

- Run approved long computations only through a standalone driver script in a detached `tmux`
  session (`output/diagnostics/<date>_<tag>/run_*.sh`), which writes a `STATUS` file and a `DONE`
  or `FAILED` sentinel and advances between approved jobs on its own.
- After launch, verify once that it started and wrote its first status line; report the tmux
  session, the output and status paths and how to resume; then stop. **Do not poll.**
- If a gate fails or an unexpected error occurs, preserve the diagnostics and report. **Never
  weaken a check, loosen a tolerance, or launch a new experiment to get past a failure**; wait for
  Ali's decision. Ask before expanding an experiment budget.

## Shell pitfalls on this server

- The interactive shell is **zsh, which does not word-split `$VARS`**: `J="julia --project=."; $J x.jl`
  fails (exit 127). Put multi-word commands in a `sh` script.
- **`$T:path` is a zsh modifier**: write `git show "${T}:code/x.jl"`, with braces.
- **`$` inside a Julia docstring interpolates**: "150,000 USD", not "$150k".
- An `rm` on a variable path (`rm $C/*`) is blocked by a safety check; use `"${C:?}"/...` or a
  literal path.

## Before the estimation (status 2026-10-02, 20:00; details `docs/SMM.md` §7, runs in `docs/HANDOVER.md`)

Done:
- **The inputs are complete**: Child_Time_Study `a4448b2` (28 block C) supplies the composition
  tables, the p99 caps and the BothCollege shares; current targets
  `output/smm_runs/2026-10-02_193217_281837_targets/targets.toml` (all 12 composition frames).
- **Ali's decisions (2026-10-02)**: no p99 caps; BothCollege share 0.2588; the `sigma_1` start at
  our parents' mean schooling; the asset grids by the advisor's rule (60% of nodes below 150k USD,
  top 1M USD); **the search ranges stay as they are**, and the search draws Sobol' points until it
  has enough valid ones (`--sobol-valid N`).
- **The valid share at the production grids is 3.8%** (15 of 400;
  `output/diagnostics/2026-10-02_valid_share/`): 53% of draws break the elasticity rule (rejected
  before a solve), 39% send no child to college (skill collapses), 3% send every child. **The
  default start (`smm_start`) is invalid on the real inputs** (every child goes to college); the
  recovery point theta0 is valid (Q 14,274).

Still open:
1. sd(log AFQT) = 0.20 and lnw0 = 1.756 are **provisional and flagged**: replace them with measured
   values before results circulate (`docs/WAGE_RETURN_ANCHOR.md`).
2. The advisor's sign-off on memo 18/19, sigma_eta = 0 and the 2026-10-02 calibration; until then
   every run is a test.
3. The stale tests rewritten for memo 19 (listed below).
4. The production run's start and budget, from the 2026-10-02 two-arm pilot (with and without theta0).

## Correlations as moments

S5–S7 are pooled correlations (Letter-Word with itself 5 years later, each input with the LW
level, each input with the 5-year LW change). The model rebuilds each as the data pool it: a
mixture over the data's composition cells (ages, or base and end ages, and odd/even money
waves) with the data's counts as weights, including the between-age term; the LW test noise
enters only the variances, never the covariances. `docs/SMM.md` §3.1 has the formulas.

## The estimation (current state: `docs/SMM.md`)

**Memo 19: 20 parameters against 67 moments** (2 parent P, 59 skill S, 5 college T, 1 wealth W),
the DFVW technology of memo 18, the TikTak module ported from `apps/Structural-estimation-v2`
(2026-10-02). **Not yet estimated**: the inputs are complete since 2026-10-02 (above); a two-arm
pilot started that evening (`docs/HANDOVER.md`). `docs/SMM.md` is the one estimation
document (model, parameters, moments, search, validation, plan, caveats); `docs/SMM_MEMO19.md`,
`docs/SMM_COMPOSITION.md` and `docs/WAGE_RETURN_ANCHOR.md` hold the memo-19 records.

- **Targets come from Stata.** `tools/make_smm_targets.py` passes through
  `Input/SMM_Moments.csv`, `SMM_VCov.csv` and `SMM_Constants.csv` (copied from
  Child_Time_Study `Output/Data/SMM/`, built by `28_smm_moments.do`) into
  `output/smm_runs/<stamp>_targets/targets.toml`, with every constant under `[constants]`.
- **Calibrated values are read from the target file, never from code defaults**: the child's
  weight by age and at the half period, the age-1 skill draw, `L0`, `m_psychic`, and the
  parent's wage process, initial assets and BothCollege share (`parent_calibration(targets)`).
  The parent constructor has **no defaults** for these: a call without them fails by design.
- **`kappa_0` is on a centred scale**: the psychic cost is
  `kappa_0 + kappa_theta*(log theta - m_psychic) + kappa_ParEd*BC`, with `m_psychic` frozen in the
  target file; `check_psychic_centring` errors on a mismatch.
- **The objective weights by `1/se_j^2`**, the diagonal of Stata's joint covariance.
- **The child block is rebuilt on every evaluation** (`build_child_solution`, `moments.jl`); only
  stages that read no estimated child parameter are cached, as solution ARRAYS under a complete
  key. Keep the refresh bit-identical to a full re-solve.
- **The runner** (`code/smm/run_smm.jl`; flags in `code/smm/README.md`): the worker flag is
  `--procs`; an unknown flag is an error; fresh runs start the local stage with
  `--bootstrap immediate_mixed`; `--resume` continues from `tiktak_state.toml` and refuses any
  change of box, restart count or design; there is no `--legacy-import` (runs from before
  2026-10-02 warm-start a new run with `--init-from`). Any edit of `code/src/TikTak/*.jl` changes
  the optimizer identity: an in-progress run then resumes only with `--allow-optimizer-change`.
  **The runtime projection printed at the start times the FIRST evaluation, which includes
  compilation** (29-30 s against about 11.5 s warmed up, 2026-10-02): it overstates a run about 2.5x.
- **`SMM_TEST_FIXTURES=1`** runs the objective on labelled stand-ins (composition, wage loading)
  for TESTS only; refused with `--preset pilot|production`; the spec name gains `_TESTFIX`. A
  stand-in replaces only a MISSING input: on the current targets (composition present) with
  sd(log AFQT) set, the switch changes nothing and the tests run on the real inputs. A test that
  assumes the default start is valid (`test_penalties.jl`'s base point) then fails that check:
  rewrite it to start from a valid point, never weaken the check.
- **Worker budget**: about 20 worker processes on this shared server by default; when the server
  is free (check `uptime`: 112 cores), up to about 40 (Ali, 2026-10-02). Ask before going further.

## Watching a run

- The driver's folder `output/diagnostics/<date>_<tag>/`: `STATUS` (one line per step), `DONE` or
  `FAILED` at the end, and each job's console output (`*.console.log`).
- Each run's folder `output/smm_runs/<stamp>_<tag>/`: `run.log` (the progress lines),
  `run_record.toml` (every setting), `restarts.csv` (one row per finished local search),
  `tiktak_state.toml` (the checkpoint `--resume` reads), `estimates.toml` at the end.
- Follow a file with `tail -f <file>` in a terminal, or open it in VS Code (it reloads).
  `tmux attach -t <session>` shows only the driver script (detach with Ctrl-b, then d);
  `tmux ls` lists the sessions.

```bash
uv run --with pandas --with numpy --with pyreadstat python tools/make_smm_targets.py   # freeze targets
julia --project=. tools/test_tiktak.jl                                   # the optimizer, synthetic
SMM_TEST_FIXTURES=1 julia --project=. tools/test_smm_resume.jl <targets.toml>
SMM_TEST_FIXTURES=1 julia --project=. tools/test_runner_start.jl <targets.toml>
SMM_TEST_FIXTURES=1 julia --project=. tools/test_runner_geometry.jl <targets.toml>
SMM_TEST_FIXTURES=1 julia --project=. tools/test_tiktak_integration.jl <targets.toml>
SMM_TEST_FIXTURES=1 julia --project=. tools/test_reopt_integration.jl <targets.toml> <start.toml>
SMM_TEST_FIXTURES=1 julia --project=. tools/test_penalties.jl <targets.toml>
```

Not yet rewritten for memo 19 (they pin the old specification or miss the wage loading):
`code/smm/selftest.jl`, `tools/test_smm_tas.jl`, `tools/test_smm_own_study.jl`,
`tools/test_smm_baseline.jl`, `tools/test_hc_process_shock.jl` (check 7),
`tools/check_jacobian_rank.jl`, and the analysis tools `jacobian.jl`, `profile_param.jl`,
`sensitivity.jl`, `grid_sensitivity.jl` (`docs/SMM.md` §5).

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

`code/src/` model modules · `code/run_all.jl` end-to-end run · `code/smm/` estimation (`run_smm.jl` drives it, `moments.jl` is the economics; `reopt.jl` and the
tests in `tools/`) · `code/src/TikTak/` the optimizer module (`code/src/tiktak.jl` is its adapter) ·
`code/transfer_CRRA_wage.ipynb` exploration and counterfactuals ·
`docs/ERRORS.md` open and resolved findings, with the measurements behind them.

`Input/` holds only what the memo-19 code reads (source data and the generated `smm_*.csv`
plotting files the notebook uses); the pre-memo-19 TAS inputs were archived on 2026-10-02
(`temp/Input_archive_2026-10-02/`, and in git history). `Input/CODEBOOK.md` describes the
PSID/CDS parent block (Part A) and the TAS-linked child block (Part B): different units of
observation, never pooled. Parental time in the moments is ACTIVE time (`parent_Act`).
