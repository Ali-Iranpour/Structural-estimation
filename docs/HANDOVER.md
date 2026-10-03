# HANDOVER — Structural-estimation (v1)

The single handover for this repository (rule in `CLAUDE.md`). The **current state** comes first;
when a phase ends, move the superseded state into **History** below (newest first) rather than
starting a new file. Paths are relative to the repository root
(`/srv/project/speech/apps/Structural-estimation`) unless absolute.

# Current state (2026-10-03, morning)

## Branches and working copies

- **One working copy: `apps/Structural-estimation` itself**, branch `fix/estimation-consistency`, pushed to
  GitHub after each commit (Ali, 2026-10-03: work in the main folder). GitHub's `main` branch is left as it
  is (it is 74 commits behind and has one old commit, `dc9b80b`, whose feature exists here in a later form).
- **The 2026-10-02 worktrees were removed on 2026-10-03** (Ali): `temp/2026-10-02_memo19_merge` (branch
  `merge/port-memo19`, deleted: identical to `fix/estimation-consistency`) and the frozen run copies
  `temp/2026-10-02_run_k16_code` (`cf0a53a`), `_pilot_code` (`9e2d760`), `_tiktak23_check` (`63bd2ac`),
  `_integration_snapshot` (`5b5e361` + the patch committed as `9e2d760`) and `_mac_ref` (`c735165`). Every
  file they held is committed; the paths in run records and driver scripts that name them are history.
  Run outputs are in `output/smm_runs/` and `output/diagnostics/` here. Scratch kept in `temp/`:
  `2026-10-02_merge_checks`, `2026-10-02_k3_checks`, `2026-10-02_tiktak_port`, the Input and docs archives.
- In this folder two uncommitted deletions, `docs/REVIEW_BRIEF_14PARAM.md` and `docs/REVIEW_TRIAGE.md`, are
  not from these sessions: left alone.
- Child_Time_Study: pulled to `a4448b2` (block C exported: composition tables, p99 caps, BC shares).
- **Targets**: `output/smm_runs/2026-10-02_193217_281837_targets/targets.toml`: the 12 composition
  frames (check_composition passes), 327 `[constants]`; moments and covariance identical to
  `2026-10-02_152454_254208`.

## Running (each writes STATUS and DONE or FAILED; how to watch: `CLAUDE.md` "Watching a run")

- **FINISHED 2026-10-03 05:25 (exit 0, 6.7 h): the K = 16 run.** Q 6,784.8 -> **5,687.3** (-16%); 14 of 16
  restarts hit the 450 cap, 2 met FTOL; the polish hit its 300 cap (it supplied the point); restarts 2 and 4
  used the seed fallback (none lost). NOT ACCEPTED: no convergence evidence; `d_1` (99.9%) and `kappa_theta`
  (98.6%) on a bound, `phi_3`, `kappa_ParEd`, `sigma_eps` within 5%. Full report with the current code:
  `output/smm_runs/2026-10-02_223909_k16/report_current_code.txt`. Findings: `docs/ERRORS.md` F1 (skill
  dispersion collapses: SD ln k 0.00 from age 11), F2 (every child gets 88% of parental assets), F3 (input SDs
  60-87% too low, 58% of Q). The real-inputs recovery: both modes NOT RECOVERED (tech worst `d_2` 13.7%; all
  worst `kappa_ParEd` 28.6%, `sigma_eps` 18.6%).
- (was running) **tmux `v1_run_k16`: the K = 16 run, launched 22:39** (Ali: held at 22:29 to consider the authors'
  algorithm, released at 22:39 to run on the CURRENT algorithm with the previous settings). `--preset pilot
  --procs 36 --sobol 10000 --restarts 16 --local-evals 450 --polish-evals 300`, local width the automatic
  floor(sqrt(16)) = 4 (no `--local-procs`), started from pilot arm A's exact end point. Checked at start:
  start Q = 6784.794690649624, bit-identical to arm A's `Q_final` (same objective, exact point); the two
  timing evaluations agree; 16 restarts, 4 at a time, 4 rounds of about 79 min; projection 6.9 h (the
  draw stage is overstated: about half the draws are rejected without a solve), so expect about 05:00-05:30.
  Code was frozen in `temp/2026-10-02_run_k16_code` (`cf0a53a`, module 2.3.0-dev; removed 2026-10-03). Run folder
  `output/smm_runs/2026-10-02_223909_k16/`; driver `output/diagnostics/2026-10-02_run_k16/` (STATUS,
  run.console.log); timeout 9 h; a killed run resumes with its full command plus `--resume <run folder>`.
- tmux `v1_recovery_real` (started 19:38): the recovery test on the real inputs. **tech** (12 free)
  finished 21:29: Q 19,458.6 -> 3.64 in 1,330 evaluations, worst parameter error 13.7% of its box: NOT
  RECOVERED by the 2% criterion. **all** (20 free) still running (eval 2,000, Q 11.96 at 22:45).
  `output/smm_runs/2026-10-02_193856_recovery_real/` (its code = `9e2d760`).

## Finished today

- **The pilot** (19:57-22:31, both arms exit 0): arm A (theta0 supplied) Q_final **6,784.8**; arm B (draws
  only) **8,704.0**; both "NOT ACCEPTED" (no polish, by design). Flaws, measured: all 20 restarts ran at once
  (`--local-procs 20`: no restart learned from another -- hence Ali's standing floor(sqrt(K)) rule), and arm
  B lost 4 restarts to penalised mixed starts (fixed since in module 2.3.0-dev). Runs
  `output/smm_runs/2026-10-02_195733_pilot_A_theta0/`, `..._pilot_B_random/`.
- The valid share (19:40, `output/diagnostics/2026-10-02_valid_share/`): 3.8%.
- The real-inputs test suite (20:03, `temp/2026-10-02_merge_checks/real_inputs_suite/`): pass: optimizer
  synthetic, runtime projection 78/78, reopt identity 38/38, reopt integration 20/20, resume. **Fail, one
  cause**: penalties 18/19, runner_start 13/30, runner_geometry 3/13 -- they start from the default point
  with a handful of plain Sobol' draws, all invalid on the real inputs. To fix in the test rewrite: a valid
  start (theta0) or `--sobol-valid`, no check weakened. (The integration test's coarse job is fixed:
  `63bd2ac`; integration 30/30 at 21:56.)

## Deferred: the authors' TikTak (Ali, 2026-10-02 22:30: "we will come back to this later")

Ali decided to implement the authors' algorithm exactly, as a selectable algorithm beside the current
one, except K (set per run): all three local solvers with `bobyqa_h` the default (translated to Julia: no
native Julia equivalent exists), oracle-first against the compiled Fortran, the v1 fallback as an option
off by default, failed points as 67 equal gaps summing to 1e12, out-of-box points solved at the clamped
point plus the authors' bound penalty, DFPMIN polish with an evaluation cap per run. The differences, the
decisions and the licence notes are in **`docs/ERRORS.md` T1**. Nothing of it is built.

## Decisions taken (2026-10-02, Ali)

- TikTak: port v2's module and runner in full; immediate start-up; no `--legacy-import`; the
  objective hooks `child_extra` / `parent_extra` / `extra_moments`; `reopt.jl`, `test_penalties.jl`,
  `bench_tiktak.jl` ported.
- Memo-19 merge: memo 19 is the base; kept from 2026-09-27: parent y 0.1632, child y 0.144,
  net tuition 0.6, omega 0.2; taken from memo 19: mu at the half period 1 - 0.654, `sigma_eps`
  estimated, the 67 moments, mu_t from the data, sigma_eta = 0, the logistic TFP, the per-input
  elasticities with persistence estimated, renamed `sigma_j_k`.
- Wage process and initial assets from Child_Time_Study 29/30, read from `[constants]`, no
  fallbacks; Np stays 5; initial assets independent of BothCollege.
- BothCollege drawn with `bc_share_children_skill` = 0.2588 (was a hard-coded 0.3), required in
  `[constants]`. No p99 caps. Asset grids: 60% of nodes below 150k USD, top 1M USD (parent
  18/30, child 19/30). `sigma_1` start at our parents' mean schooling. sd(log AFQT) = 0.20 and
  lnw0 = 1.756 (fitted at the recovery point): **provisional, flagged**.
- Parental time is active time everywhere; the runner records memo 19's settings; the
  test-fixture switch kept; Input archived to `temp/Input_archive_2026-10-02/`;
  `ESTIMATION_MEMO.md` merged into `docs/SMM.md`.

## Results and checks

- Optimizer synthetic 479/479 (module 2.3.0-dev); runtime projection 78/78; reopt identity 38/38.
- With the stand-ins (code `2351150`): resume 39/39, runner start 48/48, geometry 14/14,
  penalties 19/19, reopt integration 20/20, integration 30/30 (clean rerun, 17:47).
- Merged objective = the Mac branch to the last digit (with memo 19's own values passed in).
- **On the real inputs the SMM start point is degenerate**: every child goes to college, the
  non-college group is empty, `kth_lw17_gap` is undefined and the start is penalised
  (Q = 1e12; `temp/2026-10-02_k3_checks/real_inputs/run.out`). TikTak still runs (it seeds the
  local stage only from valid Sobol' points). Runs since start from theta0 or the pilot's end point.
- **Valid share at the production grids, real inputs** (`output/diagnostics/2026-10-02_valid_share/`,
  400 Sobol' points, 3.1 min on 16 workers): **15 valid (3.8%)**. 213 (53%) break the elasticity rule
  (every s_jt < 1 at every age; rejected before a solve; the sigma_4 and sigma_1 boxes cut most);
  168 have no usable college decision: 155 send no child to college (median mean ln k at 18 about
  1.5 against the data's 6.69: the skill technology collapses), 13 send every child; 4 hit the
  "box bounds violated" assertion (a model failure by the ported rule). The valid points' Q runs
  from 1.04e5 to 2.8e6. A solved draw costs about 11 s.
- **The recovery theta0 on the real inputs** (production grids): valid, college 0.326 (BC0 0.349,
  BC1 0.260; data 0.343, 0.299, 0.797), mean ln k at 18 6.54, **Q = 14,274.5**, seven times lower
  than the best Sobol' point (`classify_nonfinite.csv` in the same folder).
- Recovery test with the stand-ins (code `81884e3`): **not recovered** (`docs/SMM.md` §2.3).

## TikTak and runner fixes (evening, Ali: "the sobol printing is wrong", "problems in the eval stage")

- `e12a399`: the Sobol-stage progress line with `--sobol-valid` counts valid draws against the target.
- `5b78ec9`, module 2.3.0-dev (v1 only; port to v2 later, Ali): a penalised mixed start falls back to
  the restart's own seed (arm B lost 4 of 20 restarts to this); a known start value is not recomputed
  (the authors' Fortran evaluates a start only inside its solver). Runner: penalised values print as
  "penalised", the incumbent is explained, `restarts.csv` has `start_fallback`, the projection times a
  second (warm) evaluation and warns if the two differ.
- Tests: synthetic 479/479 (the legacy guard now compares the path bit for bit and the count exactly:
  lower by one per restart, one more for restart 1 and one for the polish, measured on all six cases
  before the guard was changed); resume 39/39 on the real objective; projection 78/78; a serial smoke.
  Integration test with the new module on the real inputs: 30/30 (21:56).
- The pilot ran the OLD module from its frozen worktree, so its logs keep the old lines.

## Report changes (22:55-23:20, Ali)

- The estimates table shows each parameter's ACTUAL start (it printed the code's default even with
  `--init-from`), its box and its position in the box (search coordinates; flags within 2% and 5%, the
  acceptance scale); the near-bound check itself was already v2's, ported.
- A new untargeted section: the transfer at 18 by path (college / work), by BothCollege and by
  parental-asset tertile, with the data's support figures (`ksup_*`, `val_sup_*`, `wealth2122_*`) as context.
- `tools/report_point.jl <targets> <estimates.toml>`: the report at a saved point with the current code
  (the K = 16 run's own log will have the old report: run this on its estimates when it ends).
- Answers to Ali: `kterm_med22` (median net worth at first-child ages 21-22) is matched to the median of
  `a_term = a - tr` at the age-18 half period (no age adjustment, decision 2026-10-01); the T-block numbers
  are computed correctly -- their misfit is `docs/ERRORS.md` F1 (skill dispersion collapses); the transfers
  are F2.

## The notebook on the K = 16 estimate (2026-10-03, Ali)

- `code/transfer_CRRA_wage.ipynb` now builds its baseline as the SMM does (`build_child_solution`, the
  target file's calibration, the run's exact search vector) from `output/smm_runs/2026-10-02_223909_k16`;
  until now it used `CHILD_DEFAULTS` / `PARENT_DEFAULTS`, and its parent cell no longer ran under memo 19.
  A check cell asserts that its Q equals the run's `Q_final` (it does: 5,687.283294).
- Executed up to "## Counterfactuals on Parameters." only (cells 0-25; the counterfactual cells have no
  outputs and still use the defaults). Headless execution on the server: IJulia is not in v1's environment,
  so a scratch kernel ran v1's project with v2's IJulia on the load path and a stand-in for
  `IJulia.stdio_bytes` (v1's ProgressMeter expects it; v2's newer IJulia dropped it). Nothing in either
  project changed for this; VS Code's own Julia notebook kernel needs none of it.

## Open decisions and waiting items

- The authors' TikTak (`docs/ERRORS.md` T1): decided, deferred.
- **Decided (Ali, 19:50): the search ranges stay as they are**; the search keeps drawing Sobol' points
  until it has enough valid ones (`--sobol-valid N`, with `--sobol` as the cap on attempts). The
  rejected draws cost nothing (infeasible) or about 11 s each (no college decision).
- The advisor's sign-off on memo 18/19, sigma_eta = 0 and the 2026-10-02 calibration (a test
  for now, flagged).
- The stale tests rewritten for memo 19 (`docs/SMM.md` §5).

## Next actions

1. With Ali, then the advisor: F1/F3 (and F2) are specification questions; more search on this specification
   mostly refines a point that misses the same moments. Possible diagnostics for that meeting (not approved):
   a converged polish from the K = 16 point; a self-productivity profile (sigma_3 fixed higher, the rest
   re-optimised with `tools/reopt.jl --fix`); the diaries' reliability (data side).
2. With Ali: the authors' TikTak (`docs/ERRORS.md` T1), when he returns to it.
3. Rewrite the stale tests and tools for memo 19 (including `test_penalties.jl`'s base point, which
   assumes the default start is valid).

# History

## 2026-10-02, 16:45

### Branches and working copies

- **Main checkout**: branch `fix/estimation-consistency` at `409d49e` (the 2026-09-27 alignment,
  committed 2026-10-02 as the baseline). Not yet fast-forwarded.
- **`merge/port-memo19`**, worktree `temp/2026-10-02_memo19_merge/`: the TikTak port, the memo-19
  merge, and the work after it. To be fast-forwarded into `fix/estimation-consistency` once the
  last runner test passes (Ali's recommendation, approved). **Nothing is pushed.**
  Commits after the baseline: `0629733` TikTak port · `9535f91` memo-19 merge · `81884e3`
  `a_j` → `sigma_j` rename · `eb7714b`, `e2a2cd4`, `bbf0e59` (another session: dispersion check,
  composition pass-through, wage-anchor memo) · `6f5ea4e` `SMM_TEST_FIXTURES` · `12406d2` wage
  process and initial assets from `[constants]` · `d88532d` runner records · `a4a97c0` active
  parental time in the by-age files, docs · `19bd060` `reopt.jl` fix · `412b0ab` Input archive ·
  `373a3e4` `docs/SMM.md` merged and rewritten · `a612b73` and later: `CLAUDE.md`, this file.
- `temp/2026-10-02_mac_ref/`: a detached checkout of the Mac branch (`c735165`) used for the
  equivalence check; can be removed.
- `temp/2026-10-02_tiktak_port/`: the copy where the port was built and tested (own git).

### Running

- `tools/test_tiktak_integration.jl` on the merged code with `SMM_TEST_FIXTURES=1` and the
  enlarged smoke fixture (`--sobol 400 --sobol-valid 5`), started about 16:12; output in
  `temp/2026-10-02_merge_checks/runner_tests/tiktak_integration.log` (its `.exit` is written at
  the end). Uncommitted until it passes: the fixture change in `tools/test_tiktak_integration.jl`.

### Decisions taken (2026-10-02, Ali)

- TikTak: port v2's module and runner in full; immediate start-up; no `--legacy-import`; the
  objective hooks `child_extra` / `parent_extra` / `extra_moments`; `reopt.jl`, `test_penalties.jl`,
  `bench_tiktak.jl` ported.
- Memo-19 merge: memo 19 is the base; kept from 2026-09-27: parent y 0.1632, child y 0.144,
  net tuition 0.6, omega 0.2; taken from memo 19: mu at the half period 1 − 0.654, `sigma_eps`
  estimated, the 67 moments, mu_t from the data, sigma_eta = 0, the logistic TFP, the per-input
  elasticities with persistence estimated, renamed `sigma_j_k`.
- Wage process and initial assets from Child_Time_Study 29/30 (commit `5aa297f`), read from
  `[constants]`, no fallbacks; Np stays 5; initial assets independent of BothCollege.
- Child_Time_Study was **not** pulled (a local edit of `28_smm_moments.do` overlaps the incoming
  commit; a trial merge is clean): the CSVs were read from GitHub.
- Parental time is active time everywhere; the runner records memo 19's settings; the
  test-fixture switch kept and committed; Input archived to `temp/Input_archive_2026-10-02/`;
  `ESTIMATION_MEMO.md` merged into `docs/SMM.md` (old texts in `temp/docs_archive_2026-10-02/`).

### Results and checks

- Optimizer synthetic 455/455; runtime projection 78/78; reopt identity 38/38.
- On memo-19 code with stand-ins: resume 39/39, runner start 48/48, geometry 14/14, penalties
  19/19, reopt integration 20/20; integration rerunning.
- Merged objective = the Mac branch to the last digit (with memo 19's own values passed in).
- Calibration on the built model: 25.5% at zero assets, median of the rest 3.13; Rouwenhorst SD
  0.36835, autocorrelation 0.97880; initial-node SD 0.321 against 0.306.
- Recovery test (another session, code `81884e3`, stand-ins): **not recovered** (`docs/SMM.md` §2.3).

### Open decisions and waiting items

See `CLAUDE.md` "Before the estimation can run" and `docs/SMM.md` §7: the composition tables
(Stata, on the Mac), sd(log AFQT), the p99 caps, the BothCollege share, parents' mean schooling,
the `sigma_j` boxes, the asset grid, the advisor's sign-off; then the stale tests, the valid
share at production grids, and the recovery rerun.

### Next actions

1. When the integration test passes: commit the fixture, update `docs/SMM.md` §5, fast-forward
   `fix/estimation-consistency` (no push).
2. Ali: run block C in Stata and merge `5aa297f` in Child_Time_Study; obtain sd(log AFQT).
3. Rewrite the stale tests and tools for memo 19 (`docs/SMM.md` §5).
