# HANDOVER — Structural-estimation (v1)

The single handover for this repository (rule in `CLAUDE.md`). The **current state** comes first;
when a phase ends, move the superseded state into **History** below (newest first) rather than
starting a new file. Paths are relative to the repository root
(`/srv/project/speech/apps/Structural-estimation`) unless absolute.

# Current state (2026-10-02, 20:05)

## Branches and working copies

- **Main checkout**: branch `fix/estimation-consistency` at `409d49e`. Not yet fast-forwarded.
- **`merge/port-memo19`**, worktree `temp/2026-10-02_memo19_merge/`: `9e2d760` (Ali: "commit
  first", 19:55) holds the BothCollege share from `[constants]` `bc_share_children_skill` with no
  fallback, `Input/SMM_Constants.csv` (Child_Time_Study `a4448b2`), the targets with the
  composition, the recovery test on the real inputs, the integration smoke fixture, the valid-share
  tool and measurement; the commit after it updates `CLAUDE.md` and this file. The main checkout's
  `fix/estimation-consistency` is fast-forwarded to it. **Nothing is pushed.**
- `temp/2026-10-02_pilot_code/`: detached at `9e2d760`, the pilot's frozen code.
- `temp/2026-10-02_integration_snapshot/`: detached at `5b5e361` + that patch, frozen while the
  test suite below runs (runner subprocesses load the code when they start).
- Child_Time_Study: pulled to `a4448b2` (block C exported: composition tables, p99 caps, BC shares).
- **Targets**: `output/smm_runs/2026-10-02_193217_281837_targets/targets.toml`: the 12 composition
  frames (check_composition passes), 327 `[constants]`; moments and covariance identical to
  `2026-10-02_152454_254208`.
- `temp/2026-10-02_mac_ref/` (the Mac branch, for the equivalence check) can be removed;
  `temp/2026-10-02_tiktak_port/` is the copy where the port was built.

## Running (each writes STATUS and DONE or FAILED; how to watch: `CLAUDE.md` "Watching a run")

- tmux `v1_pilot` (started 19:57, about 2.5 h; the runner's own projection of ~5 h counts the
  compile time of the first evaluation): the two-arm pilot (Ali: two arms, 20 workers each, the
  ~2.6 h size). Both arms: `--preset pilot`, draws until 150 valid (cap 8,000), 20 local searches of
  up to 600 evaluations all at once, no polish (the verdict will read NOT ACCEPTED: a learning run).
  Arm A supplies theta0 as one candidate (start Q 14,274.5), arm B uses the draws only. Driver and
  console logs `output/diagnostics/2026-10-02_pilot/`; runs
  `output/smm_runs/2026-10-02_195733_pilot_A_theta0/` and `..._pilot_B_random/`. A killed arm
  resumes with its full command plus `--resume <run folder>`.
- tmux `v1_recovery_real` (started 19:38): the recovery test (tech and all) on the real inputs,
  about 3 hours; `output/smm_runs/2026-10-02_193856_recovery_real/` (its code = `9e2d760`).
- Finished: the valid share (19:40, `output/diagnostics/2026-10-02_valid_share/`).
- Finished 20:03: the nine tests on the real inputs, in parallel, from the snapshot
  (`temp/2026-10-02_merge_checks/real_inputs_suite/`). Pass: optimizer synthetic, runtime projection
  78/78, reopt identity 38/38, reopt integration 20/20, resume 8/8. **Fail, one cause**: penalties
  18/19, runner_start 13/30, runner_geometry 3/13, integration 25/27 -- each starts from the default
  point with a handful of plain Sobol' draws (5 to 9), all invalid on the real inputs ("nothing to
  seed the local stage"; penalties: the base point is not finite). To fix in the test rewrite: a
  valid start (theta0) or `--sobol-valid`, no check weakened.

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

- Optimizer synthetic 455/455; runtime projection 78/78; reopt identity 38/38.
- With the stand-ins (code `2351150`): resume 39/39, runner start 48/48, geometry 14/14,
  penalties 19/19, reopt integration 20/20, integration 30/30 (clean rerun, 17:47).
- Merged objective = the Mac branch to the last digit (with memo 19's own values passed in).
- **On the real inputs the SMM start point is degenerate**: every child goes to college, the
  non-college group is empty, `kth_lw17_gap` is undefined and the start is penalised
  (Q = 1e12; `temp/2026-10-02_k3_checks/real_inputs/run.out`). TikTak still runs (it seeds the
  local stage only from valid Sobol' points); whether to warm-start elsewhere is open.
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
  Integration test with the new module: after the pilot (core budget).
- The pilot runs the OLD module from its frozen worktree, so its logs keep the old lines.

## Open decisions and waiting items

- The search's start on the real inputs (above).
- **Decided (Ali, 19:50): the search ranges stay as they are**; the search keeps drawing Sobol' points
  until it has enough valid ones (`--sobol-valid N`, with `--sobol` as the cap on attempts). The
  rejected draws cost nothing (infeasible) or about 11 s each (no college decision).
- The advisor's sign-off on memo 18/19, sigma_eta = 0 and the 2026-10-02 calibration (a test
  for now, flagged).
- The stale tests rewritten for memo 19 (`docs/SMM.md` §5).

## Next actions

1. Read the suite, the recovery and the pilot when they finish; report.
2. With Ali: the production run's start and budget from the pilot.
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
