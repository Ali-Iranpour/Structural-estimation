# HANDOVER — Structural-estimation (v1)

The single handover for this repository (rule in `CLAUDE.md`). The **current state** comes first;
when a phase ends, move the superseded state into **History** below (newest first) rather than
starting a new file. Paths are relative to the repository root
(`/srv/project/speech/apps/Structural-estimation`) unless absolute.

# Current state (2026-10-02, 16:45)

## Branches and working copies

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

## Running

- `tools/test_tiktak_integration.jl` on the merged code with `SMM_TEST_FIXTURES=1` and the
  enlarged smoke fixture (`--sobol 400 --sobol-valid 5`), started about 16:12; output in
  `temp/2026-10-02_merge_checks/runner_tests/tiktak_integration.log` (its `.exit` is written at
  the end). Uncommitted until it passes: the fixture change in `tools/test_tiktak_integration.jl`.

## Decisions taken (2026-10-02, Ali)

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

## Results and checks

- Optimizer synthetic 455/455; runtime projection 78/78; reopt identity 38/38.
- On memo-19 code with stand-ins: resume 39/39, runner start 48/48, geometry 14/14, penalties
  19/19, reopt integration 20/20; integration rerunning.
- Merged objective = the Mac branch to the last digit (with memo 19's own values passed in).
- Calibration on the built model: 25.5% at zero assets, median of the rest 3.13; Rouwenhorst SD
  0.36835, autocorrelation 0.97880; initial-node SD 0.321 against 0.306.
- Recovery test (another session, code `81884e3`, stand-ins): **not recovered** (`docs/SMM.md` §2.3).

## Open decisions and waiting items

See `CLAUDE.md` "Before the estimation can run" and `docs/SMM.md` §7: the composition tables
(Stata, on the Mac), sd(log AFQT), the p99 caps, the BothCollege share, parents' mean schooling,
the `sigma_j` boxes, the asset grid, the advisor's sign-off; then the stale tests, the valid
share at production grids, and the recovery rerun.

## Next actions

1. When the integration test passes: commit the fixture, update `docs/SMM.md` §5, fast-forward
   `fix/estimation-consistency` (no push).
2. Ali: run block C in Stata and merge `5aa297f` in Child_Time_Study; obtain sd(log AFQT).
3. Rewrite the stale tests and tools for memo 19 (`docs/SMM.md` §5).

# History

(none yet: this file was started on 2026-10-02)
