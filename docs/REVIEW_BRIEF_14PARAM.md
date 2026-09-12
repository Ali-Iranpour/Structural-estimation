# Review brief: the 14-parameter / 17-moment SMM extension

You are reviewing one change to a Julia + Python structural-estimation repo. **Try to
falsify the claims below.** Read the code, not just the comments — the comments are
extensive and may assert things the code does not do.

## What the change does

A simulated-method-of-moments estimation was extended from **10 parameters / 10 moments**
(all parent-block) to **14 / 17**, by adding four *child*-block parameters — `kappa_0`,
`kappa_theta`, `kappa_ParEd`, `kappa_terminal` — and seven child-level moments built from
a second dataset (TAS): college completion overall, by ability tertile, by parental
education, plus parental net worth retained after a transfer.

The model is a parent stage (child ages 1–17) handing off to the child's own lifecycle
(ages 18–68). The child chooses college vs. work at 18; the parent chooses a transfer.

## Read in this order

| File | Why |
|---|---|
| `code/smm/moments.jl` | **the core.** Parameter routing, child-solution cache, `run_pipeline`, moments, objective, reporting |
| `code/src/child_lifecycle.jl` | model changes: `m_psychic` field, `reuse_grad` seam in `solve_model_college!`, retained `sim_college` / `sim_tr_init` |
| `tools/make_smm_targets.py` | builds the frozen targets + joint covariance from two Stata micro files |
| `code/smm/run_smm.jl` | runner: cache warm-up, checkpoint compatibility, run records |
| `tools/test_smm_tas.jl` | the validation suite (75 assertions) — check it tests what it claims |
| `docs/SMM.md` | the claims, stated in one place (the 14-parameter material was merged in on 2026-09-12) |

`sensitivity.jl`, `profile_param.jl`, `grid_sensitivity.jl`, `jacobian.jl`,
`standard_errors.jl` were refactored onto one shared evaluation path; skim for drift.

## Claims to attack

1. **Cache correctness.** `build_child_solution` caches two solve stages
   (`sol_*_work`, `sol_*_grad`) keyed on `child_config`, and re-solves the rest per
   evaluation. The claim is that *no* estimated parameter can reach the cached arrays.
   **Verify by reading `solve_model_work!` and `solve_model_college!`**: does anything in
   the high-school path or the graduate block (E=1, t > t_college) read `kappa_0`,
   `kappa_theta`, `kappa_ParEd`, `kappa_terminal` or `m_psychic`, directly or through a
   helper? Is `child_config` genuinely complete — could two materially different models
   collide on one key? Can a mutated object enter the cache?
2. **Parameter routing.** `split_params` routes by `hasproperty(PARENT_DEFAULTS, n)`.
   Could a child parameter be silently absorbed by a same-named parent constructor field,
   or dropped? Are the load-time guards actually reachable?
3. **The reparameterisation.** The psychic cost became
   `kappa_0 + kappa_theta*(log θ − m_psychic)`, `m_psychic` frozen in the target file,
   default `0.0`. Claimed behaviourally neutral. Is `CHILD_DEFAULTS.kappa_0 = 0.0587`
   consistent with the legacy `(0.2728, −0.0342)` at `m_psychic = 6.2634`? Does
   `check_psychic_centring` catch drift? Does the default of 0.0 leave every *other*
   call site (`run_all.jl`, the notebook, `selftest.jl`) unchanged?
4. **Ordering and isolation.** `run_pipeline` runs an initial "demonstration" child
   simulation, then erases `sim_college` / `sim_tr_init` before the parent block. Claim:
   no final moment can come from it. Is the erasure complete — is there any other array
   the demo simulation writes that a moment later reads?
5. **The wealth moment.** Target is `E[min(W, cut)]` on data; model side is
   `mean(min(sim_a[:,T+1] − transfer, cut))`. Same functional? Unit conversion
   (÷10,000) applied consistently to the moment *and* the covariance? Is it really
   post-transfer, not pre-transfer?
6. **Covariance and weighting.** `tools/make_smm_targets.py` reconstructs TAS influence
   functions and stacks them with the parent block's on a shared cluster key. Claim: it
   reproduces the published SEs in `Input/SMM_TAS_Moments.csv` to 6 dp. Check the ratio
   influence formula (`N` = whole-frame count, not subgroup), the degrees-of-freedom
   correction, that zero-contribution clusters become zero rows rather than dropped rows,
   and that `[moment_cov].names` order matches `SMM_MOMENTS`.
7. **Tertile construction.** `rank_tertiles` cuts simulated HC at child age 17 into
   model-internal thirds. Is the tie-breaking stable across evaluations? Does the
   age-17 column really correspond to the data's `ach_age == 17`? Off-by-one on the
   handoff column (`T` vs `T+1`)?
8. **Checkpoint rejection.** Old/incompatible resumes must be refused. Are the checks
   (`spec_version`, `source_sha`, `moment_names`, `m_psychic`) reachable, and does a
   *missing* field refuse rather than pass?

## Known and accepted — do not report these

- `kappa_ParEd` is targeted on an *either*-parent college group while the model's state is
  `BothCollege`. Deliberate, by user instruction; recorded as `docs/ERRORS.md` P7c.
- The terminal-wealth timing gap (data measured ~11 years after the model's transfer date)
  is uncorrected and documented.
- `kappa_0`'s box `[-2, 5]` is ~71% a flat-zero region, and `kappa_ParEd` saturates below
  −0.3. Both measured and flagged; boxes deliberately left at the approved values.
- `moment_scale` is retained but unused by the objective, to keep one frozen regression.
- No production estimation has been run; parameter *values* are not a finding.

## Reproduce

```bash
uv run --with pandas --with numpy python tools/make_smm_targets.py    # writes targets.toml
julia --project=. tools/test_smm_tas.jl <targets.toml>                # expect 75/75
julia --project=. tools/check_jacobian_rank.jl <targets.toml>         # rank/conditioning
julia --project=. code/smm/run_smm.jl --report-only --targets <targets.toml>
```

Report: correctness bugs first, with a concrete failing scenario (inputs → wrong output)
for each. Then anything where a comment overstates what the code guarantees.
