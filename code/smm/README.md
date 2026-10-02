# SMM — sixteen parameters, seventeen moments

> **Current specification (2026-10-02): memo 19 — 20 parameters against 67 moments.** See
> `docs/SMM_MEMO19.md` (moments, technology, decisions) and `docs/SMM_COMPOSITION.md`. The
> parent's wage process and initial assets are calibrated in Child_Time_Study (29/30) and read
> from `targets.toml` `[constants]`. The sections on running, cores, watching, acceptance and
> `reopt.jl` are current; **"What is being matched", "What is being estimated" and the text below
> this note describe the pre-memo-19 specification (2026-09-11 / 09-27) and are kept as history.**

**Since 11 September 2026 (preliminary, not through the advisor):** eleven parent
parameters — the ten below plus `sigma_eta`, the SD of an i.i.d. log shock in the HC
technology — and five child parameters — `kappa_0`, `kappa_theta`, `kappa_ParEd`,
`kappa_terminal`, plus `sigma_eps`, the scale of the college taste shock — against the
ten parent moments and seven TAS moments (`k0_complete`, `kth_ga17_gap`, `kpe_g0_c`,
`kpe_g1_c`, `kterm_x_strict_w99`, `kse_w_gap`, `sd_ga17`). `R_1` is fixed at 0. The
design, the moment definitions and the pilot budget are in `docs/SMM.md`; the diagnosis
that motivated it is `docs/ERRORS.md` P13.
**Baseline promoted 2026-09-12** from run `2026-09-11_182836_exp16b` (Q 61.80, accepted):
`PARENT_DEFAULTS` and `CHILD_DEFAULTS` (now in `child_lifecycle.jl`) carry it at full
precision, so a run with no `--init-from` starts from the fit. Boxes changed for the next
run: `kappa_0 [-3, 1]`, `kappa_ParEd [-1, 0.5]`, `sigma_4_1 [-0.05, 0.30]` — the reasons
are beside each line in `moments.jl` and in `docs/SMM.md`, "The next run".

The original pilot command (the incumbent is now the baseline, so `--init-from` is no
longer needed):

```bash
julia --project=. code/smm/run_smm.jl --sobol 1000 --restarts 5 --local-evals 500 \
      --skip-polish --grid 30 --procs 20 --seed 1234 --targets <targets.toml> \
      --init-from output/smm_runs/2026-09-10_183649/estimates.toml --outdir <dir>
```

The paragraphs below describe the parent block, which is unchanged.

Estimates **ten** parent parameters against **ten** moments. From 9 September 2026,
child investment is **own study only** (`study_hrs / 112`), at ages 6–9 and 10–17.
The HC moments are mean log HC at ages **3–9 and 12–17**. `sigma_4_1` is estimated
in `[-0.05, 0.30]` (top raised from 0.15 on 2026-09-12); `mu_1` remains fixed. Equal parameter/moment counts do not
establish identification: rerun the Jacobian after fitting the new specification.

The fixed school schedule uses `school_hrs`, the median within `(Year, Child_Age)`,
averaged over nonmissing rows at each child age and divided by 112. It is frozen
in each run's `targets.toml`. The child's leisure is
`1 - parental time - own study - fixed school`; its utility remains logarithmic.
Only own study enters the child-time term in HC production. The overlap caveat
for active-plus-nearby parental time remains.

`PARENT_DEFAULTS` remains the fitted **2026-09-09_003312** school-plus-study baseline.
It is a starting vector, not an estimate under the new specification. A plain
parent constructor preserves that legacy model with `school_time=zeros(T)`;
every SMM objective, fit report, profile and sensitivity diagnostic explicitly
loads the new school schedule. Old target files are rejected. Start a fresh run,
rather than resuming a nine-parameter checkpoint.

Generate the new targets, then use the normal runner:

```sh
python tools/make_smm_targets.py
julia --threads=1 --project=. code/smm/run_smm.jl --report-only --serial
# Use your usual run flags without --report-only to estimate.
```

The report-only command evaluates the starting vector; it does not estimate it.
Validate the new time budget and gradients with:

```sh
julia --threads=1 --project=. tools/test_smm_own_study.jl
julia --threads=1 --project=. code/smm/selftest.jl
```

The older run discussions below are historical; their fitted values and nine-
parameter identification results do not apply to the new own-study specification.

## Run it

Julia 1.11 is installed for this account under `juliaup`. If `julia --version`
says 1.12 (or `julia` is not found), put juliaup's bin directory first and pin
1.11 — the `Manifest.toml` was resolved against 1.11:

```bash
export PATH="$HOME/.juliaup/bin:$PATH"
```

Then, always with `+1.11`:

```bash
cd code/smm
julia +1.11 --project=../.. run_smm.jl --report-only    # ~1 min, no search
julia +1.11 --project=../.. run_smm.jl --quick          # ~15 min, smoke test
julia +1.11 --project=../.. run_smm.jl                  # the real run
```

**Start with `--report-only`.** It solves the model once at the current
calibration, prints how far each moment is from its target, and stops. That is
the fastest way to see where you stand, and it exercises every part of the
machinery except the optimizer.

For anything long, don't hold it in your terminal — the run writes its own log,
so detach it and read the file:

```bash
nohup julia +1.11 --project=../.. run_smm.jl > /dev/null 2>&1 &
tail -f ../../output/smm_runs/<timestamp>/run.log
```

### Where the output goes

Every run creates a fresh timestamped folder — nothing is ever overwritten:

```
output/smm_runs/2026-09-06_182800/
├── run.log          full console transcript, exactly as it appeared
├── run_record.toml  EVERYTHING the run was set with: git commit, targets by content
│                    hash, all nine boxes and links, seed, grids, every tolerance and
│                    budget. Written at STARTUP and rewritten at the end, so a killed
│                    or --report-only run still records what it was run with. Check
│                    `status`: "started" means it never reached the end.
├── restarts.csv     one row per restart as it finishes: theta, start and local value,
│                    whether it improved, and its NLopt return code
├── estimates.toml   estimated parameters, Q before/after, acceptance, budget
├── checkpoint.toml  rewritten atomically after every restart; feeds --resume
└── seeds.toml       the pre-testing survivors, written once (exact continuation)
```

**Throwaway runs stay under `output/smm_runs/`,** with `--temp [label]`:

```bash
julia +1.11 --project=../.. run_smm.jl --report-only --temp smoke
# -> output/smm_runs/2026-09-06_183505_temp_smoke/
```

Always timestamped, so a re-run never overwrites the previous one and the folder sorts
chronologically. Same contents as a real run — including `run_record.toml`, because a
smoke test whose settings you cannot reconstruct is not worth keeping either. New run
directories are gitignored; delete a throwaway folder whenever.

### Flags

| Flag | Meaning |
|---|---|
| `--report-only` | Print the fit at the current calibration and stop. No search. |
| `--quick` | Small grids and a tiny budget. Smoke test, **not** an estimate. |
| `--procs N` | Worker processes. Default 20 — see below. |
| `--sobol N` | Pre-testing points (default 1000). Cheap: these run in parallel. |
| `--restarts N` | Local searches (default 100). **Each one costs ~15 min.** |
| `--grid N` | Parent `Na = Nhc` for the *search* (default 30). `20` is 2.4× faster, and the winner is then re-optimised at the full grid. |
| `--refine N` | Evaluations for that full-grid re-optimisation (default 200). Only runs when `--grid` differs from the report grid. |
| `--local-evals N` | Cap per local search (default 2000). |
| `--polish-evals N` | Cap for the final polish (default 4000). |
| `--every N` | Seconds between progress lines (default 2). |
| `--temp [label]` | Throwaway run: writes to `output/smm_runs/<timestamp>_temp[_label]/`. Same contents, timestamped and gitignored. |
| `--targets PATH` | Select a saved target file; the runner freezes a copy beside the new run. A resume uses its own snapshot and refuses conflicting targets. |
| `--outdir P` | Write the run folder to `P` instead of `output/smm_runs/<stamp>/`. |
| `--serial` | Everything on one process. Slowest, but the easiest to debug. |
| `--resume DIR` | Continue a killed or paused run from `DIR/tiktak_state.toml` — see below. |
| `--init-from FILE` | Warm start from a previous run's `estimates.toml` or `checkpoint.toml`: **exact** (bit for bit) when the file has a `[search_vector]` whose parameter names and links equal this specification's; otherwise **by name** from `[parameters]` (8 decimals, so a ROUNDED start, recorded as such). Parameters the file lacks start at their SMM starts; a value outside its box is an error. Recorded per parameter in `run_record.toml`. |
| `--expect-start-q Q` | A gate in the run itself: after the timing evaluation at the start point the run stops before the search unless Q equals the given value exactly (copy it from the `start Q (full precision)` line). |
| `--skip-polish` | Bypass the final BOBYQA polish entirely (`polish_ret = SKIPPED`). This is a real switch, not `--polish-evals 0` — NLopt reads `maxeval = 0` as *no limit*, and the runner now refuses that. |
| `--seed N` | Common-random-number seed (default 1234). Part of the checkpoint identity. |
| `--preset P` | What the run is FOR: `smoke`, `integration`, `pilot` or `production` (2026-09-27). Supplies default budgets and gates, recorded with the run; explicit flags win. Without it the run is `custom` and every default is as above. See "Development tests and estimates" below. |
| `--sobol-valid V` | Continue the Sobol' sequence until **V valid** draws (a penalised draw does not count); `--sobol` is then the hard cap on attempts (default 10V). |
| `--reuse-pretest FILE` | Take another run's `pretest_cache.toml` values (same objective and design, verified) instead of re-evaluating them; reported as reused. |
| `--pretest-chunk N` | Pre-testing values between cache writes (default 4 × workers, at least 8). |
| `--local-mode M` | `async` (default with workers): restarts run asynchronously on the workers. `serial`: the published sequential algorithm on the master. |
| `--local-procs N` | Local workers (default 0 = automatic: min(workers, floor(√restarts))). |
| `--stop-after-restarts M` | **Pause** after M committed restarts, keeping `--restarts` as the schedule denominator; `--resume` continues at M+1. 0 = right after seed selection; = `--restarts` = before the polish. |
| `--max-retries N` | Re-dispatches of a restart whose worker process died (default 1). An error in the objective is never retried. |
| `--allow-fewer-restarts` / `--require-valid-target` | Override a preset's gates: run fewer restarts when the valid pool is small / make a missed `--sobol-valid` a failure. |
| `--allow-optimizer-change` | Let `--resume` continue although a SOLVER setting (`--local-evals`, `--polish-evals`, `--skip-polish` …) or the optimizer SOFTWARE (`code/src/TikTak/*.jl`, NLopt) changed: recorded as an event and a new settings epoch (committed restarts and replayed jobs keep the settings they ran with), and the run is labelled `changed_optimizer`. A change of the search PLAN — box, `--restarts`, pre-testing design, supplied start, schedule — is refused even so (2026-09-28). |
| `--print-identity` | Print the objective identity a checkpoint records, and exit (used by the tests). |
| `--bootstrap B` | How the asynchronous local stage starts: `immediate_mixed` (the **default for a fresh run**, since 2026-10-01 in v2 -- every worker starts a restart at once, each mixed with the best point so far) or `first_alone` (restart 1 alone, then the pool). Part of the search plan: a resume keeps the policy its checkpoint was written with. |
| `--local-ftol F` | Nelder-Mead's `ftol_rel` (default 1e-3): a local search stops when 2|f_worst - f_best| / (|f_worst| + |f_best|) < F across the simplex. The reference's amoeba uses 1e-4. A solver setting (resume needs `--allow-optimizer-change`). |
| `--runtime-evals N1,N2,...` | Evaluations per restart observed in a COMPATIBLE earlier run, for the empirical runtime scenario the run prints. Display only. |
| `--normalize`, `--local-init-step F`, `--local-step-schedule fixed\|theta_shrink`, `--local-step-min F`, `--polish-init-step F` | Search geometry: box-normalized coordinates; Nelder-Mead's initial step as a fraction of each box width (0 = NLopt's default); theta_shrink makes restart j's step F·max(min, 1 - θ_j); BOBYQA's initial trust radius. Defaults are the baseline. Optimizer settings (part of the checkpoint's identity). |
| `--parent-extra k=v,...`, `--child-extra k=v,...` | NON-ESTIMATED constructor settings of the run (grid sizes, `a_max`, `Nap`, numerical options, a fixed `omega`), passed to every evaluation, the report and the refinement; recorded and verified on `--resume` (`grid_extra`). An estimated parameter here is an error. Note: the asset grids' focus share and curvature are not constructor keywords in this model. |

**Every flag is validated.** An unknown flag (`--workers`, say) is an error before
anything is written; it used to be silently ignored.

### Resuming a killed or paused run

Since 2026-09-27 the authoritative checkpoint is **`tiktak_state.toml`**, written by the
TikTak module (`../src/TikTak/`) right after seed selection, after every committed restart,
at a pause and after the polish. It is versioned (schema 2 since 2026-09-28; a schema-1 file is
migrated on reading and says so in its events), checksummed (a sha256 trailer)
and written atomically, with the previous generation kept as `tiktak_state.toml.prev`: a
truncated or edited file fails its checksum and the previous generation is used, and the
log says so. It holds the seeds and the schedule denominator K, the incumbent **with its
origin and any verification**, every committed restart (start, endpoint, evaluations,
return code, worker), the restarts in flight, and the lifetime evaluation counts.
`pretest_cache.toml` holds every pre-testing value, written chunk by chunk, so a run killed
during pre-testing loses at most one chunk. `checkpoint.toml`, `seeds.toml` and
`restarts.csv` are derived from the state after every write (`--init-from` still reads
`checkpoint.toml`).

```bash
cd code/smm && julia --project=../.. run_smm.jl <the same flags> --resume ../../output/smm_runs/2026-09-06_014210
```

Before anything is solved, the resume is refused unless the OBJECTIVE is the same (every
field of the identity: targets by content, model source, parameter names/boxes/links,
moments, centring, grids, simN, seed, overrides) and the OPTIMIZER is the same (every
TikTak setting, the requested `--restarts`, the Sobol design, the software). Each refusal
names what changed. A different `--restarts` is refused: it changes the mixing schedule, so
it is a new search (warm-start it with `--init-from DIR/checkpoint.toml`).

What a resume promises (2026-09-28): the saved state is checked for consistency first — a
checkpoint that claims completion with a restart missing or in flight, commits a restart
twice, or holds an incumbent outside its box is refused, never continued or reported.
Restarts that were IN FLIGHT are replayed from their recorded starts, theta and settings in
EITHER local mode (until 2026-09-28 a serial resume skipped them). The record's
`resume_semantics` describes the whole history: `serial_exact` only when every segment ran
serially or on one local worker and nothing was replayed — then the answer is the
uninterrupted one (tested after seed selection, after restart 2 and before the polish);
`async_continuation` when a segment used several workers or a job was replayed (completion
order can change later starts); `changed_optimizer` after an allowed solver or software change.
The acceptance gate `search_budget_complete` is validated from the state —
every planned restart committed once, none in flight — not read from the status symbol.
NLopt's internal simplex is never serialised: an interrupted local search restarts from its
start.

**Workers** (2026-09-28): the runner starts in a fresh process and owns its pool; each worker
gets `--threads=1` explicitly (workers inherit a `JULIA_NUM_THREADS` from the environment, not
the master's `--threads`), and the log and `run_record.toml` record the Julia and BLAS threads
every process actually runs with. A worker still solving when an aborted stage's drain deadline
passes is removed (the pool is the runner's own); an exception the worker itself delivered —
an `EOFError` from the objective included — is an objective error, never a lost worker.

**Pausing on purpose.** `--stop-after-restarts M` pauses after M restarts; `touch
<run dir>/PAUSE` pauses an asynchronous run gracefully (no new restart starts, the running
ones are committed). Either way the polish is not run and `--resume` continues.

**Runs written before the TikTak port (2026-10-02)** -- every earlier run of this repository -- have
`checkpoint.toml` + `seeds.toml` and no `tiktak_state.toml`. `--resume` refuses them, and there is no
`--legacy-import` here (Ali, 2026-10-02; v2 has one). Warm-start a new run from one with `--init-from
<its dir>/checkpoint.toml` instead.

## How many cores does it use?

**20 worker processes, out of 112 on the machine (18%).** `haflinger` is a
shared server — the cap is a house rule in `run_smm.jl` (`WORKER_BUDGET = 20`),
not a hardware limit, and it binds long before RAM or cores would:

```
workers = min( 20 ,  CPU_THREADS - 1 ,  RAM_GB / 2 )
            ^^        ^^^^^^^^^^^^^^^    ^^^^^^^^^^
        house rule     111 here          251 here
```

The run prints which of the three bound. Override with `--procs N`, but on a
machine other people are using, raise it by agreement, not because the cores
look idle. Each worker also gets **one BLAS thread** — without that, 20 processes
each open a BLAS pool sized to all 112 cores and the machine thrashes.

**Threads are deliberately not used.** NLopt.jl is not thread-safe in this
project — with threads the objective killed the process with exit 0 and no error
message. Worker *processes* each own their NLopt state, so the hazard cannot
arise. `Threads.@threads` must not be reintroduced here.

### Both stages run on the workers (since 2026-09-27)

| Stage | Parallel? | How |
|---|---|---|
| Sobol pre-testing | **yes** | Every worker gets the next candidate as soon as it is free (no batch barrier); values cached chunk by chunk. |
| Local restarts | **yes, asynchronously** (`--local-mode async`) | By default (`--bootstrap immediate_mixed`) every local worker starts a restart at once; each idle worker then immediately gets the next restart, mixed with the best point *committed* so far — restarts still running are not waited for. With `--bootstrap first_alone` restart 1 runs alone first. |

The asynchronous local stage is TikTak's asynchronous variant (the reference repository's
way to scale), **not** the published sequential algorithm: restart *j* may be mixed with an
incumbent that does not yet include restarts still running, so the search path depends on
completion order. With one local worker it is exactly the sequential algorithm, and
`--local-mode serial` keeps the sequential algorithm on the master as the reproducible
reference.

**How many local workers.** The default is min(workers, floor(√restarts)): about 2 for 5
restarts, 10 for 100, the full 20 for 1,000. That follows the reference's empirical
suggestion of roughly √(restarts) workers; it is a conservative default, not a mathematical
limit (`--local-procs` overrides it). Too many workers for few restarts turns TikTak into
plain multistart, because the mixing then sees little of the other searches.

The run prints both stages before the search begins:

```
projected runtime
  sobol stage     1001 evals / 20 workers  =   33.4 min   (parallel)
  local stage   165000 evals / 20 workers  = ...          (asynchronous; 20 at a time from the start)
```

The local projection is optimistic: restarts of unequal length leave workers idle at the end.

## Making the local stage faster

Besides the asynchronous workers above, the levers are *cheaper evaluations* and
*fewer of them*. Measured on this machine, in order of value:

**1. Search on a coarser grid — 2.5× (`--grid 20`).** 98% of an evaluation is
`solve_model!`, and its cost scales with the parent's `Na × Nhc`:

| parent grid | solve | simulate | `mean_c_p` | `mean_l_p` | `mean_e_p` |
|---|---|---|---|---|---|
| `Na=Nhc=30` | 11.62 s | 0.24 s | 3.0153 | 0.4703 | 2.1884 |
| `Na=Nhc=20` | 4.78 s | 0.03 s | 3.0149 | 0.4701 | 2.1924 |
| `Na=Nhc=15` | 2.71 s | 0.01 s | 3.0279 | 0.4693 | 2.2179 |

At 20 the targeted moments move by **0.01%, 0.04% and 0.2%** — against the 3%,
11% and 461% gaps the estimation exists to close. The optimizer does not need
resolution the answer needs, so `--grid 20` searches cheap while the reported fit
is still re-solved at 30. Below 20 the drift starts to show (`e_p` 1.3% at 15).

**2. Fewer restarts — linear (`--restarts 5`).** Each restart costs ~15 min at
full grid. In testing, restart 1
alone drove `Q` from 0.081 to 1e-5. Ten restarts is insurance against local
minima that this objective has not shown any sign of. Pair a cut here with a
larger `--sobol`, which is free and buys back the same insurance.

**3. Do not touch `simN`.** It is 2% of the cost, and cutting it to 500 moved
`c_p` *more* (3.0275) than halving the grid did — that is simulation noise, which
is the one thing common random numbers exist to keep out of the objective.

Combining 1 and 2, `--grid 20 --restarts 5 --sobol 400` is roughly **30 minutes**
instead of 2.6 hours, and the reported fit is still at full resolution.

A note on the old `batch` option (removed 2026-09-27): it ran SYNCHRONOUS batches of
restarts on threads, all reading one frozen incumbent and waiting for the slowest member, and
its measured cost on Rastrigin was severe (`f` 2.985 sequential → 6.965 at batch 4). The
asynchronous process-parallel stage that replaced it is a different scheme — every restart
is mixed with the latest committed incumbent — and at √(restarts) workers it is the
reference's way to scale. Measured on a synthetic CPU-bound objective (16 restarts, 2 ms per
evaluation; `apps/Structural-estimation-v2/output/diagnostics/2026-09-27_tiktak_fix/bench/`): the local stage took 3.18 s
serially, 1.88 s on 2 workers and 1.03 s on 4, with final values 2.9859 / 2.9863 / 2.9860.
One synthetic function is not evidence about the SMM objective's quality/time trade-off; a
pilot at 1, 2, 4 and 8 workers on the real objective is the way to settle it.

### A stopping-rule bug this uncovered — fixed 2026-08-27

`tiktak.jl`'s local searches originally stopped on `ftol_rel` and `maxeval` only.
`ftol_rel` tests `|Δf| ≤ ftol_rel · |f|`, so **as `f → 0` the threshold collapses
with it and the test can never be satisfied.** A just-identified SMM drives `Q` to
~0 by construction, so every restart ran to `maxeval = 2000` regardless of having
converged. Measured: restart 1 reached `Q ≈ 0` at evaluation 61 and was still
running at 290.

At 15.5 s an evaluation that is **8 hours per restart instead of 15 minutes** —
the default run would have taken days, not hours. `local_ftol_abs` and
`local_xtol_rel` now stop it. A correction (2026-09-27 review): neither is scale-free.
`ftol_abs` is in units of Q (a sum of inverse-variance-weighted squared errors), and
`xtol_rel` is relative to |x|; NLopt 2.10's relative test also accepts two equal values,
including 0. They are backstops set far below `ftol_rel`, which still stops a search with a
non-zero optimum first. The self-test still passes (sphere reaches 7.7e-45).

## Watching a run

Progress prints every 2 seconds (`--every N` to change), to both the console and
`run.log`:

```
  sobol      140/201  70%   best Q    0.38734   0.9 min elapsed, ~0.4 min left
  sobol    complete: 201 evaluations, best Q 0.31552, 1.8 min
  restart   1/10   eval   240   this Q    0.04120   best Q    0.03310   6.2 min
  restart   1/10  DONE   this    0.03310   best Q    0.03310   8.1 min, ~73 min left
```

`best Q` should fall and then flatten. If it is still dropping at the last
restart, the budget was too small — raise `--sobol` first.

With workers, the Sobol lines are printed as each value reaches the master. The local stage
prints ONE LINE PER FINISHED EVALUATION of each running restart, and nothing while an
evaluation is still running (since 2026-09-29; before, the running restarts' state was
reprinted every `--every` seconds, so a 35 s evaluation gave ~15 identical lines). The polish
on a worker prints NOTHING until it ends (the module dispatches it without progress
telemetry): at production grids, 200 polish evaluations are ~2 h of a silent `run.log`
before the `Finished` banner.

```
  restart   1/21   eval    17   this Q     917.2345   best in this search     916.3000   incumbent     916.3000    47.3 min   [1 running]
  restart   1/21  DONE  FTOL_REACHED     start Q     916.3000   end Q     915.8120   incumbent     915.8120   512.0 min
```

`this Q` is the value just evaluated, `best in this search` the restart's best so far,
`[n running]` how many restarts are in progress (1 while restart 1 runs alone, then up to
`--local-procs`). The `DONE` line says why the search stopped: `FTOL_REACHED` /
`XTOL_REACHED` (the tolerance test) or `MAXEVAL_REACHED` (`--local-evals` ran out); its
worker then takes the next restart. Q is printed with four decimals so a small improvement
is visible. The telemetry is coalesced — at most one message per job per 0.2 s, only the
latest kept — and never blocks a worker. (Until 2026-09-27 workers pushed every value into a
bounded `RemoteChannel` that only the Sobol stage drained; a parallel local stage would have
filled it and stalled the workers.)

## What is being matched

> **Historical (pre-memo-19).** This table is the 2026-09-11 moment set; its `t_p` rows used
> `par_time_tot` (active + nearby) until 2026-09-27. Memo 19's parental-time moments (S6–S8) use
> ACTIVE time, `taup = parent_Act / 112` (Child_Time_Study `28_smm_moments.do`), matched to the
> model's `sim_t`. Current moments: `docs/SMM_MEMO19.md`.

| Moment | Data source | Data mean | N |
|---|---|---|---|
| mean consumption | `cons_exhous_real_w99` | 3.158 (= $31,577/yr) | 6,742 |
| mean **work** hours | `(wh_mom+wh_dad)/2 / 112` | 0.307 (= 34.4 hrs/wk) | 15,665 |
| mean **child time**, ages 1–9 | `par_time_tot / 112` | 0.4672 (= 52.3 hrs/wk) | 475 |
| mean **child time**, ages 10–17 | `par_time_tot / 112` | 0.3232 (= 36.2 hrs/wk) | 590 |
| mean investment, ages 1–9 | `m_method2_final_w99` | 0.353 (= $3,532/yr) | 8,178 |
| mean investment, ages 10–17 | `m_method2_final_w99` | 0.441 (= $4,414/yr) | 7,182 |

**Leisure `l_p` is no longer targeted.** `l_p = 1 - h_p - t_p` identically, so
targeting leisure pins the *sum* of work and child time and says nothing about the
split — and the split is where the model was wrong. The 2026-08-27 estimate
matched leisure *exactly* while working 29.6 hrs/wk against 34.4 in data and doing
23.2 hrs of childcare against 18.2: two errors that cancel inside `l_p` and are
invisible to it. `l_p` is still printed, as the residual check that the time budget
closes.

**Which active-time variable.** `Mom_Total_Act` / `Dad_Total_Act`, **not**
`par_time_act` or `parent_Act`. Only the first pair closes the identity
`leisure = 112 - work - active` per parent — verified on the data:

| candidate | mom | dad | want |
|---|---|---|---|
| `Mom_Total_Act` / `Dad_Total_Act` | **112.00** | **112.00** | 112.00 ✓ |
| `par_time_act` | 117.28 | 124.60 | ✗ |
| `parent_Act` | 117.28 | 124.60 | ✗ |

`par_time_act` and `parent_Act` are identical household-level "any parent"
measures; either would break the time budget.

**Why `t_p` is split but `h_p` is not.** `h_p` is flat in child age (0.3062 early
vs 0.3080 late) — one pooled mean, on 15,665 observations. `t_p` **halves** over
the family stage (30.3 → 9.7 hrs/wk, late/early **0.512×**) and does so
*monotonically*, which is exactly the shape `exp(sigma_1_0 + sigma_1_1(t−1))` can
produce. That makes `sigma_1_1` better identified than its `sigma_2_1`
counterpart, whose investment profile is U-shaped and cannot be matched by a
monotone form.

Targets are generated into `output/smm_runs/<timestamp>_targets/targets.toml` by
`tools/make_smm_targets.py`. A new estimation selects the newest timestamped target
snapshot by default, or the file passed with `--targets PATH`, and copies it into its
own directory before evaluation. Resume always uses that saved copy. Jacobian and
sensitivity tools accept `--targets`; with `--at`, they default to `targets.toml`
beside the selected estimates. Standard errors use the targets recorded by the Jacobian.
`Input/` holds only Stata/CSV source data and the codebook. Regenerate with:

```bash
python3 tools/make_smm_targets.py
```

### Units

**One model unit = $10,000/year.** Confirmed three ways: `ASSET_RESCALE = 10`;
the model's mean after-tax household income is 5.23 model units = $52,264, a
plausible US figure; and the older targets in `docs/SMM.md` use the same scale.

**Time is a share of the 112-hour non-sleep week, per parent.** The model splits
`l_p + h_p + t_p = 1`; the data builds leisure as `112 − own work − own active
childcare`, where 112 = 168 less a 56-hour sleep allowance. Same identity —
verified on the data, `mean(leisure + work + active) = 112.00` exactly.

**Per parent, not per household.** `wage_func` multiplies by 2, so one modelled
adult stands for two earners sharing one time allocation. The data counterpart is
the *average* of mother and father. Using `leis_hh` would double the target.

## What is being estimated

> **Historical (pre-memo-19).** Memo 19 estimates 20 parameters: the age-varying elasticities
> `sigma_j_0`, `sigma_j_1` (j = 1..4, persistence included), the logistic TFP `d_0..d_3`, `phi_2`,
> `phi_3`, `lambda_2` and the five child parameters; `sigma_eta` = 0 and `R_0`/`R_1`, `mu_0`/`mu_1`
> no longer exist. See `docs/SMM_MEMO19.md`.

**Fourteen parameters against seventeen moments — over-identified by three (2026-09-10).**
Ten parent parameters against ten parent moments, plus four child parameters
(`kappa_0`, `kappa_theta`, `kappa_ParEd`, `kappa_terminal`) against seven TAS moments.
Because the system is over-identified the WEIGHTING MATRIX now decides the answer and not
merely the path to it: the objective weights each residual by `1/se_j^2` from the joint
cluster-robust covariance. `report_fit` prints each moment's share of `Q`; read it before
trusting a fit, because inverse-variance weighting concentrates on whichever moment is
most precisely measured when the model cannot fit any of them to within sampling error.

The paragraph below describes the historical nine-against-ten design and its argument is
unchanged.

**Nine parameters against ten moments — over-identified by one.** `Q` cannot reach
zero and the weighting matrix is *not* irrelevant at the optimum, so equal weights are a
real assumption. Counting nine against ten establishes nothing about identification on its
own; what does is the residual Jacobian — **and that is now a saved artefact, not a
recollection**. Run `jacobian.jl` and read `output/identification/<dir>/jacobian.toml`.

Measured 2026-09-06 at the incumbent, grid 30, central differences at 0.5/1/2% of each
box width, columns scaled to a full-box move:

| columns | condition number | smallest singular value | thin-SVD rank |
|---|---:|---:|---|
| 9 (current) | 51.2 | 0.266 | 9 of 9 |
| 10, adding `sigma_4_1` | 228.9 | 0.060 | 10 of 10 |
| 11, adding `sigma_4_1` and `mu_1` | 183.4 | 0.089 | 10 of **11** — one direction is unidentified by construction |

`sigma_min` is stable across the three steps (spread 7% of its level), so it is the model's
and not the finite difference's.

**The pairwise cosines reproduce; the condition-number ratio does not.** Cosines are
invariant to column scaling and come out as reported — `sigma_4_0`/`mu_1` **0.991**,
`sigma_4_0`/`sigma_4_1` **0.814**, `sigma_4_1`/`mu_1` **0.807**. Condition numbers are
*not* scale-invariant, and the previously circulated 49.2 → **1067.1** does not reproduce:
under a stated box for `sigma_4_1` of [−0.05, 0.05] the same comparison is 51.2 → **228.9**,
a 4.5× degradation rather than 21.7×. The direction of the conclusion survives; the
magnitude was never reproducible because the box it depended on was never recorded.

**A finding that changes the emphasis.** Among the nine parameters actually estimated, the
worst-separated pair is **`sigma_1_0` vs `sigma_1_1` at 0.908** — *higher* than the
`sigma_4_0`/`sigma_4_1` 0.814 that is the stated reason for leaving `sigma_4_1` out. Both
`t_p` and `i_c` are split at the same two age groups, so if 0.814 disqualifies a slope
parameter, 0.908 is a problem for one already in the set. Take this as an argument for
richer age moments, not for dropping `sigma_1_1`.

**Estimate correlations are worse than the cosines suggest.** From the sandwich
(`standard_errors.jl`): `phi_3`/`R_0` **−0.998**, `R_0`/`sigma_4_0` **+0.987**,
`R_0`/`sigma_1_0` **+0.986**. Valuation against technology is the binding problem, and it
is not visible in a pairwise column cosine.

**Two scales, deliberately.** Level moments are scaled by their own target so every
residual is a proportional error. The two HC moments are means of *logs*, where the
residual is already proportional, so they are scaled by 1 — dividing them by a log
W-score of ~6.1 shrank them 6.1× and made a 60% error in the level of human capital
score like a 7.7% miss. See `moment_scale` in `moments.jl`.

**Ages are matched on both sides.** The data is weighted equally per child age, as
the simulation is, and the HC moments start at child age 3 because the composite is
not administered earlier. See `SMM_AGE_HC_LO` and `AGE_HC_LO`.

| Parameter | Moves | Bounds | Link | Incumbent |
|---|---|---|---|---|
| `phi_2` | leisure weight → `h_p` | [0.01, 20.0] | log | 0.14372194 |
| `phi_3` | parents' weight on child skill → `t_p`, `e_p` | [0.05, 20.0] | log | 1.02014337 |
| `lambda_2` | child's weight on skill → `i_c` | [0.05, 20.0] | log | 8.67925673 |
| `R_0` | HC technology TFP → the **level** of log HC | [0.5, 100.0] | log | 50.60319558 |
| `sigma_1_0` | **level** of HC elasticity to parent *time* → early `t_p` | [−4.0, −0.1] | level | -0.22324257 |
| `sigma_1_1` | **age slope** of that elasticity → late `t_p` | [−0.20, 0.05] | level | -0.14134183 |
| `sigma_2_0` | **level** of HC elasticity to *money* → early `e_p` | [−5.0, −0.5] | level | -3.75506167 |
| `sigma_2_1` | **age slope** of that elasticity → late `e_p` | [−0.10, 0.05] | level | -0.04998108 |
| `sigma_4_0` | HC elasticity to the child's *school plus study* → `i_c` | [−6.0, −1.0] | level | -5.98521880 |

`phi_1` and `lambda_1` are **normalised to 1** — utility is defined only up to relative
weights, so two of the five must be pinned. `sigma_4_1 = 0.02` and `mu_1 = −0.04` are held
fixed; see *Identification* below for why, and for the qualification that goes with it.

`sigma_1_0` is the right partner for `t_p` on the model's own evidence:
`parent_family.jl` records that `tau_p` sits at 0.011–0.023 for *every* `phi_2`
from 0.05 to 3.0, because the FOC scales with `phi_2` on both sides — "tau_p is
set by sigma_1 and the value of the child's HC". So `phi_2_0` identifies work and
`sigma_1_0` identifies child time, through separate channels.

### β is calibrated, not estimated

`beta_0 = 0.98` (was 0.97), set by instruction — **not** estimated, because
consumption enters as a single pooled mean and nothing in the objective identifies
patience. Consumption was flat because `β(1+r) = 0.97×1.03 = 0.9991`, i.e. the
Euler condition was almost exactly balanced:

| β | β(1+r) | growth/yr | over 16 yrs |
|---|---|---|---|
| 0.97 | 0.9991 | −0.06% | −1% |
| **0.98** | 1.0094 | +0.63% | **+10.5%** |
| 0.99 | 1.0197 | +1.31% | +23.1% |
| — | — | — | *data: +21.8%* |

So 0.98 recovers about half the observed tilt; ~0.989 would match it. To target the
profile properly, split `c_p` early/late and let `beta_0` be estimated against it —
the same trick used for `sigma_1_1` and `sigma_2_1`. **Note this changes the
baseline for the notebook and counterfactuals too, not only the estimation.**

Bounded parameters are searched on a linked (log) scale so a step can never
produce a negative weight.

### Why investment is split by age

`sigma_2_t = exp(sigma_2_0 + sigma_2_1·(t−1))`, so `sigma_2_1` is an **age slope
that compounds from age 1** — there is no kink at 9, and the split is a property
of the *moments*, not the model. A single pooled mean of `e_p` cannot separate a
slope from a level: many `(sigma_2_0, sigma_2_1)` pairs reproduce the same
average, so adding `sigma_2_1` to a 3-moment design would be **under-identified**
— it would return an answer determined by the Sobol seed, not the data. The
second investment moment is what pins the slope down.

### Two caveats on the slope

**The data profile is U-shaped; the model's is monotone.** Investment falls from
0.353 at age 1 to a trough of 0.241 at 12, then nearly triples to 0.650 by 17.
`exp(sigma_2_0 + sigma_2_1·(t−1))` cannot bend. Two group means are the most this
functional form can honestly be asked to match — a good fit on them is *not* the
model reproducing the age profile.

**The late moment contains an end-of-horizon spike.** The model's `e_p` runs
2.83 → 4.09 → 7.17 over ages 15–17: with the age-18 handoff approaching,
investment pays off immediately and parents front-load it. Ages 16–17 are 2 of 8
years in the late group but lift its mean from 2.07 to 2.96 — a 43% distortion.
That is the *terminal condition*, not the elasticity slope, so `sigma_2_1` will
partly absorb it. The data rises at 16–17 too (college spending), roughly 11×
less steeply. To exclude it, set the late group to ages 10–15 in **both**
`AGE_SPLIT` handling in `tools/make_smm_targets.py` and `model_moments`.

## Two things to know before you read a result

**1. The moments are not independent.** The budget binds every period:

```
c_p + e_p + saving = (1+r)·a + after-tax income + y
```

The model currently spends 2.19 on `e_p` against a data value of 0.39. Cutting
investment frees ~1.8 units, which pushes consumption up *on its own* — so fixing
the investment moment may largely fix consumption for free. Do not read the three
moments as three independent successes.

The flip side: the targets *jointly* imply a saving rate. At the current wage
process they leave `5.83 − 3.16 − 0.39 = 2.27` per period, i.e. **39% of
resources**, which over 17 years accumulates to far more than the ~$250k terminal
assets discussed elsewhere. The run prints the implied saving rate and terminal
assets on every report so this tension stays visible instead of hiding inside a
converged objective.

**2. SDs are reported but not targeted.** The model's only cross-sectional
heterogeneity is a 5-node wage shock plus initial asset, HC and college draws.
It cannot reach the data's dispersion — leisure SD is 7.4× too small, consumption
4× too small. Targeting SDs now would push parameters to extremes chasing
variance the model structurally cannot generate, and damage the means doing it.
Adding heterogeneity is a model change, not an estimation setting.

## Development tests and estimates

Five restarts are a **development test**; a final estimation may use 100 to 1,000 restarts
with a pre-testing pool about ten times larger (the paper's benchmark ratio). `--preset`
records which one a run is:

| preset | budgets it supplies | passing means |
|---|---|---|
| `smoke` | 64 Sobol', 5 restarts × 60 evals, no polish | it ran: finite outputs, workers, checkpoints. **Convergence not required** — the verdict line says `execution PASSED` |
| `integration` | 150 valid draws (cap 1,500), 5 × 500, polish 200 | a realistic local solve and a bounded polish converge and are reported correctly; no global-fit claim |
| `pilot` | 1,000 valid (cap 10,000), 50 × 1,500, polish 1,000; a too-small valid pool fails | quality against time, cap adequacy, basins, worker counts |
| `production` | 10,000 valid (cap 100,000), 1,000 × 2,000, polish 4,000; refuses `--quick` | a substantial global search — to be followed by the validation below. Set `--local-evals` from pilot traces. |

The budgets are starting points from `tiktak_fix_plan.md`'s run ladder, not tuned values.

**A five-restart run is not the first five restarts of a 1,000-restart run.** The mixing
weight is θ_j = min(max(0.1, √(j/K)), 0.995): with K = 5 the five restarts walk the whole
schedule (0, 0.63, 0.77, 0.89, 0.995); with K = 1000 the first five all sit at 0.1. To look
at the start of a production run, run the production configuration with
`--stop-after-restarts 5` (and `--reuse-pretest` to avoid paying for its pool twice), then
continue it with `--resume`.

## What a run reports — execution, convergence, acceptance

`estimates.toml` and the log keep separate things separate (2026-09-27):

| field | meaning |
|---|---|
| `point_origin`, `point_origin_ret` | where the reported point came from (sobol, supplied, local restart, polish, refine) and that search's own NLopt code |
| `verification` | `verified` when a later converged solve returned the same point (distance ≤ 1e-9 of the box) — so an unchanged optimum is no longer reported as unconverged. Equal Q at a different point is not a verification |
| `execution_ok` | no stage threw (objective exceptions, a failed refinement) |
| `candidate_valid` | a valid point with a valid, in-domain value |
| `local_converged` | convergence evidence for the reported point **on the reporting objective**: after a coarse `--grid` search only the full-grid refinement's own evidence counts, and a refinement that stopped on `MAXEVAL_REACHED` is budget-limited however much it improved |
| `search_budget_complete` | the planned restarts ran (not paused) |
| `accepted`, `acceptance_reasons` | all of the above, plus no parameter within 2% of a bound; each failing condition listed |
| `verdict` | the line that separates execution from estimation by purpose (a smoke run passes on clean execution) |
| `n_restarts_effective`, `n_sobol_valid` | how many restarts actually ran and how many pre-testing draws were valid (never assume N = 1000 means 1000 useful points) |

None of these is a claim of global optimality. Optimizer termination, economic fit and
identification are separate claims: a converged restart says the local test was met; `Q`
says how far the moments are; neither says the parameters are identified.

**On Q and the moment count.** Just-identification by counting moments and parameters does
not guarantee Q = 0 — the model may not reach the data, and the moments are not independent
(the budget ties them). Over-identification does not logically force Q > 0 either. Count is
not rank.

**Before a result travels**, use the existing tools rather than new definitions:
`jacobian.jl` / `tools/check_jacobian_rank.jl` (local identification: singular values, weak
directions — over several finite-difference steps), `profile_param.jl` (weak directions and
bound pressure), `grid_sensitivity.jl` (numerical accuracy), `standard_errors.jl`. If the
search ran on a coarse grid, refine SEVERAL distinct good candidates on the reporting grid,
not only the winner — a coarse ranking of basins can change. Keep common random numbers within
an optimisation; check other simulation seeds and sample sizes separately afterwards.

## Re-optimizing from a point: `tools/reopt.jl`

A bounded re-optimization from one or more given points, without a new global search (ported from v2's
`tools/reopt_v5e.jl` on 2026-10-02). `--targets` and `--start` are required.

```bash
julia --project=. tools/reopt.jl --outdir <dir> --targets <targets.toml> --start <estimates.toml>[,<toml>...]
      [--fix name=value,...] [--bounds name=lo:hi,...] [--local-evals 300] [--procs 1]
      [--sobol 0 | --sobol N --restarts K [--polish-evals P]] [--extra-moments a,b --extra-targets <toml>]
      [--free omega=LO:HI] [--parent-extra k=v,...] [--child-extra k=v,...] [--resume]
```

- `--sobol 0` (default): pure local Nelder-Mead (or `--local-alg bobyqa`) from every start, in parallel over
  `--procs` workers; `--ftol-rel` (default 1e-4), `--init-step`, checkpoints every `--checkpoint-every` evaluations,
  `--resume` restarts from the checkpointed best (not an exact optimizer-state resume, and the record says so).
- `--sobol N`: a bounded TikTak run through the module (its checkpoints, pre-testing cache, asynchronous restarts
  on `--procs` > 1, `--stop-after-restarts`, `--resume`).
- `--fix` holds estimated parameters (or `omega`) at a value; `--bounds` sets run-specific boxes; `--free omega`
  estimates omega as one extra coordinate (allowed because mu is not searched here); `--extra-moments` adds rows to
  Q for an experiment, reported apart as `Q_extra`.
- A start is VALIDATED, never clamped: non-finite or out-of-box values refuse the run. The objective identity
  (`tools/reopt_identity.jl`: target contents, extra rows, moments, parameters and boxes, numerics, overrides,
  model and adapter source) is checked against any checkpoint before the model is warmed up.
- Writes `results.toml`, `best_estimates.toml` (usable with `run_smm.jl --init-from`), `table.txt` (Q and each
  row's t-statistic and share, `tools/experiment_table.jl`) and `run.log`.


## Reading the output

`Q` is the weighted relative distance — a sum of squared percentage gaps, so
`Q = 0` is a perfect match and `Q ≈ 21` (the current incumbent) means the moments
are badly off. The report prints each moment in model units *and* in dollars or
hours per week, because "0.53" is hard to sanity-check and "59 hours a week" is
not.

The untargeted block underneath is what tells you whether a good `Q` is
believable: if the saving rate or terminal assets have gone somewhere absurd to
buy a good fit on ten moments, that is worth knowing before the numbers travel.

## What SMM is doing here, in four paragraphs

The model has parameters nobody can observe directly — how much parents value
leisure, how productive money is at building a child's human capital. But for any
*guess* at those parameters the model can be solved and a cohort simulated, which
produces simulated counterparts of things that *are* observed: average
consumption, average leisure, average investment.

Simulated Method of Moments picks the parameters that make the simulated averages
line up with the averages in the data. "Method of moments" because it matches
moments (here, means) rather than a likelihood; "simulated" because this model has
no closed form, so the moments have to come out of a simulation.

The thing being minimised is `Q`, the summed squared relative gap between the
ten simulated moments and their ten data counterparts. Minimising it is hard because `Q`
has no derivative anyone can write down and may have several local minima — hence
TikTak (`../src/tiktak.jl`), which scatters Sobol points over the parameter box to
find promising regions, then runs local searches from the best of them.

One detail that makes the whole thing work: **common random numbers**. Every
evaluation builds the model with the same `seed`, so the simulated shocks are
identical across parameter guesses. Without that, `Q` would jump around with
simulation noise and no derivative-free optimizer could converge — it would be
chasing the random number generator instead of the parameters.

## Files

| File | What |
|---|---|
| `run_smm.jl` | Driver: worker setup, budget, progress, TikTak, logging, reporting. Ported from v2 on 2026-10-02 (v1's specification). |
| `moments.jl` | Targets, model moments, objective, fit report. The economics. Since 2026-10-02 the objective takes `child_extra`, `parent_extra` and `extra_moments` (defaults leave Q bit-identical). |
| `runtime_projection.jl` | The runtime scenarios the runner prints. |
| `jacobian.jl`, `standard_errors.jl`, `sensitivity.jl`, `profile_param.jl`, `grid_sensitivity.jl`, `finite_differences.jl` | Post-estimation tools (identification, sampling uncertainty, target sensitivity, profiles, grid accuracy). |
| `selftest.jl` | Injects each failure a guard exists for and checks that it fires; its optimizer checks (A2, A3) run on the TikTak module. |
| `../src/TikTak/` | The optimizer as a module (Arnoud, Guvenen & Kleineberg 2022), identical to v2's: configuration and validation, status and acceptance, local solves, pre-testing and its cache, the versioned checkpoint, the asynchronous scheduler, presets. Loading it starts no worker. |
| `../src/tiktak.jl` | The adapter every caller includes; brings `tiktak`, `ret_class`, `ret_tally`, `tiktak_selftest` into scope. |
| `../../tools/test_tiktak.jl` | 455 synthetic regression tests (no model). |
| `../../tools/test_smm_resume.jl` | `--resume` on real checkpoints: every identity field refused by name; old-format runs refused. |
| `../../tools/test_runner_start.jl`, `test_runner_geometry.jl`, `test_tiktak_integration.jl`, `test_runtime_projection.jl` | The runner's flags end to end, its geometry flags, the optimizer on the real objective at `--quick` grids, the runtime estimator. |
| `../../tools/reopt.jl` (+ `reopt_identity.jl`, `reopt_settings.jl`, `reopt_objective.jl`, `start_loader.jl`, `run_bounds.jl`, `experiment_table.jl`) | Re-optimization from given points (above); `test_reopt_identity.jl`, `test_reopt_integration.jl` test it. |
| `../../tools/test_penalties.jl` | The penalty and failure-origin rules the optimizer relies on. |
| `../../tools/bench_tiktak.jl` | Measurements: step sizes and geometry, scheduling, checkpoint cost, the real objective's start-up and cold/warm cost. |

