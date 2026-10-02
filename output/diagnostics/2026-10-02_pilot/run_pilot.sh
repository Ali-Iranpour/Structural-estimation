#!/bin/sh
# The 2026-10-02 pilot (Ali, 19:55): two arms at once, 20 workers each, ~2.6 h. Purpose: learn where the
# search goes on the real inputs -- the ranges (they stay as they are, option a) and a start for the next run.
#   arm A: theta0 (output/smm_runs/2026-10-02_theta0_start/theta0.toml) supplied as ONE candidate
#   arm B: Sobol' draws only
# Same everything else: --preset pilot, targets 2026-10-02_193217_281837, production grids, draws until 150 valid
# (cap 8,000 attempts), 20 local searches (Nelder-Mead, up to 600 evaluations each) all at once on 20 workers,
# immediate_mixed start-up, no polish (a learning run, not the estimate; the verdict will read NOT ACCEPTED).
# Code: worktree temp/2026-10-02_pilot_code, detached at 9e2d760, frozen for the run.
C=/srv/project/speech/apps/Structural-estimation/temp/2026-10-02_pilot_code
W=/srv/project/speech/apps/Structural-estimation/temp/2026-10-02_memo19_merge
D=/srv/project/speech/apps/Structural-estimation/temp/2026-10-02_memo19_merge/output/diagnostics/2026-10-02_pilot
T=$W/output/smm_runs/2026-10-02_193217_281837_targets/targets.toml
A=$W/output/smm_runs/2026-10-02_195733_pilot_A_theta0
B=$W/output/smm_runs/2026-10-02_195733_pilot_B_random
say() { echo "$(date '+%F %T') $*" >> $D/STATUS; }
cd $C || exit 1
unset SMM_TEST_FIXTURES
rm -f $D/DONE $D/FAILED
[ -z "$(git status --porcelain -- code tools)" ] || { say "FAILED: the code worktree has uncommitted edits"; touch $D/FAILED; exit 1; }
say "start; code $(git rev-parse --short HEAD); targets $(sha256sum $T | cut -c1-16); arm A -> $A; arm B -> $B"
COMMON="--preset pilot --targets $T --procs 20 --sobol-valid 150 --sobol 8000 --restarts 20 --local-procs 20 --local-evals 600 --skip-polish --bootstrap immediate_mixed"
( timeout 18000 julia --project=. --threads=1 --startup-file=no code/smm/run_smm.jl $COMMON --init-from $W/output/smm_runs/2026-10-02_theta0_start/theta0.toml --outdir $A > $D/arm_A.console.log 2>&1
  echo $? > $D/arm_A.exit; say "arm A exit=$(cat $D/arm_A.exit)" ) &
( timeout 18000 julia --project=. --threads=1 --startup-file=no code/smm/run_smm.jl $COMMON --outdir $B > $D/arm_B.console.log 2>&1
  echo $? > $D/arm_B.exit; say "arm B exit=$(cat $D/arm_B.exit)" ) &
wait
[ "$(cat $D/arm_A.exit)" = 0 ] && [ "$(cat $D/arm_B.exit)" = 0 ] && { say "both arms exit 0"; touch $D/DONE; } || { say "an arm did not exit 0"; touch $D/FAILED; }
