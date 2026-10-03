#!/bin/sh
# The K = 16 run (Ali, 2026-10-02 ~21:45): 10,000 Sobol' attempts on 36 workers, K = 16 restarts with the
# AUTOMATIC local width floor(sqrt(16)) = 4 (no --local-procs), 450 local evaluations, polish 300,
# started from the pilot's best end point. Projected ~7.2-7.8 h; timeout 9 h (checkpointed, resumable).
# Gates, checked by this script before anything starts:
#   G1 the integration test of TikTak 2.3.0-dev on the real inputs passed (its DONE)
#   G2 the pilot finished with both arms exit 0 (its DONE)
# Code: worktree temp/2026-10-02_run_k16_code (detached, frozen). Targets: 2026-10-02_193217_281837.
W=/srv/project/speech/apps/Structural-estimation/temp/2026-10-02_memo19_merge
C=/srv/project/speech/apps/Structural-estimation/temp/2026-10-02_run_k16_code
D=$W/output/diagnostics/2026-10-02_run_k16
I=$W/output/diagnostics/2026-10-02_tiktak23_integration
P=$W/output/diagnostics/2026-10-02_pilot
T=$W/output/smm_runs/2026-10-02_193217_281837_targets/targets.toml
say() { echo "$(date '+%F %T') $*" >> $D/STATUS; }
fail() { say "FAILED: $*"; touch $D/FAILED; exit 1; }
rm -f $D/DONE $D/FAILED
say "waiting for G1 (integration test) and G2 (pilot)"
while [ ! -e $I/DONE ] && [ ! -e $I/FAILED ]; do sleep 60; done
[ -e $I/DONE ] || fail "G1: the integration test of the new module did not pass ($I/STATUS); nothing started"
say "G1 passed: $(tail -1 $I/STATUS)"
while [ ! -e $P/DONE ] && [ ! -e $P/FAILED ]; do sleep 60; done
[ -e $P/DONE ] || fail "G2: the pilot did not finish cleanly ($P/STATUS); nothing started"
best=""; bestq=""
for a in A_theta0 B_random; do
  e=$W/output/smm_runs/2026-10-02_195733_pilot_$a/estimates.toml
  [ -f $e ] || fail "G2: $e is missing"
  q=$(grep -E '^Q_final' $e | head -1 | sed 's/^Q_final *= *//; s/ .*//')
  say "pilot arm $a: Q_final $q"
  if [ -z "$bestq" ] || [ "$(echo "$q < $bestq" | bc -l)" = 1 ]; then best=$e; bestq=$q; fi
done
say "G2 passed; start from $best (Q_final $bestq)"
cd $C || fail "no code worktree $C"
[ -z "$(git status --porcelain -- code tools)" ] || fail "the code worktree has uncommitted edits"
OUT=$W/output/smm_runs/$(date +%Y-%m-%d_%H%M%S)_k16
unset SMM_TEST_FIXTURES
say "launch: code $(git rev-parse --short HEAD); targets $(sha256sum $T | cut -c1-16); out $OUT"
timeout 32400 julia --project=. --threads=1 --startup-file=no code/smm/run_smm.jl --preset pilot --targets $T \
  --procs 36 --sobol 10000 --restarts 16 --local-evals 450 --polish-evals 300 --bootstrap immediate_mixed \
  --init-from $best --outdir $OUT > $D/run.console.log 2>&1
rc=$?
say "run exit=$rc (124 = the 9 h timeout; resume with the same command plus --resume $OUT)"
[ $rc -eq 0 ] && touch $D/DONE || touch $D/FAILED
