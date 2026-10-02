#!/bin/sh
# run_recovery.sh -- the memo-19 parameter-recovery test (docs/SMM_MEMO19.md section 6), Ali 2026-10-02.
# Two processes, one thread each: mode tech (12 technology parameters free) and all (20 free).
# Code: worktree temp/2026-10-02_memo19_merge, branch merge/port-memo19 (refuses to start if it has uncommitted edits).
# Fixture composition and placeholder wage loading (tools/smm_test_fixtures.jl): a test, not an estimate.
cd /srv/project/speech/apps/Structural-estimation/temp/2026-10-02_memo19_merge
OUT=output/smm_runs/2026-10-02_105731_recovery
T=output/smm_runs/2026-10-01_225333_185835_targets/targets.toml
say() { echo "$(date '+%F %T') $*" >> $OUT/STATUS; }
rm -f $OUT/DONE $OUT/FAILED
[ -z "$(git status --porcelain -- code tools)" ] || { say "FAILED: uncommitted code edits"; touch $OUT/FAILED; exit 1; }
say "start; code commit $(git rev-parse --short HEAD); targets $(sha256sum $T | cut -c1-16)"
for m in tech all; do
  ( julia --project=. --threads=1 --startup-file=no tools/test_param_recovery.jl $T $m $OUT > $OUT/$m.console.log 2>&1
    echo $? > $OUT/$m.exit; say "$m exit=$(cat $OUT/$m.exit)" ) &
done
wait
[ "$(cat $OUT/tech.exit)" = 0 ] && [ "$(cat $OUT/all.exit)" = 0 ] && { say "both exit 0"; touch $OUT/DONE; } || { say "a mode did not exit 0"; touch $OUT/FAILED; }
