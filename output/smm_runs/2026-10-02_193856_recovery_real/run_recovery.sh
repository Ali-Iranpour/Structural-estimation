#!/bin/sh
# run_recovery.sh -- the memo-19 parameter-recovery test rerun on the REAL inputs (Ali 2026-10-02, plan item 5):
# the data's composition (targets 2026-10-02_193217_281837), the anchored wage loading (sd(log AFQT) 0.20, lnw0 1.756,
# both provisional), BothCollege share 0.2588 from [constants]. Two processes, one thread each: tech (12 free), all (20).
# Code: HEAD 5b5e361 + the UNCOMMITTED edits saved in uncommitted_code.patch (Ali: do not commit yet).
cd /srv/project/speech/apps/Structural-estimation/temp/2026-10-02_memo19_merge
OUT=output/smm_runs/2026-10-02_193856_recovery_real
T=output/smm_runs/2026-10-02_193217_281837_targets/targets.toml
say() { echo "$(date '+%F %T') $*" >> $OUT/STATUS; }
unset SMM_TEST_FIXTURES
rm -f $OUT/DONE $OUT/FAILED
git diff -- code tools | cmp -s - $OUT/uncommitted_code.patch || { say "FAILED: code differs from the saved patch"; touch $OUT/FAILED; exit 1; }
say "start; HEAD $(git rev-parse --short HEAD) + uncommitted_code.patch; targets $(sha256sum $T | cut -c1-16)"
for m in tech all; do
  ( timeout 21600 julia --project=. --threads=1 --startup-file=no tools/test_param_recovery.jl $T $m $OUT > $OUT/$m.console.log 2>&1
    echo $? > $OUT/$m.exit; say "$m exit=$(cat $OUT/$m.exit)" ) &
done
wait
[ "$(cat $OUT/tech.exit)" = 0 ] && [ "$(cat $OUT/all.exit)" = 0 ] && { say "both exit 0"; touch $OUT/DONE; } || { say "a mode did not exit 0"; touch $OUT/FAILED; }
