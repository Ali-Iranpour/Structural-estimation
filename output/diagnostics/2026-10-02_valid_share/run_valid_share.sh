#!/bin/sh
# Valid share of the search box at the production grids, real inputs (2026-10-02, item 5 of the plan).
# Targets: 2026-10-02_193217_281837 (composition tables present). 400 Sobol' points, 16 workers.
W=/srv/project/speech/apps/Structural-estimation/temp/2026-10-02_memo19_merge
D=$W/output/diagnostics/2026-10-02_valid_share
T=$W/output/smm_runs/2026-10-02_193217_281837_targets/targets.toml
cd $W || exit 1
echo "$(date '+%F %T') started: 400 points, grid 30, 16 workers, code $(git rev-parse --short HEAD) + uncommitted BC-share edits" > $D/STATUS
unset SMM_TEST_FIXTURES
timeout 14400 julia --project=. --threads=1 --startup-file=no tools/measure_valid_share.jl $T $D --n 400 --procs 16 --grid 30 > $D/run.log 2>&1
rc=$?
echo "$(date '+%F %T') finished exit=$rc" >> $D/STATUS
if [ $rc -eq 0 ]; then touch $D/DONE; else touch $D/FAILED; fi
