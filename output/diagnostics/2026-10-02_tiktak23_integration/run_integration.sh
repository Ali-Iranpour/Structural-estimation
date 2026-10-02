#!/bin/sh
# Integration test of TikTak 2.3.0-dev on the REAL inputs (Ali, 2026-10-02 evening): code = worktree
# temp/2026-10-02_tiktak23_check, detached at 63bd2ac, frozen; targets 2026-10-02_193217_281837; no fixtures.
C=/srv/project/speech/apps/Structural-estimation/temp/2026-10-02_tiktak23_check
D=/srv/project/speech/apps/Structural-estimation/temp/2026-10-02_memo19_merge/output/diagnostics/2026-10-02_tiktak23_integration
T=/srv/project/speech/apps/Structural-estimation/temp/2026-10-02_memo19_merge/output/smm_runs/2026-10-02_193217_281837_targets/targets.toml
cd $C || exit 1
unset SMM_TEST_FIXTURES
rm -f $D/DONE $D/FAILED
echo "$(date '+%F %T') start; code $(git rev-parse --short HEAD); targets $(sha256sum $T | cut -c1-16)" >> $D/STATUS
timeout 7200 julia --project=. --threads=1 --startup-file=no tools/test_tiktak_integration.jl $T --out $D/runs > $D/integration.log 2>&1
rc=$?; echo $rc > $D/integration.exit
echo "$(date '+%F %T') tiktak_integration exit=$rc" >> $D/STATUS
[ $rc -eq 0 ] && touch $D/DONE || touch $D/FAILED
