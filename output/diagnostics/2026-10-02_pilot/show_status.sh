#!/bin/sh
# show_status.sh -- one look at everything running on 2026-10-02 evening:  sh <this file>
W=/srv/project/speech/apps/Structural-estimation/temp/2026-10-02_memo19_merge
R=$W/output/smm_runs
echo "=== tmux sessions (attach: tmux attach -t NAME; detach: Ctrl-b then d)"
tmux ls 2>/dev/null | grep '^v1_'
echo; echo "=== PILOT driver ($W/output/diagnostics/2026-10-02_pilot/STATUS)"
cat $W/output/diagnostics/2026-10-02_pilot/STATUS
for a in A_theta0 B_random; do
  d=$R/2026-10-02_195733_pilot_$a
  echo; echo "--- arm $a: last progress lines ($d/run.log)"
  tail -n 3 $d/run.log
  [ -f $d/restarts.csv ] && echo "    local searches finished: $(tail -n +2 $d/restarts.csv | wc -l) of 20"
done
echo; echo "=== RECOVERY test ($R/2026-10-02_193856_recovery_real)"
cat $R/2026-10-02_193856_recovery_real/STATUS
for m in tech all; do printf "  %-4s " $m; grep -E '^ +eval' $R/2026-10-02_193856_recovery_real/recovery_$m.log | tail -n 1; done
echo; echo "=== TEST SUITE on the real inputs"
cat $W/../2026-10-02_merge_checks/real_inputs_suite/STATUS
echo; echo "=== server load (112 cores)"; uptime
