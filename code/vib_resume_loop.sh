#!/bin/bash
# Auto-resume loop for the rot2 0.1-grid vib. Re-runs vib_stations.py --resume
# until the free-energy CSV reaches 121 rows (all stations), surviving the
# ~45-min background-task reap by being detached via setsid (new session).
# Guards: stops if no progress between attempts, if python exits 0 with <121
# rows, or after 25 attempts. Writes its own PID to a pid file as a kill switch.
cd "/Users/cafierom/Library/CloudStorage/OneDrive-UniversityofReading/Research-Reading/Python/python_bootcamp/Rotaxanes" || exit 1
LOG=output_files/rot2_vib_allstations_tblite_0p1.log
CSV=output_files/rot2_freeenergy_tblite.csv
PIDF=output_files/rot2_vib_resume_loop.pid
echo $$ > "$PIDF"
attempt=0
prev_rows=-1
while [ $attempt -lt 25 ]; do
  attempt=$((attempt+1))
  rows=$( [ -f "$CSV" ] && tail -n +2 "$CSV" | grep -c . || echo 0 ); rows=${rows:-0}
  echo "=== [loop] attempt $attempt: $rows/121 stations in CSV ===" >> "$LOG"
  if [ "$rows" -ge 121 ]; then echo "=== [loop] all 121 done before run; complete ===" >> "$LOG"; break; fi
  .venv/bin/python code/vib_stations.py --engine tblite \
    --input output_files/rot2_displaced_tblite.xyz \
    --all-stations --all-stations-step 0.10 --relax-fmax 0 --resume >> "$LOG" 2>&1
  rc=$?
  rows=$( tail -n +2 "$CSV" | grep -c . 2>/dev/null || echo 0 ); rows=${rows:-0}
  echo "=== [loop] attempt $attempt exited rc=$rc, now $rows/121 ===" >> "$LOG"
  if [ "$rows" -ge 121 ]; then echo "=== [loop] complete (121 rows) ===" >> "$LOG"; break; fi
  if [ "$rows" -le "$prev_rows" ]; then echo "=== [loop] NO PROGRESS ($rows <= $prev_rows) -- stopping to avoid infinite loop ===" >> "$LOG"; break; fi
  prev_rows=$rows
  if [ $rc -eq 0 ]; then echo "=== [loop] python exited 0 with only $rows/121 -- stopping ===" >> "$LOG"; break; fi
  sleep 5
done
final=$( tail -n +2 "$CSV" | grep -c . 2>/dev/null || echo 0 )
echo "=== [loop] finished after $attempt attempts, rows=$final/121 ===" >> "$LOG"
rm -f "$PIDF"