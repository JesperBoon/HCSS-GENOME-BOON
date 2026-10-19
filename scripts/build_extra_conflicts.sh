#!/bin/zsh
# build_extra_conflicts.sh: add more conflicts to the dashboard (quarterly + weekly data).
#
# FREE: only unauthenticated count queries, no API key and no credits. Resumable.
# About 18 minutes per conflict. New conflicts appear in the dashboard's conflict list by themselves.
#
# Keep the Mac awake while it runs:   caffeinate -i zsh scripts/build_extra_conflicts.sh
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python

EXTRA=("Iran USA" "Russia USA" "India Pakistan" "Afghanistan Pakistan")

for pair in $EXTRA; do
  echo "== ${pair}: quarterly counts"
  $PY scripts/build_dyad_counts.py ${=pair}
  echo "== ${pair}: weekly counts"
  $PY scripts/build_weekly_counts.py ${=pair}
done
echo "Done. Restart the dashboard to see the new conflicts."
