#!/usr/bin/env bash
# Run from the 5201-CA2 repository root in Git Bash.
set -e

python -u strassen/src/run.py --output result/strassen_run1
python strassen/src/analyze.py result/strassen_run1

python -u cw.py

python -u cannon.py --mode benchmark --max-size 2048 --groups 0,1,2 --max-procs 6 --pool-timing included

python -u fox.py
