STRASSEN - CA2 Group 9, Member 3 (n=1)

Requires Python 3.9+ (https://www.python.org/downloads/),
NumPy (https://numpy.org/install/) and
Matplotlib (https://matplotlib.org/stable/install/index.html).
Validated versions are pinned in requirements.txt.

Run all commands from the 5201-CA2 repository root.

Install dependencies:
  python -m pip install -r strassen/requirements.txt

Full experiment (three groups, 16-2048):
  python -u strassen/src/run.py --output result/strassen_run2
  python strassen/src/analyze.py result/strassen_run2

Short check (three groups, 16, 32, 64):
  python -u strassen/src/run.py --max-size 64 --output result/strassen_check
  python strassen/src/analyze.py result/strassen_check

Use a new output directory for each run and pass the same directory to
analyze.py. Input preparation is automatic; no separate setup script is needed.

Input: data/G0, G1, G2, each containing A_SIZE.npy and B_SIZE.npy for
SIZE=16,32,64,128,256,512,1024,2048. Use --data PATH for another directory
with the same layout. Input checks and hashes are recorded automatically.
Default: three groups, one repetition, cutoff 32, sequential and
p=1..min(6,usable CPUs). Existing output directories cannot be overwritten;
interrupted runs cannot resume.

Files (under strassen/src/):
  strassen.py - seven recursive products, i-j-k base case, Pool tasks.
  run.py      - input loading, timing, verification and raw records.
  analyze.py  - cumulative timing tables, speedups and plots.

Timing includes input conversion, computation, communication and fresh Pool
startup/exit for each pair. File loading, input checks, output verification
and writing are excluded. No explicit pre-call garbage collection is used.
Workers recurse sequentially; all core arithmetic uses Python integer loops.
NumPy handles inputs only; five numerical-library thread limits are set to 1.
Verification: eight modular-vector checks, twelve exact element checks and
cross-mode result hashes, all outside timing.

Outputs in the selected directory:
  measurements.jsonl - raw timings and verification records.
  environment.json, configuration.json, input_manifest.json - run metadata.
  status.json / complete.json - progress / successful completion.
  derived/B-1.1.csv, B-1.2.csv, C-1.1.csv, C-1.2.csv - report tables.
  derived/D-1.1.csv, D-1.2.csv - cumulative speedups.
  derived/D-1.1_processes.png, D-1.1_workload.png - required plots.
  derived/D-1.2_groups.png - supplementary group plot.
Short runs produce partial tables; use full runs for the final report.
