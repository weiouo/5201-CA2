STRASSEN - CA2 Group 9, Member 3 (n=1)

REQUIREMENTS
Python 3.9+: https://www.python.org/downloads/
NumPy: https://numpy.org/install/
Matplotlib: https://matplotlib.org/stable/install/index.html
Package version ranges are listed in requirements.txt.

RUN
Run these commands from the 5201-CA2 repository root:

python -m pip install -r strassen/requirements.txt
python strassen/src/prepare_manifest.py
python strassen/src/benchmark.py --output strassen/results/run1
python strassen/src/analyze.py strassen/results/run1

Input files: data/G0, data/G1 and data/G2, each containing A_SIZE.npy and
B_SIZE.npy for SIZE = 16,32,64,128,256,512,1024,2048. The manifest script
checks the 48 input files and records their hashes; it does not generate data.

Defaults: three groups, one repetition, cutoff 32, sequential execution and
1 to min(6, usable CPU cores) worker processes. Use a new output directory
for each run. Interrupted runs cannot resume.
For other inputs, run prepare_manifest.py --data PATH --output MANIFEST,
then pass --manifest MANIFEST to benchmark.py. The same CA2 layout is required.
Use --repeats N to repeat the experiment N times.

SMALL RUN (16, 32 and 64 only)
python strassen/src/benchmark.py --max-size 64 --output strassen/results/check1
python strassen/src/analyze.py strassen/results/check1

OUTPUTS
All outputs are written under the specified output directory:
- derived/B-1.1.csv, B-1.2.csv, C-1.1.csv, C-1.2.csv: timing tables.
- derived/D-1.1_processes.png, D-1.1_workload.png: D-1.1 plots.
- derived/D-1.1.csv, D-1.2.csv: speedups; D-1.2_groups.png: group plot.
- measurements.jsonl: individual timings and verification results.
- environment.json, configuration.json, input_manifest.json: run metadata.
- status.json: progress; complete.json: successful completion.
Small runs produce partial tables. Report tables require the full run.
Generated results, manifests and caches are excluded from Git.

IMPLEMENTATION
strassen.py implements the seven products in Appendix A, with an i-j-k
base case at size <=32. Pool workers handle the seven top-level products
and recurse sequentially. Core arithmetic uses Python integers and loops.
benchmark.py runs the experiment; analyze.py produces tables and plots;
common.py and prepare_manifest.py handle inputs; validation.py checks outputs.

Each parallel multiplication creates a fresh Pool, including p=1 and small
inputs. Timing includes pool startup/exit, communication and multiplication;
it excludes input loading/conversion, verification, pre-call garbage collection
and file output. NumPy is used for input handling, with five numerical-library
thread limits set to 1 before import. No shared memory is used.
Verification uses eight modular-vector checks, twelve exact element checks
and output-hash agreement across modes. It runs outside the timed region.
