# CEG5201 CA2 – Matrix Multiplication Benchmarking

## 1. Overview

This project implements four matrix multiplication algorithms using sequential and multiprocessing approaches:

- Strassen
- Coppersmith–Winograd
- Cannon
- Fox

## 2. Requirements

- Python: https://www.python.org/downloads/
- NumPy: https://numpy.org/install/
- Matplotlib: https://matplotlib.org/stable/install/index.html
- Pandas: https://pandas.pydata.org/docs/getting_started/install.html
- Git Bash (Windows): https://git-scm.com/downloads

Install the required Python packages:

```bash
python -m pip install numpy matplotlib pandas
```

## 3. Input Data

All four algorithms use the same input matrices stored in the `data/` directory.

The dataset consists of three groups (G0, G1, G2), each containing eight pairs of square matrices with sizes 16, 32, 64, 128, 256, 512, 1024, and 2048.

The matrices were randomly generated using NumPy and saved as `.npy` files. No external dataset was used.

Example file paths:

```text
data/G0/A_16.npy
data/G0/B_16.npy
data/G1/A_32.npy
data/G1/B_32.npy
```

All required input files must be available before running the benchmarks.

## 4. Running the Benchmarks

**Step 1:** Download or clone the repository. (https://github.com/weiouo/5201-CA2.git)

**Step 2:** Install the required packages listed above.

**Step 3:** Open Git Bash and navigate to the `5201-CA2` repository root directory.

**Step 4:** Execute the following command:

```bash
bash run_all.sh
```

The script runs the four algorithms sequentially in the following order:

1. Strassen
2. Coppersmith–Winograd
3. Cannon
4. Fox

The script also runs the Strassen analysis program after its benchmark completes.

The script stops automatically if any command fails.

## 5. Benchmark Parameters

The provided script uses the parameters defined in each algorithm's Python implementation.

For Cannon, the script explicitly sets:

- Maximum matrix size: 2048
- Matrix groups: G0, G1, G2
- Maximum worker processes: 6
- Pool creation and shutdown time: included

For Strassen, the output directory is set to `result/strassen_run1`.

Coppersmith–Winograd and Fox are executed using their default hard-coded parameters. To change these settings, edit the corresponding Python files before running the script.

The Strassen output directory must be unused before starting a new run.

## 6. Output Results

Benchmark results are saved in the `result/` directory, with Strassen results stored under `result/strassen_run1`.

Execution times are measured for sequential and multiprocessing implementations. Speedup is calculated as:

**Speedup = Cumulative Sequential Time / Cumulative Multiprocessing Time**

All four algorithms should be executed on the same computer using identical input matrices for fair performance comparison.