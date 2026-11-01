# Weather GPU Benchmark

Measure the cost of verifying weather forecasts with NumPy CPU, PyTorch CPU and CUDA: **Bias, MAE and RMSE** on synthetic FP32 arrays. This learning portfolio focuses on reproducible timing, numerical agreement and the difference between GPU compute speed and complete workflow speed.

## Three key findings

1. **Fast GPU compute does not imply the same end-to-end speedup.** At 10M grid points in run02, CUDA compute takes 2.055 ms versus NumPy's 30.368 ms (14.78x). Including input transfer and scalar return takes 13.533 ms (2.24x). At 1M, this GPU E2E path is slower than NumPy in both baseline runs.
2. **Reusing inputs on GPU removes substantial transfer cost.** In the resident experiment at 10M, repeatedly uploading inputs takes 14.086 ms per iteration; keeping inputs on GPU and returning all three metrics takes 2.194 ms (6.42x faster). This excludes the initial upload and reuses the same arrays; it is not a whole weather-model speedup.
3. **A speedup ratio can rise because the CPU baseline slows down.** Between run01 and run02 at 10M, NumPy takes 16.12% longer while CUDA compute takes 0.81% longer. The compute speedup nevertheless rises from 12.83x to 14.78x. Inspect absolute times and timing spread alongside speedup.

## Hardware and software

Observed on 2026-11-01 in the existing WSL environment. The [hardware inventory](results/hardware_inventory.json) was captured **after the experiments**. It identifies the current machine, but does not establish driver version, clocks, thermals or background load during earlier runs.

| Item | Observed value |
|---|---|
| CPU | Intel Core i5-12400F; 12 logical CPUs visible to WSL |
| GPU | NVIDIA GeForce RTX 3050, 8192 MiB (8 GiB) VRAM |
| GPU compute capability | 8.6 |
| OS | Ubuntu 22.04.1 LTS on WSL2 |
| NVIDIA driver | 560.94 at inventory capture |
| Python / NumPy | 3.10.12 / 2.2.6 |
| PyTorch / CUDA runtime | 2.14.1+cu126 / 12.6, as reported by the environment |
| Protocol | FP32, seed 42, one PyTorch CPU thread, 5 warmups, 20 measured repeats |

Each run preserves its own `environment.json`. NumPy thread environment variables were unset; thread counts were not explicitly controlled for NumPy. Dependencies in `requirements.txt` are compatibility ranges, not an exact lock of the measured environment.

## Quick start

Use Python 3.10+ in a virtual environment. In the existing WSL setup, activate it with `source ~/venvs/weather-gpu/bin/activate`.

```bash
python -m pip install -r requirements.txt
# Install a compatible CUDA-enabled PyTorch using https://pytorch.org/get-started/locally/
python -m unittest discover -s tests -v

# Use a new output folder for each run; the runner does not protect against overwrites.
python benchmark.py --sizes 1000000 5000000 10000000 --repeats 20 --warmup 5 --output results/my_run01
python plot_results.py results/my_run01/summary.csv --output results/my_run01/overview.png
```

CPU-only smoke test without PyTorch:

```bash
python benchmark.py --numpy-only --sizes 1000 10000 --repeats 3 --warmup 1
```

GPU-resident experiment (requires CUDA):

```bash
python benchmark.py --resident --sizes 1000000 5000000 10000000 --repeats 20 --warmup 5 --output results/my_resident01
python plot_results.py results/my_resident01/summary.csv --output results/my_resident01/overview.png
```

Outputs: `summary.csv` contains medians, quartiles and speedups; `samples.csv` preserves every timed repetition; `environment.json` describes the recorded environment and boundaries. The resident option also saves the resident GPU metric values and one initial-upload observation per size. Record `nvidia-smi`, CPU information and load **before** future runs.

## Benchmark method

All backends receive identical, finite synthetic FP32 forecast and observation arrays. Error is forecast minus observation. Bias = mean(error), MAE = mean(abs(error)), RMSE = sqrt(mean(error squared)). The validated entry point rejects empty, mismatched or non-finite inputs.

| Measurement | Timed boundary |
|---|---|
| NumPy / PyTorch CPU | Compute all three metrics with inputs already on CPU |
| CUDA compute | Compute metrics with inputs already on GPU; synchronized host wall time |
| H2D | Copy both pageable host tensors into existing GPU buffers |
| GPU E2E | Copy both inputs, compute metrics and return three Python scalars |
| GPU resident E2E (`--resident`) | Reuse GPU inputs, compute metrics and return three Python scalars |
| Initial upload (`--resident`) | Two copies into new GPU tensors, including allocation; one observation per size |

Five warmups precede 20 timed repetitions per method/size. CUDA synchronization brackets each wall-clock timing. Data generation, validation, disk I/O and GPU context initialization are excluded; intermediate computation allocations are included. These are not kernel-only CUDA-event timings. Initial upload has no repeat distribution and is excluded from resident steady-state speedups.

Report the median and p25/p75. IQR/median describes spread within a run, not a confidence interval. Methods run in fixed order. H2D and compute are separately sampled; adding their medians does not produce a measured E2E latency. PyTorch outputs are checked against NumPy at `rtol=atol=1e-5` before saving results.

The workload reuses the same arrays and applies unweighted global reductions. It does not yet include real weather data, time/coordinate alignment, regridding, area weights or missing-data handling. A CPU-origin input that changes each iteration still needs another transfer.

## Results: repeated baseline runs

The user supplied run01 and run02. Their recorded environments/protocols are identical; all **600 timing samples** reconcile with the saved medians, quartiles and speedups. These timing files do not save per-backend metric values, so they cannot independently establish numerical agreement.

| Grid points | CUDA compute ms, run01 / run02 | GPU E2E ms, run01 / run02 | Compute speedup, run01 / run02 | E2E speedup, run01 / run02 |
|---:|---:|---:|---:|---:|
| 1M | 0.296 / 0.315 | 1.697 / 1.468 | 4.26x / 4.37x | 0.74x / 0.94x |
| 5M | 1.074 / 1.075 | 7.583 / 7.024 | 12.09x / 11.53x | 1.71x / 1.76x |
| 10M | 2.038 / 2.055 | 13.823 / 13.533 | 12.83x / 14.78x | 1.89x / 2.24x |

![Run02 runtime, speedup, memory and transfer](results/rtx3050_run02/overview.png)

The memory panel uses `gpu_peak_allocated_mb`, calculated as bytes / 2^20, so the unit is **MiB** despite the CSV suffix. It is PyTorch peak allocated memory during CUDA compute, including resident inputs and intermediate tensors. It is not total process/device VRAM, allocator-reserved memory or a peak measurement of the entire E2E/resident workflow. Run02 reports 15.264, 80.004 and 152.592 MiB at 1M, 5M and 10M respectively.

See [run01 analysis](results/rtx3050_run01/analysis.md) and the [run01/run02 comparison](results/rtx3050_run02/comparison.md). At 1M/5M/10M, PyTorch CPU medians change by -56.40%/+45.10%/+10.35% across runs. Two sessions do not establish long-term stability; scheduling, clocks and load are possible factors, not diagnosed causes.

## Results: GPU-resident inputs

Executed directly through the existing WSL / RTX 3050 setup. All **360 timing samples** reconcile. Saved resident GPU Bias/MAE/RMSE values agree with NumPy at the specified tolerance.

| Grid points | NumPy ms | Reupload E2E ms | Resident + scalar return ms | Speedup vs reupload | Speedup vs NumPy |
|---:|---:|---:|---:|---:|---:|
| 1M | 1.223 | 1.572 | 0.386 | 4.07x | 3.16x |
| 5M | 12.483 | 6.921 | 1.152 | 6.01x | 10.84x |
| 10M | 28.047 | 14.086 | 2.194 | 6.42x | 12.78x |

![Resident experiment runtime, speedup, memory and transfer](results/rtx3050_resident_run01/overview.png)

Initial uploads including input allocation take 3.778 / 7.659 / 15.521 ms at 1M / 5M / 10M. These are single observations, not medians. The table excludes them from steady-state speedups. Only one resident session has been measured; this does not establish a universal acceleration factor. See the [full resident report](results/rtx3050_resident_run01/analysis.md).

## Reproduce the reports and figures

```bash
python scripts/analyze_run.py results/rtx3050_run01
python scripts/compare_runs.py results/rtx3050_run01 results/rtx3050_run02
python plot_results.py results/rtx3050_run02/summary.csv --output results/rtx3050_run02/overview.png
python plot_results.py results/rtx3050_resident_run01/summary.csv --output results/rtx3050_resident_run01/overview.png
```

`analyze_run.py` is the original run01 report generator with run01-specific narrative. `compare_runs.py` compares the original five-method runs. `plot_results.py` supports CPU-only, original GPU, resident and historical CSVs.

## Historical results

The five original rows from the shared conversation remain unchanged in [legacy_rtx3050.csv](results/legacy_rtx3050.csv), with their [historical chart](results/legacy_benchmark.png). They lack raw repeat timings and full metadata. The reported maximum compute speedup is 13.73x at 20M. At 50M, H2D + compute implies an **estimated** 1.68x speedup, not directly measured E2E latency. The legacy memory field's measurement boundary is unknown. Do not merge these rows with the current protocol or treat them as independently reproduced results.

## Roadmap

1. **FP32 baseline: completed.** Repeated timings, raw samples, four-panel figures, hardware inventory and the resident-input experiment. Further repeat sessions and profiling remain useful for assessing stability.
2. **v3: FP32 vs FP16.** Use identical source inputs and compare runtime, peak allocated memory and Bias/MAE/RMSE errors. Separate input quantization from arithmetic error, check finite outputs/overflow, and evaluate FP32 accumulation for low-precision inputs. Report the speed, memory and accuracy tradeoff before recommending a precision.
3. **v4: real weather verification.** Read public NetCDF/ERA5 data with xarray; align valid times, grids and units; define NaN/mask policy and area weighting; validate Bias/MAE/RMSE before timing.
4. **Weather AI ecosystem.** Explore Earth-2 / PhysicsNeMo after the real-data verification workflow is established. Consider profiling, pinned memory and compiled/fused kernels against that workflow.

Learning goal: explain how data movement, numerical precision and measurement boundaries affect a useful weather-verification pipeline.
