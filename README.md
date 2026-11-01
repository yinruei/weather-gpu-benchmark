# Weather GPU Benchmark

Measure the cost of verifying weather forecasts with NumPy CPU, PyTorch CPU and CUDA: **Bias, MAE and RMSE**. Synthetic FP32 and precision benchmarks are complemented by a first real-NetCDF pipeline using reanalysis temperatures and a manufactured forecast. This learning portfolio focuses on reproducible timing, numerical agreement and the difference between GPU compute speed and complete workflow speed.

## Three key findings

1. **Fast GPU compute does not imply the same end-to-end speedup.** At 10M grid points in run02, CUDA compute takes 2.055 ms versus NumPy's 30.368 ms (14.78x). Including input transfer and scalar return takes 13.533 ms (2.24x). At 1M, this GPU E2E path is slower than NumPy in both baseline runs.
2. **Reusing inputs on GPU removes substantial transfer cost.** In the resident experiment at 10M, repeatedly uploading inputs takes 14.086 ms per iteration; keeping inputs on GPU and returning all three metrics takes 2.194 ms (6.42x faster). This excludes the initial upload and reuses the same arrays; it is not a whole weather-model speedup.
3. **Lower precision trades accuracy for speed and memory.** Two precision runs show native FP16 about 1.81x faster than FP32 at 10M, with peak allocated memory 80.002 vs 152.590 MiB. Its RMSE error against the FP64 source reference is 0.001254, versus roughly 7.1e-8 for FP32. Casting FP16 inputs back to FP32 improves arithmetic accuracy but cannot undo input quantization; this eager mixed implementation is slower and has no peak-memory advantage.

## Hardware and software

Observed on 2026-11-01 in the existing WSL environment. The [hardware inventory](results/hardware_inventory.json) was captured **after the FP32 baseline experiments**. It identifies the current machine, but does not establish driver version, clocks, thermals or background load during earlier runs. The later precision experiments also save their own pre/post GPU state snapshots.

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

In the original synthetic benchmark, all backends receive identical, finite FP32 forecast and observation arrays. Error is forecast minus observation. Bias = mean(error), MAE = mean(abs(error)), RMSE = sqrt(mean(error squared)). The validated entry point rejects empty, mismatched or non-finite inputs. The later NetCDF pipeline defines joint masking and weighted reductions separately.

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

At 10M, the NumPy baseline slows down 16.12% between runs while CUDA compute slows down 0.81%. The compute speedup nevertheless rises from 12.83x to 14.78x. Compare absolute times alongside ratios.

## Results: GPU-resident inputs

Executed directly through the existing WSL / RTX 3050 setup. All **360 timing samples** reconcile. Saved resident GPU Bias/MAE/RMSE values agree with NumPy at the specified tolerance.

| Grid points | NumPy ms | Reupload E2E ms | Resident + scalar return ms | Speedup vs reupload | Speedup vs NumPy |
|---:|---:|---:|---:|---:|---:|
| 1M | 1.223 | 1.572 | 0.386 | 4.07x | 3.16x |
| 5M | 12.483 | 6.921 | 1.152 | 6.01x | 10.84x |
| 10M | 28.047 | 14.086 | 2.194 | 6.42x | 12.78x |

![Resident experiment runtime, speedup, memory and transfer](results/rtx3050_resident_run01/overview.png)

Initial uploads including input allocation take 3.778 / 7.659 / 15.521 ms at 1M / 5M / 10M. These are single observations, not medians. The table excludes them from steady-state speedups. Only one resident session has been measured; this does not establish a universal acceleration factor. See the [full resident report](results/rtx3050_resident_run01/analysis.md).

## v3: FP32 vs FP16 precision tradeoff

`precision_benchmark.py` compares three GPU-resident variants on exactly the same FP32 source arrays. Every timed call returns all three metrics as host scalars:

| Mode | Stored inputs | Arithmetic |
|---|---|---|
| `fp32` | FP32 | FP32 intermediates and outputs |
| `fp16_native` | FP16 | Native FP16 intermediate/output tensors; internal reduction accumulation is backend-defined |
| `fp16_fp32_math` | FP16 | Cast to FP32 before subtraction inside each call; FP32 intermediates and reductions |

Input conversion to FP16 and initial upload are excluded from timing. Only one mode's input tensors reside on GPU at a time. Each mode has 5 warmups and 20 measured repeats. Mode order is seeded per size and repeats across sessions; this is not an order-balanced experiment.

Two measured sessions preserve **360 timings with per-repeat Bias/MAE/RMSE values**. References are calculated in FP64 on (a) the original FP32 input values and (b) the quantized FP16 input values, separating input quantization from arithmetic error. Results at **10M** points:

| Mode | Median ms run01 / run02 | FP32 / mode runtime run01 / run02 | Input MiB | Peak allocated MiB |
|---|---:|---:|---:|---:|
| FP32 | 2.179 / 2.149 | 1.00x / 1.00x | 76.294 | 152.590 |
| FP16 native | 1.207 / 1.188 | 1.81x / 1.81x | 38.147 | 80.002 |
| FP16 inputs / FP32 math | 3.364 / 2.774 | 0.65x / 0.77x | 38.147 | 154.441 |

Maximum absolute metric errors across measured repeats against the original-source FP64 reference are identical in the two sessions at 10M:

| Mode | Bias error | MAE error | RMSE error |
|---|---:|---:|---:|
| FP32 | 2.29446e-8 | 6.55823e-8 | 7.10578e-8 |
| FP16 native | 8.99801e-5 | 4.34095e-4 | 1.25368e-3 |
| FP16 inputs / FP32 math | 1.05432e-5 | 2.15113e-5 | 2.48368e-3 |

![Precision runtime, memory and accuracy tradeoff](results/rtx3050_precision_run02/precision.png)

The mixed mode closely matches the FP64 reference **for quantized inputs**, but cannot recover the original input information. At 10M, quantization increases the reference RMSE by about 0.002484. Native FP16 arithmetic then changes RMSE by about -0.001230, partially cancelling that input error. Its smaller total RMSE error in this example does not make it a more reliable algorithm.

All benchmark outputs are finite, and all FP32/mixed results agree with their own input references at `rtol=atol=1e-5`. Native FP16 fails that diagnostic tolerance, and neither FP16 variant matches the original-source reference at that tolerance. This is a numerical diagnostic, **not an established weather-verification error budget**. Separate unit tests demonstrate native-half intermediate overflow and verify that casting before subtraction avoids those test failures. The mixed mode materializes FP32 tensors; it is not a fused low-precision implementation.

Keep FP32 as the baseline until real-data accuracy requirements justify a change. See the [first precision report](results/rtx3050_precision_run01/analysis.md) and [repeat precision report](results/rtx3050_precision_run02/analysis.md) for all sizes, quartiles, saved metric values and error decomposition. Two short sessions do not establish long-term stability or statistical significance.

```bash
python precision_benchmark.py --sizes 1000000 5000000 10000000 --repeats 20 --warmup 5 --output results/my_precision01
python scripts/analyze_precision.py results/my_precision01
```

The precision runner requires CUDA and refuses an existing output directory. Its CSV schema is separate from the original benchmark; use `analyze_precision.py` for validation and figures.

## v4: first real-NetCDF pipeline

The first case reads **real NCEP reanalysis air temperatures** from the [xarray tutorial dataset](https://docs.xarray.dev/en/stable/examples/area_weighted_temperature.html), selecting 2013-01-01 at 00/06/12/18 UTC over 30–50 N, 130–90 W. This is **sigma-level 0.995 air temperature**, not 2-m temperature or Taiwan station data. [Data and provenance](data/weather_demo/README.md) document the pinned source, checksum, selection and original metadata. Reanalysis data provided by NOAA PSL, Boulder, Colorado, USA; see the [source documentation](https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html).

The paired forecast is a deterministic **manufactured perturbation**. Ten deliberately missing time/grid values test joint masking. These are artificial errors and missing cells, not operational forecast skill or measured data availability.

The implemented flow is:

```text
NetCDF decode -> temperature units/coordinates -> exact valid-time/grid check
-> joint finite mask -> cosine-latitude weights -> CPU/CUDA metrics -> report/plots
```

`weather_verify.py` accepts canonical `time/lat/lon` dimensions or their supported aliases. It converts K/degC to FP32 kelvin, sorts coordinates, wraps longitude to [-180,180), rejects duplicate/mismatched coordinates, and normalizes weights over joint finite cells. Cosine-latitude weighting is restricted to regular latitude/longitude grids. All-missing or zero-weight inputs fail explicitly. Empty individual times produce blank metrics and zero coverage rather than a fabricated zero error.

**Caller responsibilities:** select the same physical variable and vertical level; supply UTC **valid time**, resolving initialization plus lead time beforehand. Matching coordinate arrays do not establish matching scientific meaning. No regridding, curvilinear grids or calendar conversion is implemented. The committed files use NetCDF3; reading NetCDF4/HDF5 needs an additional compatible xarray engine.

Two WSL / RTX 3050 sessions each preserve 15 per-backend/per-time metric rows and 80 timing samples. There are **602 joint finite values out of 612 (98.37%)**, with 98.41% of the total area-time weight retained. FP32 CPU/CUDA values agree with FP64 arithmetic on the same packed FP32 inputs and weights at `rtol=atol=1e-5`.

| Pooled metric | FP64 reference | NumPy FP32 | CUDA FP32 |
|---|---:|---:|---:|
| Bias K | 0.685742374 | 0.685742438 | 0.685742438 |
| MAE K | 0.770504044 | 0.770504117 | 0.770504057 |
| RMSE K | 0.934485432 | 0.934485435 | 0.934485435 |

These pool valid weighted cells over four sampled times. Pooled RMSE is calculated from pooled squared errors; it is not the mean of the four RMSEs and is not a continuous-day average.

| Measured boundary | NumPy median ms, run01 / run02 | CUDA median ms, run01 / run02 |
|---|---:|---:|
| Packed resident inputs -> metrics | 0.0071 / 0.0091 | 0.3022 / 0.2393 |
| Warm-cache NetCDF read -> checks/mask/weights -> packing -> upload if CUDA -> metrics | 8.8929 / 8.7880 | 9.9870 / 10.0710 |

**CUDA is slower for this small correctness case.** The workload has only 602 valid values; these measurements do not establish a GPU benefit. The actual warm-cache file pipeline also exposes the cost of reading/decoding and preprocessing. Context initialization, downloads and writing output files are excluded. Timings use 5 warmups and 20 repeats in fixed method blocks; this is not a cold-disk or order-balanced experiment.

![Real reference, manufactured forecast, errors, coverage and timing](results/rtx3050_weather_run02/weather.png)

See the [first NetCDF report](results/rtx3050_weather_run01/analysis.md) and [repeat report](results/rtx3050_weather_run02/analysis.md). Eighteen tests pass in the WSL environment, including the earlier precision tests and checks for known weighted answers, coordinate/time mismatch, units, duplicate coordinates, radians, irregular grids, missing values and NetCDF fill-value decoding. GitHub CPU CI installs the weather dependencies and exercises the committed NetCDF case without downloading data; PyTorch tests still skip when PyTorch is not installed.

```bash
python -m pip install -r requirements-weather.txt
python weather_verify.py --forecast data/weather_demo/manufactured_forecast.nc --reference data/weather_demo/reference.nc --backend both --output results/my_weather01
python scripts/analyze_weather.py results/my_weather01
```

For a CPU-only run, use `--backend numpy`. Outputs are `metrics.csv`, `timing_summary.csv`, `timing_samples.csv` and `environment.json`; the analysis script validates input hashes, coverage, backend agreement and raw timing summaries before generating `analysis.md` and `weather.png`. With other input files, pass their paths again using `analyze_weather.py --forecast ... --reference ...`.

Next: replace the manufactured forecast with a genuine forecast/reference pair for a matched variable, level, valid time and region. Then evaluate a larger domain and define an application error budget before revisiting FP16.

## Reproduce the reports and figures

```bash
python scripts/analyze_run.py results/rtx3050_run01
python scripts/compare_runs.py results/rtx3050_run01 results/rtx3050_run02
python plot_results.py results/rtx3050_run02/summary.csv --output results/rtx3050_run02/overview.png
python plot_results.py results/rtx3050_resident_run01/summary.csv --output results/rtx3050_resident_run01/overview.png
python scripts/analyze_precision.py results/rtx3050_precision_run01
python scripts/analyze_precision.py results/rtx3050_precision_run02
python scripts/analyze_weather.py results/rtx3050_weather_run01
python scripts/analyze_weather.py results/rtx3050_weather_run02
```

`analyze_run.py` is the original run01 report generator with run01-specific narrative. `compare_runs.py` compares the original five-method runs. `plot_results.py` supports CPU-only, original GPU, resident and historical CSVs.

## Historical results

The five original rows from the shared conversation remain unchanged in [legacy_rtx3050.csv](results/legacy_rtx3050.csv), with their [historical chart](results/legacy_benchmark.png). They lack raw repeat timings and full metadata. The reported maximum compute speedup is 13.73x at 20M. At 50M, H2D + compute implies an **estimated** 1.68x speedup, not directly measured E2E latency. The legacy memory field's measurement boundary is unknown. Do not merge these rows with the current protocol or treat them as independently reproduced results.

## Roadmap

1. **FP32 baseline: completed.** Repeated timings, raw samples, four-panel figures, hardware inventory and the resident-input experiment. Further repeat sessions and profiling remain useful for assessing stability.
2. **v3: first precision comparison completed.** Two sessions compare FP32, native FP16 and FP16 storage with FP32 arithmetic, including input/error decomposition, allocated memory and overflow tests. Next establish real-data accuracy requirements; investigate fused or chunked mixed arithmetic only if the workflow warrants it.
3. **v4: first NetCDF correctness case completed.** Real NCEP reference, explicitly manufactured forecast/missing cells, unit/coordinate checks, joint masks, area-weighted metrics, actual warm-file pipeline timing and CPU/CUDA agreement. Next obtain a genuine forecast/reference pair and validate its variable, level, units, valid times and grid before making skill claims. ERA5 can be a reanalysis reference, not automatically station truth.
4. **Weather AI ecosystem.** Explore Earth-2 / PhysicsNeMo after the real-data verification workflow is established. Consider profiling, pinned memory and compiled/fused kernels against that workflow.

Learning goal: explain how data movement, numerical precision and measurement boundaries affect a useful weather-verification pipeline.
