# RTX 3050 GPU-resident experiment

Source: environment.json, samples.csv and summary.csv in this directory, copied unchanged from the WSL experiment.

Executed through the existing WSL Python environment and RTX 3050. FP32 synthetic data, seed 42, 5 warmups, 20 measured repeats per size/method, one PyTorch CPU thread.

Validation passed: all 360 positive finite timing samples, group membership, repeat IDs, medians, quartiles and four speedup fields reconcile. Saved resident GPU Bias/MAE/RMSE agree with the NumPy references at rtol=atol=1e-5. No samples were removed.

| Points | NumPy ms | Reupload E2E ms | Resident + scalar return ms | Resident IQR/median | Speedup vs reupload | Speedup vs NumPy | Initial upload incl. allocation ms (single sample) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,000,000 | 1.223 | 1.572 | 0.386 | 3.67% | 4.07x | 3.16x | 3.778 |
| 5,000,000 | 12.483 | 6.921 | 1.152 | 1.50% | 6.01x | 10.84x | 7.659 |
| 10,000,000 | 28.047 | 14.086 | 2.194 | 2.31% | 6.42x | 12.78x | 15.521 |

## Measurement boundaries

- Resident: inputs already on GPU, compute all three metrics and return three Python scalars on every iteration. Synchronization brackets wall-clock samples.
- Reupload E2E: same calculation and scalar return, but copy both pageable host inputs into existing device buffers on every iteration.
- Initial upload: two host-to-device copies into new GPU tensors, including input allocation. GPU context initialization is excluded. Only one observation per size; no warmup/repeat distribution applies to this field.
- Data generation, file I/O, input checks and correctness checks are outside the measured regions. Temporary reduction allocations remain inside compute timing.

## Interpretation

- At 10M points the resident workflow takes 2.194 ms versus 14.086 ms with repeated uploads, a 6.42x ratio of medians. Relative to this session's NumPy baseline the ratio is 12.78x.
- Avoiding repeated input transfer explains the main difference in the measured workflow boundaries. Fixed method ordering and unrecorded system-load changes can also affect timings; a single session does not establish a universal speedup.
- The same two synthetic inputs are reused across all iterations. This illustrates reusable input data or GPU-produced model outputs, not changing CPU inputs. Compute-only timing already used resident inputs in earlier experiments; this new path additionally includes scalar return.
- Initial upload is excluded from resident steady-state speedups. A one-off CPU-origin workload still pays that cost. No combined upload-plus-resident batch total was measured, so do not present sums of these observations as measured pipeline latency.
- IQR/median describes spread within this session and is not a confidence interval.

## Environment observation after the experiment

Post-run nvidia-smi reported NVIDIA GeForce RTX 3050, driver 560.94, 39 C, 10.17 W and 10% utilization. This snapshot was collected after completion; it does not describe load or power during the timing samples. The result metadata records software versions. Exact CPU model and pre-run load were not captured.

## Repeat experiment

```bash
source ~/venvs/weather-gpu/bin/activate
cd ~/weather-gpu-benchmark-github
python benchmark.py --resident --sizes 1000000 5000000 10000000 --repeats 20 --warmup 5 --output results/rtx3050_resident_run02
```

Use a fresh output directory. Record CPU, driver and load before running. A later experiment should update or generate inputs on GPU to represent the intended production workflow.
