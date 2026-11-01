# Phase 5B: local Zarr-to-FNO-to-file performance

The same fixed, untrained 132,353-parameter FNO and FP32 input transformation are retained from phase 5A. No forecast-quality evaluation, training or Earth2Studio pipeline is involved.

## Primary complete-workflow timings

| Session | Batch/grid | CPU P50 ms | CUDA P50 ms | CPU/CUDA | CPU P95 ms | CUDA P95 ms | CUDA frames/s |
|---|---|---:|---:|---:|---:|---:|---:|
| rtx3050_physics_pipeline_b16_check | 16/64x64 | 40.559 | 27.259 | 1.49x | 45.490 | 31.200 | 587.0 |

For the final listed case B16/64x64, observed complete-workflow CPU/CUDA ratios range from 1.49x to 1.49x. Ratios below 1 elsewhere are retained, including any nonmonotonic batch behavior. These results do not justify assuming every batch benefits from CUDA.

Primary wall time is measured directly through file close. It includes Zarr open/read/decode and source coordinate validation, fixed scaling and contiguous FP32 packing, frame/coordinate output metadata construction, host tensor creation, CUDA upload, model forward, full CPU return and uncompressed NPZ creation/write/close. CPU and CUDA models/weights stay resident; each invocation reopens and rereads the source. File readback, reference checks, hashing, cleanup and benchmark CSV/JSON writing are excluded.

Primary repetitions are separate from stage-profile repetitions. CUDA completion is guaranteed by the blocking full host copy before writing. Profiles additionally synchronize GPU stages and expose instrumentation cost; their mean segments plus residual sum to the profile mean, not the primary P50. Stage medians are never added to estimate total latency.

Profile means can differ materially from primary timings because synchronization boundaries and execution time/system state differ. In particular, GPU forward wall times in this workflow may be longer than the earlier forward-only benchmark. The current samples do not isolate driver scheduling, power state, shape history or other causes. Stage shares describe the synchronized profile experiment, not an exact decomposition of primary latency or proof of a sole bottleneck.

## Independent stage profiles

| Session | Batch/grid | Backend | Read/decode/check ms | Prepare ms | Upload ms | Forward ms | Return ms | Write/close ms | Other ms | Profile mean ms |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rtx3050_physics_pipeline_b16_check | 16/64x64 | cpu | 22.009 | 0.254 | 0.000 | 31.082 | 0.017 | 0.931 | 0.047 | 54.341 |
| rtx3050_physics_pipeline_b16_check | 16/64x64 | cuda | 24.405 | 0.247 | 0.473 | 3.440 | 0.290 | 1.126 | 0.048 | 30.029 |

## Output/source correctness and interpretation

Every first, warmup, primary, profile and memory-probe invocation writes a real NPZ file, reads it back, verifies exact array values/dtypes/coordinates/initialization/lead/member/valid-time metadata and compares the full model output elementwise with the archived CPU FP64 reference (rtol=1e-4, atol=1e-5). Each session independently recomputes that FP64 reference using the archived FP32 weights cast to double. Numerical differences describe arithmetic consistency, not weather error. Output variable is dimensionless model_output, not a temperature forecast.

Only the last primary output per case/backend is retained. Deterministic payload/file hashes, output byte counts and numerical checks are recorded for all primary/profile/first observations; discarded trial files are removed after validation outside timing. The CPU-only analyzer checks retained files against original frames, archived weights/reference and every recorded trial hash, then independently reconciles raw quantiles, method ordering and stage sums. CPU/CUDA write identical byte counts with uncompressed np.savez.

The source is the pinned local Zarr, not a fresh remote download. Cases follow listed order; primary CPU/CUDA order is seeded and shuffled per repetition, profiles use a separate seed. Reads occur after source hash/reference setup and may use OS/library cache; first_observed.csv is an initialized-process observation, not cold storage/process timing. No cache is dropped. Validation readback and trial deletion also affect cache/system state outside timing. The range shading covers session medians and is not a confidence interval. Short-session P95 values are estimates.

Write completion means file close at the application/OS level; fsync and durable media persistence are not requested. Compression, metadata-heavy Zarr output, asynchronous overlap, remote acquisition, model/process startup and training are separate workloads. Setup duration is recorded outside primary timing; GPU allocator memory.csv is a separate warmed full-invocation observation and is not total process VRAM.

The earlier forward/upload/return curve is context from an earlier measurement session, not a contemporaneous decomposition of current end-to-end times. Compare current CPU/CUDA within each session for the complete-workflow speedup.

## Read coverage

| Batch/grid | Packed input bytes | Materialized frames | Selection bytes before truncation | Touched local chunks | Nominal uncompressed chunk bytes |
|---|---:|---:|---:|---:|---:|
| 16/64x64 | 262144 | 16 | 262144 | 4 | 3342336 |

Selection matches phase 5A: native rectangular init/lead/member selection, then first B flattened frames. B64 materializes 102 frames before truncation to 64. Chunk estimates use the local v2 chunk layout; they are neither measured physical disk bytes nor remote-archive transfer sizes. This baseline deliberately preserves that selection behavior so later reader changes can be assessed separately.

## Reproduce

```bash
source .venv/physicsnemo/bin/activate
python physicsnemo_pipeline.py --output results/my_physics_pipeline01
python physicsnemo_pipeline.py --order-seed 20261103 --output results/my_physics_pipeline02
python scripts/analyze_physicsnemo_pipeline.py results/my_physics_pipeline01 results/my_physics_pipeline02 --output results/my_physics_pipeline_comparison
```

Use new output paths. The existing phase-5 isolated environment and exact lock apply; CPU-only dependencies suffice to revalidate saved artifacts and regenerate this report. No JAX installation is required.

![Complete local workflow, speedup and separate stage profiles](physicsnemo_pipeline.png)
