# Genuine ECMWF ensemble batch and GFS analysis Zarr

This checked-in local Zarr v2 subset supports offline CPU/CUDA batch verification. It contains **6,684,672 actual forecast values**, not copies of the earlier small fixture. The local store is approximately 6.8 MB compressed (decimal).

| Selection | Value |
|---|---|
| Forecast | ECMWF IFS ENS, members 0..50 individually; not HRES or AIFS |
| Initializations | 2024-05-06, 07, 08, 09 at 00 UTC |
| Leads | 24, 30, 36, 42, 48, 54, 60, 66 hours |
| Native grid | 0.25 degrees, 64 x 64 cells |
| Latitude / longitude | 18.25..34 N / 116..131.75 E |
| Reference | GFS f000 2-m analysis at each exact valid time; 20 unique cycles |
| Local units | Both fields remain degree_Celsius; evaluator normalizes to FP32 Kelvin |
| Forecast shape | init_time x lead_time x ensemble_member x latitude x longitude |
| Reference shape | init_time x lead_time x latitude x longitude; shared across members |

Sources are [ECMWF IFS ENS via dynamical.org](https://dynamical.org/catalog/ecmwf-ifs-ens-forecast-15-day-0-25-degree/) and [NOAA GFS via dynamical.org](https://dynamical.org/catalog/noaa-gfs-forecast/). Attribution: ECMWF IFS ENS forecast and NOAA NWS NCEP GFS data processed by dynamical.org from their open-data archives. Archives are licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); [ECMWF Terms of Use](https://www.ecmwf.int/en/forecasts/datasets/open-data) also apply to ECMWF-derived data. Provider processing converts GRIB temperatures to Celsius and rounds floating-point bits, so these values are not bitwise native GRIB.

Both upstream sources are read through **read-only Icechunk snapshots**, pinned in [manifest.json](manifest.json). ECMWF snapshot `G76A041BF9PWGCNRBFVG`; GFS snapshot `7FB36MTVTBHMA65ETQB0`. The acquisition validates valid time = initialization + lead, reference initialization = valid time, reference lead = 0, exact coordinate equality and finite historical values. There is no interpolation, tiling, artificial forecast noise or injected missing data. Multiple forecasts may share a reference time; initialization/lead identity is retained rather than dropping duplicate valid times.

The ECMWF selection touches 16 source chunks (284,098,560 nominal uncompressed bytes). GFS touches 80 chunks (491,937,600 nominal uncompressed bytes); combined 740.1 MiB. These estimates describe source chunk layout, **not measured HTTP traffic**. Reading selected values took 12.674 s (ECMWF) and 33.822 s (GFS), separately from metadata opening. These are single acquisition observations, not network throughput benchmarks. Remote transfer is excluded from local CPU/GPU timings.

Local forecast chunks are `(1,4,51,32,32)` and reference chunks `(1,4,32,32)`, with consolidated Zarr v2 metadata. Zero-valued integer/time coordinate chunks are explicitly stored for legacy Zarr 2 compatibility. `.gitattributes` preserves metadata/chunk bytes; the manifest directory hash uses the same sorted path/digest algorithm as the earlier Zarr case. This store has been read directly in the existing WSL Python 3.10/Zarr 2.18.3 CUDA environment and the Windows acquisition environment.

```bash
# Existing WSL CUDA environment; no remote download required
source ~/venvs/weather-gpu/bin/activate
python -m pip install -r requirements-zarr.txt
python weather_batch.py --output results/my_batch01 --group-loop
python weather_batch.py --output results/my_batch02 --group-loop --seed 20261102
python scripts/analyze_weather_batch.py results/my_batch01 results/my_batch02 --output results/my_batch_comparison
```

Remote recreation needs Python >=3.12 and `requirements-icechunk.txt`:

```bash
python -m pip install -r requirements-icechunk.txt
python scripts/prepare_ecmwf_batch.py --output data/ecmwf_batch_recreated
```

The script refuses an existing output path and enforces a 1,024 MiB nominal uncompressed source-chunk budget before materializing either temperature selection. Use `--data data/ecmwf_batch_recreated/batch.zarr` with the benchmark/analyzer for that output. Publisher snapshot retention is external to this repository.

The scores are per-initialization/lead/member forecast-minus-GFS-analysis errors. GFS analysis is model-dependent, not independent truth, and the box includes land and ocean. Ensemble members and overlapping valid times are correlated. Pooled individual-member errors are not metrics of an ensemble-mean forecast. This short four-initialization case does not rank models or establish general forecast skill.
