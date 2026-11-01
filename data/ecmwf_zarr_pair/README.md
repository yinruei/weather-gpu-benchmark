# ECMWF IFS ENS control from Zarr

Forecast source: **ECMWF IFS ENS control member 0**, distributed as Icechunk/Zarr by [dynamical.org](https://dynamical.org/catalog/ecmwf-ifs-ens-forecast-15-day-0-25-degree/). This is not IFS HRES, AIFS or an ensemble mean. ECMWF-derived data processed by dynamical.org from ECMWF Open Data; [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) and [ECMWF Terms of Use](https://www.ecmwf.int/en/forecasts/datasets/open-data) apply.

| Property | Selection |
|---|---|
| Archive variable | `temperature_2m`, instantaneous, degree_Celsius |
| Initialization | 2024-05-06 00 UTC |
| Member | 0, unperturbed ENS control |
| Leads | 24, 30, 36, 42 h |
| Valid times | 2024-05-07 00, 06, 12, 18 UTC |
| Grid and region | Native 0.25-degree grid, 18.25–26 N, 116–123.75 E |
| Selected shape | 4 x 32 x 32 = 4,096 time/grid values |
| Reference | Same-valid-time GFS f000 2-m analysis, copied exactly from the documented NOAA subset |
| Remote snapshot | `G76A041BF9PWGCNRBFVG` |

The reference comes from [the GFS source case](../gfs_taiwan_pair/README.md), retained in Kelvin. It is model-dependent analysis, not independent station truth. The domain includes land and ocean. The archive provider converts ECMWF GRIB to Celsius and rounds floating-point bits for compression, so these are provider-processed values, not bitwise native GRIB.

`forecast.zarr` and `reference.zarr` are consolidated **local Zarr v2** stores, directly read by the verification runner. No NetCDF conversion is required for this evaluation. Initialization time, lead hours, member and 2-m height remain explicit coordinates. Forecast Celsius and reference Kelvin are normalized before computing weighted metrics. No artificial errors, injected masks or regridding are introduced.

The region is intentionally aligned with one source spatial chunk. The remote chunk shape is `[1,85,51,32,32]`: selecting member 0 and four leads still involves a chunk containing other leads/members. The selected temperature payload is 16,384 logical bytes, while nominal uncompressed source chunk size is 17,756,160 bytes. These are storage-layout sizes, **not measured network traffic**. This illustrates why Zarr performance depends on chunk design and query shape.

The first observed remote metadata open and selected-data materialization took 4.207 s and 3.472 s. They are single read observations, not cold-cache throughput measurements. Remote access/download is separate from local CPU/CUDA benchmark timings.

## Evaluate offline

```bash
python -m pip install -r requirements-zarr.txt
python weather_verify.py --forecast data/ecmwf_zarr_pair/forecast.zarr --reference data/ecmwf_zarr_pair/reference.zarr --backend both --output results/my_ecmwf_zarr01
python scripts/analyze_weather.py results/my_ecmwf_zarr01 --forecast data/ecmwf_zarr_pair/forecast.zarr --reference data/ecmwf_zarr_pair/reference.zarr
```

Use `--backend numpy` without CUDA. Evaluation was tested with the existing WSL Python 3.10 / Zarr 2.18.3 environment; all 31 tests pass there. Local Zarr v2 metadata and chunks are covered by directory hashes. `.gitattributes` preserves their bytes across Git checkouts. Explicit zero-valued coordinate chunks prevent undefined missing-chunk behavior in older Zarr readers (notably initialization offsets and control member 0).

## Recreate from the remote archive

Remote acquisition uses Icechunk v2 and requires **Python >=3.12**. The existing Windows project Python 3.13 environment was used; the WSL CUDA Python does not need upgrading.

```bash
python -m pip install -r requirements-icechunk.txt
python scripts/prepare_ecmwf_zarr.py --snapshot-id G76A041BF9PWGCNRBFVG --output data/ecmwf_zarr_pair_recreated
```

The repository session is read-only. The script enforces exact initialization/member/lead selection, checks the archive's valid time, enforces a nominal uncompressed source-chunk budget, selects an exact matching GFS reference grid, and refuses an existing output path. [manifest.json](manifest.json) records the catalog/asset, snapshot, query, units, processing, source reference hash, versions and local directory hashes. Recreating this snapshot reproduced both directory hashes in the acquisition environment; future remote snapshot retention is controlled by the publisher.

The observed errors cannot be directly compared with the earlier 41 x 41 GFS case: this box is 32 x 32, and the forecast configurations/processing differ. GFS analysis can also favor its own model. This is a single-case data/format test, not a model ranking or a general skill estimate.
