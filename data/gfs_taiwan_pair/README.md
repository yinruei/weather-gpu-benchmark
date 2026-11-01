# Genuine GFS forecast / analysis pair near Taiwan

This case contains **real operational forecast values and real analysis/initialization values**, with no manufactured perturbations, injected missing cells, interpolation or regridding.

Source: NOAA NCEP GFS 0.25-degree `pgrb2` products through the [NOAA open-data archive](https://registry.opendata.aws/noaa-gfs-bdp-pds/). See [NCEP product documentation](https://www.nco.ncep.noaa.gov/pmb/products/gfs/).

| Property | Forecast | Reference |
|---|---|---|
| Parameter | TMP at 2 m above ground, K | Same |
| Initialization UTC | 2024-05-06 00:00 | Each matching valid-time cycle |
| Lead hours | 24 / 30 / 36 / 42 | 0 / 0 / 0 / 0 |
| Valid times UTC | 2024-05-07 00 / 06 / 12 / 18 | Same |
| Grid | Native regular 0.25-degree latitude/longitude | Same |
| Subset | 18–28 N, 115–125 E; 41 x 41 cells per time | Same |

`forecast.nc` and `reference.nc` store the selected decoded temperatures as FP32 `air(time,lat,lon)`. Auxiliary coordinates preserve initialization time, lead hours and 2-m height. All 6,724 selected time/grid values are finite in this case.

The reference is the **GFS f000 analysis/initialization-time field** (the source index labels it `anl`). It is part of the same model/assimilation system. It is **not independent station truth**. The domain includes ocean and surrounding land, not just Taiwan's land area. Scores describe one historical forecast cycle/day and this region; they do not establish overall GFS skill. Valid time and forecast lead change together, so four points do not isolate a general lead-time dependence.

[manifest.json](manifest.json) records all eight original message URLs, matching initialization/lead/valid times, selected index lines, HTTP byte ranges, message/index SHA-256 hashes, response metadata, decoder versions and hashes of these prepared NetCDF files. Only the TMP/2-m messages were downloaded (4,064,456 GRIB bytes plus the small index files), not eight full global GRIB files.

Recreate in a fresh directory:

```bash
python -m pip install -r requirements-gfs.txt
python scripts/prepare_gfs_pair.py --output data/gfs_taiwan_pair_recreated
```

The acquisition step needs internet access and ecCodes/cfgrib. Cached message hashes and framing are checked on reuse. `requirements-weather.txt` is sufficient to evaluate the small committed NetCDF files offline; it does not require a GRIB decoder.

Evaluate with the existing WSL CUDA environment:

```bash
python weather_verify.py --forecast data/gfs_taiwan_pair/forecast.nc --reference data/gfs_taiwan_pair/reference.nc --backend both --output results/my_gfs_pair01
python scripts/analyze_weather.py results/my_gfs_pair01 --forecast data/gfs_taiwan_pair/forecast.nc --reference data/gfs_taiwan_pair/reference.nc
```

Use `--backend numpy` without CUDA. The analysis checks prepared-file hashes against the adjacent manifest, checks forecast initialization + lead = valid time, checks f000 reference time = valid time, and checks the shared 2-m height before interpreting this as a GFS pair.
