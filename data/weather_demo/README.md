# One-day NetCDF verification demo

Real reference: the NCEP reanalysis air-temperature sample used in the [xarray area-weighting example](https://docs.xarray.dev/en/stable/examples/area_weighted_temperature.html), from the pinned source recorded in [manifest.json](manifest.json).

Data provided by NOAA PSL, Boulder, Colorado, USA. See the [NCEP reanalysis source documentation](https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html).

- Date: 2013-01-01, 00/06/12/18 UTC; four samples, not a continuous-day average.
- Region: 30–50 N, 230–270 E (equivalent to 130–90 W), regular 2.5-degree grid; 9 x 17 cells per time.
- Variable: air temperature at **sigma level 0.995**, in kelvin. It is not 2-m temperature, ERA5 or station observations.
- `reference.nc`: selected decoded temperatures adopted as FP32, with five deliberately missing values.
- `manufactured_forecast.nc`: reference temperatures plus a deterministic spatial/time perturbation, with five different deliberately missing values.
- `manifest.json`: pinned source URL/hash, original global metadata, selection, perturbation/mask definitions and hashes of the two prepared NetCDF files.

There are 612 time/grid values and 602 joint finite values. Missing cells in these prepared files were injected for software testing; they are not evidence of missingness in the original reanalysis.

The source global title contains the historical string “1948”; the selected **decoded valid-time coordinates are in 2013**, as recorded in the manifest and NetCDF files.

This case tests file decoding, units, coordinates, joint masking, weighting and CPU/CUDA numerical agreement. **The manufactured forecast is not an operational model forecast, so its metrics do not measure forecast skill.**

Recreate in a new directory (downloads the fixed source NetCDF and verifies its SHA-256):

```bash
python -m pip install -r requirements-weather.txt
python scripts/prepare_weather_demo.py --output data/weather_demo_recreated
```

For offline recreation, add `--source /path/to/air_temperature.nc`. The cached file must match the pinned source checksum. No network download is needed to use the small committed files or run the tests.
