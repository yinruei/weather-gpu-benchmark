"""Validate weather-verification outputs and plot fields, errors and timings."""
import argparse
from collections import defaultdict
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from weather_verify import METRICS, load_pair, numpy_weighted, pack, sha256


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--forecast', type=Path, default=Path('data/weather_demo/manufactured_forecast.nc'))
    p.add_argument('--reference', type=Path, default=Path('data/weather_demo/reference.nc'))
    p.add_argument('--manifest', type=Path, help='Optional provenance manifest; otherwise use the input directory')
    a = p.parse_args()
    env = json.loads((a.run/'environment.json').read_text(encoding='utf-8'))
    if sha256(a.forecast) != env['forecast_sha256'] or sha256(a.reference) != env['reference_sha256']:
        raise ValueError('Input hashes differ from the recorded experiment')
    f, r, mask, weights = load_pair(a.forecast, a.reference, env['variable'], env['reference_variable'], env['weighting'])
    manifest_path = a.manifest or a.forecast.parent/'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
    if manifest and (manifest.get('files', {}).get(a.forecast.name) != env['forecast_sha256'] or
                     manifest.get('files', {}).get(a.reference.name) != env['reference_sha256']):
        raise ValueError('Provenance manifest does not match the input file hashes')
    is_demo = bool(manifest.get('injected_missing_cells'))
    is_gfs = manifest.get('case_kind') == 'gfs_forecast_vs_f000'
    if is_gfs:
        initial = np.datetime64(datetime.fromisoformat(manifest['forecast_initialization_utc']).replace(tzinfo=None), 'ns')
        leads = np.array(manifest['forecast_lead_hours'])
        np.testing.assert_array_equal(f.forecast_reference_time.values, np.full(f.sizes['time'], initial))
        np.testing.assert_array_equal(f.lead_hours.values, leads)
        np.testing.assert_array_equal(initial+leads.astype('timedelta64[h]'), f.time.values)
        np.testing.assert_array_equal(r.forecast_reference_time.values, r.time.values)
        np.testing.assert_array_equal(r.lead_hours.values, np.zeros(r.sizes['time']))
        if float(f.height_above_ground.values) != 2. or float(r.height_above_ground.values) != 2.:
            raise ValueError('GFS pair must retain its shared 2-m height coordinate')
    metrics, timings, samples = [read_csv(a.run/name) for name in ['metrics.csv', 'timing_summary.csv', 'timing_samples.csv']]
    scopes = [('all_times', f.values, r.values, mask, weights)] + [
        (str(t), f.values[i], r.values[i], mask[i], weights[i]) for i, t in enumerate(f.time.values)]
    expected = {}
    for label, ff, rr, mm, ww in scopes:
        arrays = pack(ff, rr, mm, ww)
        ref = numpy_weighted(*(v.astype(np.float64) for v in arrays)) if len(arrays[0]) and arrays[2].sum() > 0 else None
        expected[label] = (ref, mm, ww)
    backends = ['reference_fp64', 'numpy_fp32'] + (['cuda_fp32'] if env['torch'] else [])
    keys = {(label, b) for label in expected for b in backends}
    if len(metrics) != len(keys) or {(row['scope'], row['backend']) for row in metrics} != keys:
        raise ValueError('Missing, extra or duplicate metric rows')
    for row in metrics:
        ref, mm, ww = expected[row['scope']]
        assert int(row['valid_points']) == int(mm.sum()) and int(row['total_points']) == mm.size
        np.testing.assert_allclose(float(row['valid_fraction']), mm.mean(), rtol=1e-12, atol=0)
        np.testing.assert_allclose(float(row['valid_weight_fraction']), ww[mm].sum(dtype=np.float64)/ww.sum(dtype=np.float64), rtol=1e-12, atol=0)
        if ref is None:
            assert all(not row[k] for k in METRICS)
        else:
            np.testing.assert_allclose([float(row[k]) for k in METRICS], ref, rtol=1e-5, atol=1e-5)
    groups = defaultdict(list)
    for row in samples:
        groups[row['method']].append(row)
    expected_methods = {'numpy_resident', 'numpy_warm_file_pipeline'}
    if env['torch']:
        expected_methods |= {'cuda_resident', 'cuda_warm_file_pipeline'}
    if set(groups) != expected_methods or len(timings) != len(expected_methods) or {r['method'] for r in timings} != expected_methods:
        raise ValueError('Missing, duplicate or unexpected timing methods')
    for row in timings:
        raw = groups[row['method']]
        assert sorted(int(s['repeat']) for s in raw) == list(range(env['repeats']))
        seconds = np.array([float(s['seconds']) for s in raw])
        assert np.isfinite(seconds).all() and (seconds > 0).all()
        for q, key in [(25, 'p25'), (50, 'median'), (75, 'p75')]:
            np.testing.assert_allclose(float(row[key+'_sec']), np.percentile(seconds, q), rtol=1e-12, atol=0)
        np.testing.assert_allclose([[float(s[k]) for k in METRICS] for s in raw],
                                   np.broadcast_to(expected['all_times'][0], (env['repeats'], 3)), rtol=1e-5, atol=1e-5)
    pooled = {row['backend']: row for row in metrics if row['scope'] == 'all_times'}
    if is_demo:
        case_description = ('NCEP reanalysis air temperature at sigma 0.995 is a real reference field. '
                            'The forecast perturbation and missing cells are deliberately manufactured. '
                            'This is a software-validation case, not a forecast-skill assessment, not ERA5 and not 2-m station observations.')
    elif is_gfs:
        case_description = (f"Real GFS 2-m temperature forecasts initialized {manifest['forecast_initialization_utc']}, "
                            f"at leads {manifest['forecast_lead_hours']} hours, paired with GFS f000 analysis/initialization fields "
                            'from the matching valid-time cycles. No perturbations, injected masks, interpolation or regridding. '
                            'The reference is from the same model system, not independent station observations. '
                            'These are forecast-minus-analysis errors for one case, not a general estimate of forecast skill.')
    else:
        case_description = 'User-supplied temperature fields. Scientific provenance and matching physical variable/vertical level must be established separately.'
    lines = [f'# NetCDF weather pipeline: {a.run.name}', '',
             f"Forecast role: **{env['forecast_role']}**.", '',
             f"Reference role: {env['reference_role']}.", '',
             case_description, '',
             f"Coverage: {int(mask.sum())}/{mask.size} joint finite values ({mask.mean():.2%}); "
             f"{f.sizes['time']} valid times, {f.sizes['lat']} x {f.sizes['lon']} grid. "
             f"{str(f.time.values[0])} to {str(f.time.values[-1])} UTC. "
             f"Latitude {float(f.lat.min()):g}..{float(f.lat.max()):g}, longitude {float(f.lon.min()):g}..{float(f.lon.max()):g} degrees.", '',
             'Normalization: supported temperature units -> K -> FP32; coordinates sorted and longitude wrapped to [-180,180). '
             'Valid time and spatial coordinates must match exactly. No regridding, inner join or init-time/lead-time inference. '
             f"Weighting: {env['weighting']}. The denominator includes only joint finite cells. "
             'All-times metrics pool valid weighted cells over the sampled times; RMSE is not an average of the per-time RMSE values. '
             'It is not a continuous-day average.', '',
             'Reference arithmetic uses FP64 on the same packed FP32 data and weights. All saved backend values and all timed '
             f"metric values agree at rtol=atol=1e-5; {len(samples)} positive finite timing samples and all summaries reconcile. "
             'NetCDF input hashes match. No timing samples were removed.', '',
             '## Pooled verification metrics', '',
             '| Backend | Bias K | MAE K | RMSE K | Valid fraction | Area-weighted valid fraction |',
             '|---|---:|---:|---:|---:|---:|']
    for backend in backends:
        row = pooled[backend]
        lines.append(f'| {backend} | ' + ' | '.join(f'{float(row[k]):.9f}' for k in METRICS) +
                     f" | {float(row['valid_fraction']):.2%} | {float(row['valid_weight_fraction']):.2%} |")
    if is_gfs:
        lines += ['', '## Valid-time errors against matching GFS analysis', '',
                  '| Valid time UTC | Forecast lead h | Bias K | MAE K | RMSE K |',
                  '|---|---:|---:|---:|---:|']
        for t, lead in zip(f.time.values, manifest['forecast_lead_hours']):
            row = next(row for row in metrics if row['scope'] == str(t) and row['backend'] == 'reference_fp64')
            lines.append(f"| {np.datetime_as_string(t, unit='m')} | {lead} | " +
                         ' | '.join(f'{float(row[k]):.6f}' for k in METRICS) + ' |')
    lines += ['', '## Timing boundaries', '',
              f"{env['warmup']} warmups and {env['repeats']} measured repetitions per method. Resident compute reuses packed "
              'joint-valid inputs and weights; CUDA includes scalar return. The file pipeline reopens both NetCDF files, '
              'decodes units/coordinates, checks alignment, builds the mask and weights, packs valid cells, '
              'allocates/uploads inputs for CUDA and returns metrics. OS/file caches are warm, not a cold-storage experiment. '
              'Context initialization, downloading and result-file output are excluded. The file-pipeline method is actually timed, '
              'not estimated by adding separately sampled components.', '',
              '| Method | Median ms | p25 ms | p75 ms |', '|---|---:|---:|---:|']
    for row in timings:
        lines.append(f"| {row['method']} | " + ' | '.join(f'{float(row[k+"_sec"])*1000:.4f}' for k in ['median', 'p25', 'p75']) + ' |')
    lines += ['', '## Interpretation and limits', '',
              '- This deliberately small input is a correctness case, not a GPU scaling benchmark. Inspect the observed runtimes; '
              'do not extrapolate them to large domains or claim GPU acceleration if CUDA is slower.',
              ('- Errors reflect manufactured perturbations and masks, not an operational forecast.' if is_demo else
               '- This case covers one forecast initialization and one valid date. Lead time and valid time change together; '
               'these four points cannot isolate a general lead-time dependence. The regional average includes both land and ocean. '
               'Same-system analysis is not independent station truth.' if is_gfs else
               '- Scientific interpretation requires matching variable meaning and vertical level, and a suitable reference.'),
              '- Cosine-latitude weights approximate relative cell areas on regular latitude/longitude grids. '
              'Curvilinear/projected grids and dateline-crossing subsets require additional grid/area handling.',
              '- Methods execute in blocks in a fixed order. Only these short sessions and this small region are represented; '
              'pre/post GPU snapshots do not describe load throughout the measurement.', '']
    if is_demo:
        manifest_link = Path(os.path.relpath(manifest_path, a.run)).as_posix()
        lines += [f'Source and preparation: [manifest]({manifest_link}), '
                  '[xarray example](https://docs.xarray.dev/en/stable/examples/area_weighted_temperature.html), '
                  '[NOAA PSL NCEP reanalysis](https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html). '
                  'Reanalysis data provided by NOAA PSL, Boulder, Colorado, USA.', '']
    elif is_gfs:
        manifest_link = Path(os.path.relpath(manifest_path, a.run)).as_posix()
        lines += [f'Source message URLs, byte ranges, SHA-256 hashes, initialization/lead/valid-time matching and decoder versions: [manifest]({manifest_link}). '
                  '[NOAA GFS open data](https://registry.opendata.aws/noaa-gfs-bdp-pds/), '
                  '[NCEP product documentation](https://www.nco.ncep.noaa.gov/pmb/products/gfs/).', '']
    else:
        lines += ['Input file identities and supplied roles are recorded in environment.json. Verify their scientific provenance before interpreting metrics.', '']
    lines += [
              '![Reference, forecast, errors, coverage and timing](weather.png)', '']
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    first_valid = next(i for i in range(f.sizes['time']) if mask[i].any())
    common = mask[first_valid]
    ref = np.where(common, r.values[first_valid]-273.15, np.nan)
    forecast = np.where(common, f.values[first_valid]-273.15, np.nan)
    error = np.where(common, f.values[first_valid]-r.values[first_valid], np.nan)
    bounds = (min(np.nanmin(ref), np.nanmin(forecast)), max(np.nanmax(ref), np.nanmax(forecast)))
    temp_cmap = plt.get_cmap('coolwarm').copy()
    temp_cmap.set_bad('dimgray')
    error_cmap = plt.get_cmap('RdBu_r').copy()
    error_cmap.set_bad('dimgray')
    forecast_label = ('Manufactured forecast' if is_demo else
                      f"GFS forecast +{manifest['forecast_lead_hours'][first_valid]}h" if is_gfs else 'Forecast')
    reference_label = 'GFS f000 analysis' if is_gfs else 'Reference'
    for ax, value, title in zip(axes[0], [ref, forecast, error], [reference_label+' (deg C)', forecast_label+' (deg C)', 'Forecast - reference (K)']):
        options = dict(cmap=error_cmap, vmin=-max(np.nanmax(np.abs(error)), 1e-6), vmax=max(np.nanmax(np.abs(error)), 1e-6)) if value is error else dict(cmap=temp_cmap, vmin=bounds[0], vmax=bounds[1])
        plot = ax.pcolormesh(f.lon.values, f.lat.values, value, shading='auto', **options)
        fig.colorbar(plot, ax=ax, shrink=.85)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Longitude (degrees east)')
        ax.set_ylabel('Latitude (degrees north)')
    per_time = [row for row in metrics if row['backend'] == 'reference_fp64' and row['scope'] != 'all_times']
    hours = [str(t)[5:16].replace('T', '\n') for t in f.time.values]
    if is_gfs:
        hours = [label+f'\n+{lead}h' for label, lead in zip(hours, manifest['forecast_lead_hours'])]
    for key in METRICS:
        axes[1, 0].plot(hours, [float(row[key]) if row[key] else np.nan for row in per_time], 'o-', label=key.upper())
    axes[1, 0].set_title('Forecast-minus-analysis metrics' if is_gfs else 'Weighted metrics at each valid time')
    axes[1, 0].set_xlabel('Valid time (UTC)')
    axes[1, 0].set_ylabel('Metric (K)')
    axes[1, 0].legend()
    order = ['numpy_resident', 'cuda_resident', 'numpy_warm_file_pipeline', 'cuda_warm_file_pipeline']
    selected = [next(row for row in timings if row['method'] == method) for method in order if method in groups]
    med = np.array([float(row['median_sec'])*1000 for row in selected])
    low = np.array([float(row['p25_sec'])*1000 for row in selected])
    high = np.array([float(row['p75_sec'])*1000 for row in selected])
    axes[1, 1].bar(np.arange(len(selected)), med, yerr=[med-low, high-med], capsize=3)
    labels = [row['method'].replace('_warm_file_pipeline', '\nfile pipeline').replace('_resident', '\nresident') for row in selected]
    axes[1, 1].set_xticks(np.arange(len(selected)), labels, fontsize=8)
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_ylim(top=max(high)*3)
    axes[1, 1].set_ylabel('Median runtime (ms, log scale)')
    axes[1, 1].set_title('Observed runtimes (bars: p25 to p75)')
    for i, value in enumerate(med):
        axes[1, 1].text(i, value*1.3, f'{value:.3f}', ha='center', fontsize=8)
    xx = np.arange(len(hours))
    axes[1, 2].bar(xx-.18, [float(row['valid_fraction'])*100 for row in per_time], width=.35, label='Valid cells')
    axes[1, 2].bar(xx+.18, [float(row['valid_weight_fraction'])*100 for row in per_time], width=.35, label='Valid area weight')
    axes[1, 2].set_xticks(xx, hours)
    axes[1, 2].set_ylim(0, 110)
    axes[1, 2].set_ylabel('Coverage (%)')
    axes[1, 2].set_title('Joint finite coverage' + (' (demo missing cells)' if is_demo else ''))
    axes[1, 2].legend(fontsize=8)
    for ax in axes[1]:
        ax.grid(axis='y', alpha=.2)
    title = ('NCEP sigma-0.995 NetCDF demo: real reference + manufactured forecast' if is_demo else
             'GFS 2-m temperature: genuine forecast vs matching f000 analysis' if is_gfs else 'NetCDF temperature verification')
    subtitle = ('Not a forecast-skill evaluation.' if is_demo else
                'Same-system analysis, not independent station truth.' if is_gfs else 'Verify input provenance before interpreting forecast skill.')
    display_time = np.datetime_as_string(f.time.values[first_valid], unit='m')
    fig.suptitle(title+'\nDisplayed valid time: '+display_time+' UTC; dark-gray cells excluded jointly. '+subtitle, fontsize=12)
    fig.tight_layout()
    fig.savefig(a.run/'weather.png', dpi=180)
    plt.close(fig)
    (a.run/'analysis.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'PASS: {len(metrics)} metric rows and {len(samples)} timing/metric samples validated; report and figure saved')


if __name__ == '__main__':
    main()
