"""Exact native frame selection and local Zarr-v2 read coverage diagnostics."""
import itertools
import json
from pathlib import Path
import re

import numpy as np
import xarray as xr

from weather_batch import validate_batch


READER_POLICIES = {
    'open_once': dict(reuse=True, selection='bounding'),
    'reopen': dict(reuse=False, selection='bounding'),
    'exact_once': dict(reuse=True, selection='exact'),
    'exact_reopen': dict(reuse=False, selection='exact'),
    'gather_once': dict(reuse=True, selection='gather'),
    'gather_reopen': dict(reuse=False, selection='gather'),
}


def frame_indices(shape, batch, variant):
    if isinstance(batch, bool) or not isinstance(batch, (int, np.integer)) or batch < 1 or variant not in [0, 1]:
        raise ValueError('Positive integer batch and A/B variant required')
    if len(shape) != 3 or any(n < 1 for n in shape) or batch*(variant+1) > int(np.prod(shape)):
        raise ValueError('Insufficient distinct native frames')
    return np.array(np.unravel_index(np.arange(batch*variant, batch*(variant+1)), shape)).T


def read_exact_from_dataset(ds, case, variant):
    """Vectorized paired isel: preserve flattened init/lead/member frame order."""
    validate_batch(ds)
    side = case['side']
    if side < 1 or side > min(ds.sizes['latitude'], ds.sizes['longitude']):
        raise ValueError('Native spatial subset required; no tiling')
    dimensions = ['init_time', 'lead_time', 'ensemble_member']
    indices = frame_indices(tuple(ds.sizes[d] for d in dimensions), case['batch'], variant)
    indexers = {d: xr.DataArray(indices[:, i], dims='frame') for i, d in enumerate(dimensions)}
    selected = ds.forecast.isel(**indexers, latitude=slice(-side, None), longitude=slice(side)).load()
    # Each temporal indexer shares the frame dimension; this is not a Cartesian product.
    selected = selected.transpose('frame', 'latitude', 'longitude')
    initial, lead, member = [selected[d].values for d in dimensions]
    frames = [dict(initialization=str(i), lead_hours=int(l/np.timedelta64(1, 'h')),
                   member=int(m), valid_time=str(i+l)) for i, l, m in zip(initial, lead, member)]
    coordinates = {d: selected[d].values.copy() for d in ['latitude', 'longitude']}
    identity = dict(frames=frames, latitude=[float(selected.latitude.min()), float(selected.latitude.max())],
                    longitude=[float(selected.longitude.min()), float(selected.longitude.max())],
                    source_units='degree_Celsius', transform='(Celsius - 20.0) / 20.0 in FP64, then FP32; fixed benchmark scaling, not trained-model normalization')
    return selected.values[:, None], identity, coordinates


def read_exact(data, case, variant, dataset=None):
    if dataset is not None:
        return read_exact_from_dataset(dataset, case, variant)
    with xr.open_zarr(data, chunks=None, consolidated=True) as ds:
        return read_exact_from_dataset(ds, case, variant)


def read_gather_from_dataset(ds, case, variant):
    """One minimal rectangular basic slice, then paired NumPy gather in frame order."""
    validate_batch(ds)
    side = case['side']
    if side < 1 or side > min(ds.sizes['latitude'], ds.sizes['longitude']):
        raise ValueError('Native spatial subset required; no tiling')
    dimensions = ['init_time', 'lead_time', 'ensemble_member']
    indices = frame_indices(tuple(ds.sizes[d] for d in dimensions), case['batch'], variant)
    starts, stops = indices.min(axis=0), indices.max(axis=0)+1
    selected = ds.forecast.isel(**{d: slice(int(a), int(b)) for d, a, b in zip(dimensions, starts, stops)},
        latitude=slice(-side, None), longitude=slice(side)).load()
    local = indices-starts
    # This advanced indexing copies only B frames; backend receives basic slices only.
    raw = selected.values[local[:, 0], local[:, 1], local[:, 2]][:, None]
    initial, lead, member = [ds[d].values[indices[:, i]] for i, d in enumerate(dimensions)]
    frames = [dict(initialization=str(i), lead_hours=int(l/np.timedelta64(1, 'h')),
                   member=int(m), valid_time=str(i+l)) for i, l, m in zip(initial, lead, member)]
    coordinates = {d: selected[d].values.copy() for d in ['latitude', 'longitude']}
    identity = dict(frames=frames, latitude=[float(selected.latitude.min()), float(selected.latitude.max())],
                    longitude=[float(selected.longitude.min()), float(selected.longitude.max())],
                    source_units='degree_Celsius', transform='(Celsius - 20.0) / 20.0 in FP64, then FP32; fixed benchmark scaling, not trained-model normalization')
    return raw, identity, coordinates


def read_gather(data, case, variant, dataset=None):
    if dataset is not None:
        return read_gather_from_dataset(dataset, case, variant)
    with xr.open_zarr(data, chunks=None, consolidated=True) as ds:
        return read_gather_from_dataset(ds, case, variant)


def read_coverage(data, case, variant, selection):
    """Logical materialization and unique stored chunks, not physical disk I/O."""
    data = Path(data)
    meta = json.loads((data/'forecast/.zarray').read_text())
    if meta['zarr_format'] != 2 or meta.get('dimension_separator', '.') != '.':
        raise ValueError('Coverage diagnostic requires local dot-separated Zarr v2')
    shape, chunks = meta['shape'], meta['chunks']
    batch, side = case['batch'], case['side']
    indices = frame_indices(tuple(shape[:3]), batch, variant)
    if side < 1 or side > min(shape[-2:]):
        raise ValueError('Native spatial subset required')
    if selection == 'exact':
        selected_frames = indices
    elif selection == 'gather':
        starts, stops = indices.min(axis=0), indices.max(axis=0)+1
        selected_frames = np.array(list(itertools.product(*(range(int(a), int(b)) for a, b in zip(starts, stops)))))
    elif selection == 'bounding':
        count = batch*(variant+1)
        ni, nl, nm = shape[:3]
        runs = (count+nl*nm-1)//(nl*nm)
        leads = min(nl, (count+nm-1)//nm) if runs == 1 else nl
        members = min(nm, count)
        selected_frames = np.array(list(itertools.product(range(runs), range(leads), range(members))))
    else:
        raise ValueError('Unknown frame selection')
    temporal = {tuple(row//np.array(chunks[:3])) for row in selected_frames}
    lat_chunks = range((shape[-2]-side)//chunks[-2], (shape[-2]-1)//chunks[-2]+1)
    lon_chunks = range((side-1)//chunks[-1]+1)
    keys = sorted('forecast/'+'.'.join(map(str, (*t, a, b))) for t in temporal for a in lat_chunks for b in lon_chunks)
    if any(not (data/k).is_file() for k in keys):
        raise ValueError('Coverage requires explicit stored forecast chunks')
    dtype = np.dtype(meta['dtype'])
    result = dict(selection=selection, requested_frames=batch, materialized_frames=len(selected_frames),
                requested_raw_bytes=batch*side*side*dtype.itemsize,
                materialized_raw_bytes=len(selected_frames)*side*side*dtype.itemsize,
                packed_input_bytes=batch*side*side*4, chunk_shape=chunks,
                chunk_keys=keys, chunks_touched=len(keys),
                stored_chunk_bytes=sum((data/k).stat().st_size for k in keys),
                nominal_chunk_uncompressed_bytes=len(keys)*int(np.prod(chunks))*dtype.itemsize,
                frame_indices=indices.tolist())
    if selection == 'gather':
        result.update(slice_starts=starts.tolist(), slice_stops=stops.tolist(),
            returned_backing_bytes=batch*side*side*dtype.itemsize,
            slice_plus_gather_bytes=(len(selected_frames)+batch)*side*side*dtype.itemsize)
    return result


def probe_reads(data, case, variant, selection):
    """Separate instrumented Store get probe; never used for performance timing.

    Counts store-returned compressed bytes, not OS/disk/network traffic or decode calls.
    All successful reads are retained, including coordinate/metadata accesses.
    """
    from threading import Lock
    from physicsnemo_graph_pipeline import read_variant
    class ReadLog:
        def record(self, key, value, byte_range):
            with self.lock:
                self.records.append(dict(key=key, bytes=0 if value is None else len(value),
                                         found=value is not None, partial=byte_range is not None))
    try:
        from zarr.storage import LocalStore
    except ImportError:
        # The original Python 3.10 environment retains Zarr 2; do not upgrade it.
        from zarr.storage import DirectoryStore
        class RecordingStore(ReadLog, DirectoryStore):
            def __init__(self, root):
                super().__init__(root)
                self.records, self.lock = [], Lock()

            def __getitem__(self, key):
                try:
                    value = super().__getitem__(key)
                except KeyError:
                    self.record(key, None, None)
                    raise
                self.record(key, value, None)
                return value
    else:
        class RecordingStore(ReadLog, LocalStore):
            def __init__(self, root, *, read_only=True):
                super().__init__(root, read_only=read_only)
                self.records, self.lock = [], Lock()

            async def get(self, key, prototype=None, byte_range=None):
                value = await super().get(key, prototype, byte_range)
                self.record(key, value, byte_range)
                return value

            def get_sync(self, key, *, prototype=None, byte_range=None):
                value = super().get_sync(key, prototype=prototype, byte_range=byte_range)
                self.record(key, value, byte_range)
                return value

            async def get_partial_values(self, prototype, key_ranges):
                return [await self.get(key, prototype, interval) for key, interval in key_ranges]

    store = RecordingStore(data)
    with xr.open_zarr(store, chunks=None, consolidated=True) as dataset:
        raw, identity, coordinates = read_variant(data, case, variant, dataset, selection=selection)
    records = list(store.records)
    forecast = [r for r in records if re.fullmatch(r'forecast/\d+(?:\.\d+){4}', r['key'])]
    coverage = read_coverage(data, case, variant, selection)
    if sorted(r['key'] for r in forecast) != coverage['chunk_keys'] or any(not r['found'] or r['partial'] for r in forecast) or sum(r['bytes'] for r in forecast) != coverage['stored_chunk_bytes']:
        raise ValueError('Actual store gets differ from predicted unique full-chunk coverage')
    backing = raw
    while isinstance(backing.base, np.ndarray):
        backing = backing.base
    return dict(records=records, forecast_gets=len(forecast), forecast_returned_bytes=sum(r['bytes'] for r in forecast),
                raw_bytes=raw.nbytes, backing_array_bytes=backing.nbytes, identity=identity,
                coordinates={k: v.tolist() for k, v in coordinates.items()})
