#!/usr/bin/env python
"""Physics Evaluation Runner for WeatherBench 2 Zarr Datasets.

Streams Model predictions and reference ground-truth data from WeatherBench 2 Zarr
buckets, computes all physics metrics at specified forecast horizons, and saves a
single long-format CSV.

Output Long-Format CSV Columns:
    date | lead_time_hours | metric_name | model_value | ref_value | n_levels | sp_method | ensemble_member

Ensemble / Probabilistic Model Support:
    Automatically detects extra ensemble dimensions ('ens', 'realization', 'member',
    'ensemble', 'number') in input datasets. Evaluates metrics for each ensemble
    member individually and labels output rows with the corresponding `ensemble_member`
    identifier (defaulting to 0 for deterministic models).

Usage:
    physeval-run --year 2022
    physeval-run --dates 2022-01-01 2022-01-02 --workers 4
    physeval-run --prediction-zarr <path_to_zarr> --output-dir ./results
"""

from __future__ import annotations

import argparse
import calendar
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import dask
import numpy as np
import pandas as pd
import xarray as xr

# Import physics metrics companion library
from physmetrics_weather.physics_metrics import (
    ENSEMBLE_DIM_NAMES,
    MSL_NAMES,
    PHI_NAMES,
    PRED_TD_NAMES,
    Q_NAMES,
    SP_NAMES,
    T2M_NAMES,
    T_NAMES,
    U_NAMES,
    V_NAMES,
    ZSFC_NAMES,
    _detect_ensemble_dim,
    _detect_level_dim,
    _detect_pred_td_dim,
    _find_effective_resolution,
    _find_var,
    compute_conservation_scalars,
    compute_drift_percentages,
    compute_drift_slope,
    compute_geostrophic_imbalance,
    compute_hydrostatic_imbalance,
    compute_ke_spectrum,
    compute_pure_tcwv,
    compute_q_spectrum,
    compute_spectral_scores,
    derive_surface_pressure,
    get_grid_cell_area,
)

# Suppress xarray timedelta decoding warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*prediction_timedelta.*")


# ============================================================================
# Configuration & Constants
# ============================================================================

DEFAULT_MODEL_ZARR: str = "gs://weatherbench2/datasets/aurora/2022-1440x721.zarr"
REF_ZARR: str = "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"

IFS_T0_ZARR: str = "gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr"
IFS_T0_LOWRES_ZARR: str = "gs://weatherbench2/datasets/hres_t0/2016-2022-6h-512x256_equiangular_conservative.zarr"

DEFAULT_OUTPUT_DIR: Path = Path.cwd() / "results"

LEAD_TIMES: List[Tuple[str, np.timedelta64]] = [
    ("12h", np.timedelta64(12, "h")),
    ("5d", np.timedelta64(120, "h")),
    ("10d", np.timedelta64(240, "h")),
]

DRIFT_WINDOW_END: Dict[int, np.timedelta64] = {
    12: np.timedelta64(24, "h"),
    120: np.timedelta64(120, "h"),
    240: np.timedelta64(240, "h"),
}

DEFAULT_WORKERS: int = 4


# ============================================================================
# Zarr I/O & Dataset Loading
# ============================================================================

def open_zarr_anonymous(url: str) -> xr.Dataset:
    """Open a public GCS Zarr store without authentication.

    Args:
        url: Storage URL or local file path to the Zarr store.

    Returns:
        xr.Dataset: Opened and cleaned xarray Dataset.
    """
    ds = xr.open_zarr(url, storage_options={"token": "anon"})
    rename = {}
    for v in ds.data_vars:
        if v != v.strip():
            rename[v] = v.strip()
    for d in ds.dims:
        if d != d.strip():
            rename[d] = d.strip()

    if "lat" in ds.dims and "latitude" not in ds.dims:
        rename["lat"] = "latitude"
    if "lon" in ds.dims and "longitude" not in ds.dims:
        rename["lon"] = "longitude"

    if rename:
        ds = ds.rename(rename)
    return ds


def load_static_fields(ds_ref: xr.Dataset) -> xr.Dataset:
    """Extract static fields (geopotential at surface, land-sea mask) at time=0.

    Args:
        ds_ref: Reference Dataset.

    Returns:
        xr.Dataset containing static fields.

    Raises:
        ValueError: If no static fields are found in the reference dataset.
    """
    static_vars = {}

    def _extract_static(ds: xr.Dataset, name: str) -> xr.DataArray:
        var = ds[name]
        if "time" in var.dims:
            var = var.isel(time=0, drop=True)
        return var

    for name in ("geopotential_at_surface", "z_sfc", "orography"):
        if name in ds_ref.data_vars:
            static_vars[name] = _extract_static(ds_ref, name)
            break

    for name in ("land_sea_mask", "lsm"):
        if name in ds_ref.data_vars:
            static_vars[name] = _extract_static(ds_ref, name)
            break

    if not static_vars:
        raise ValueError(
            f"No static fields found in reference dataset. Available: {list(ds_ref.data_vars)[:20]}"
        )

    return xr.Dataset(static_vars)


def _get_ps(
    ds: xr.Dataset,
    ds_static: xr.Dataset,
    level_dim: str = "level",
) -> xr.DataArray:
    """Extract or derive surface pressure array from dataset.

    Args:
        ds: Dataset containing surface pressure or mean sea level pressure.
        ds_static: Static dataset containing surface geopotential.
        level_dim: Pressure level dimension name.

    Returns:
        xr.DataArray: Surface pressure in Pascals.

    Raises:
        ValueError: If surface pressure cannot be extracted or derived.
    """
    sp_name = _find_var(ds, SP_NAMES)
    if sp_name is not None:
        sp = ds[sp_name]
        sp.attrs["derivation_method"] = "direct_sp"
        return sp

    if _find_var(ds, MSL_NAMES) is not None and _find_var(ds_static, ZSFC_NAMES) is not None:
        sp = derive_surface_pressure(ds, ds_static)
        sp.attrs["derivation_method"] = "hypsometric_msl_standard_atm"
        return sp

    raise ValueError(
        f"Cannot derive surface pressure: no SP variable and hypsometric MSL derivation failed. "
        f"Available: {list(ds.data_vars)}"
    )


# ============================================================================
# Grid Alignment & Date Handling
# ============================================================================

def _grids_match(
    ds_a: xr.Dataset,
    ds_b: xr.Dataset,
    lat_name: str = "latitude",
    lon_name: str = "longitude",
    atol: float = 1e-3,
) -> bool:
    """Check if two datasets share identical latitude and longitude grids.

    Args:
        ds_a: First dataset.
        ds_b: Second dataset.
        lat_name: Latitude dimension name.
        lon_name: Longitude dimension name.
        atol: Absolute tolerance for coordinate comparison.

    Returns:
        True if grids match within tolerance, False otherwise.
    """
    if ds_a.sizes.get(lat_name, 0) != ds_b.sizes.get(lat_name, 0):
        return False
    if ds_a.sizes.get(lon_name, 0) != ds_b.sizes.get(lon_name, 0):
        return False

    lat_a = np.sort(ds_a[lat_name].values)
    lat_b = np.sort(ds_b[lat_name].values)
    if not np.allclose(lat_a, lat_b, atol=atol):
        return False

    lon_a = np.sort(ds_a[lon_name].values)
    lon_b = np.sort(ds_b[lon_name].values)
    if not np.allclose(lon_a, lon_b, atol=atol):
        return False

    return True


def _align_ref_to_model(
    ds_ref: xr.Dataset,
    ds_model: xr.Dataset,
    lat_name: str = "latitude",
    lon_name: str = "longitude",
) -> xr.Dataset:
    """Align reference grid to match model grid for spectral evaluation.

    Args:
        ds_ref: Reference Dataset.
        ds_model: Model Dataset.
        lat_name: Latitude dimension name.
        lon_name: Longitude dimension name.

    Returns:
        xr.Dataset: Aligned Reference dataset.

    Raises:
        ValueError: If grid size mismatch exceeds 1 row.
    """
    n_ref = ds_ref.sizes.get(lat_name, 0)
    n_model = ds_model.sizes.get(lat_name, 0)
    n_lon_ref = ds_ref.sizes.get(lon_name, 0)
    n_lon_model = ds_model.sizes.get(lon_name, 0)

    if n_lon_ref != n_lon_model:
        raise ValueError(
            f"Longitude grid mismatch: Reference has {n_lon_ref}, Model has {n_lon_model}."
        )

    if n_ref == n_model:
        result = ds_ref
    elif n_ref == n_model + 1:
        lats = ds_ref[lat_name].values
        if lats[0] > lats[-1]:
            result = ds_ref.isel({lat_name: slice(0, -1)})
        else:
            result = ds_ref.isel({lat_name: slice(1, None)})
    else:
        raise ValueError(
            f"Latitude grid mismatch: Reference has {n_ref} rows, Model has {n_model}. "
            f"Only exact match or 1-row pole difference supported."
        )

    result = result.assign_coords({lat_name: ds_model[lat_name].values})

    if lat_name in result.dims and lon_name in result.dims:
        dims_list = list(result.dims)
        idx_lon = dims_list.index(lon_name)
        idx_lat = dims_list.index(lat_name)
        if idx_lon < idx_lat:
            dims_list[idx_lon], dims_list[idx_lat] = dims_list[idx_lat], dims_list[idx_lon]
            result = result.transpose(*dims_list)

    return result


def _resolve_dates(args: argparse.Namespace) -> List[str]:
    """Parse ISO date strings from command line arguments.

    Args:
        args: Parsed command-line arguments.

    Returns:
        List of ISO date strings (formatted YYYY-MM-DDTHH:MM:SS).
    """
    if args.dates:
        return [d if "T" in d else f"{d}T00:00:00" for d in args.dates]
    if args.month:
        year, month = args.month.split("-")
        n_days = calendar.monthrange(int(year), int(month))[1]
        return [f"{year}-{month}-{d:02d}T00:00:00" for d in range(1, n_days + 1)]

    year = args.year
    dates = []
    for m in range(1, 13):
        n_days = calendar.monthrange(year, m)[1]
        for d in range(1, n_days + 1):
            dates.append(f"{year}-{m:02d}-{d:02d}T00:00:00")
    return dates


def _parse_lead_times(spec: str) -> List[Tuple[str, np.timedelta64]]:
    """Parse comma-separated lead-time string into list of tuples.

    Args:
        spec: Comma-separated string like "12h,5d,10d".

    Returns:
        List of (label, timedelta) tuples.

    Raises:
        ValueError: If a lead time token cannot be parsed.
    """
    result = []
    for token in spec.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token.endswith("d"):
            days = int(token[:-1])
            td = np.timedelta64(days * 24, "h")
            result.append((token, td))
        elif token.endswith("h"):
            hours = int(token[:-1])
            td = np.timedelta64(hours, "h")
            label = f"{hours // 24}d" if (hours % 24 == 0 and hours >= 48) else f"{hours}h"
            result.append((label, td))
        else:
            raise ValueError(f"Cannot parse lead-time token: {token!r}")
    return result


# ============================================================================
# Single Slice Evaluation Workhorse
# ============================================================================

def _evaluate_one(
    model_zarr_path: str,
    ref_zarr_path: str,
    date_str: str,
    lead_label: str,
    lead_td: np.timedelta64,
    counter: int,
    total: int,
    mode: str,
    verbose: bool,
    static_zarr_path: Optional[str] = None,
    model_name: str = "model",
    extended_spectra: bool = False,
    sp_ablation: str = "default",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Fetch, preprocess, and evaluate physics metrics for one date and lead time slice.

    Ensemble Support:
        Detects ensemble dimensions ('ens', 'realization', 'member', 'ensemble', 'number')
        in the model slice and computes metrics for each member, assigning the corresponding
        `ensemble_member` ID (defaulting to 0 for deterministic data).

    Args:
        model_zarr_path: Path/URL to prediction Zarr.
        ref_zarr_path: Path/URL to reference Zarr.
        date_str: Forecast initialization date string.
        lead_label: Label for target lead time (e.g. "12h", "5d").
        lead_td: Target lead time timedelta.
        counter: Current task index in batch.
        total: Total number of tasks in batch.
        mode: Evaluation mode ('joint', 'ref', 'model').
        verbose: Print progress logs.
        static_zarr_path: Path to static fields Zarr.
        model_name: Name of evaluated model.
        extended_spectra: Compute extra 850hPa KE and Q spectra.
        sp_ablation: Surface pressure ablation study option.

    Returns:
        Tuple of (summary_rows, ts_rows, spectrum_rows, lr_dist_rows).
    """
    dask.config.set(scheduler="synchronous")

    def _log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    _log(f"  [{counter}/{total}] init={date_str} lead={lead_label} — Connecting to dataset...")

    ds_model_full = None
    if mode in ("joint", "prediction", "model"):
        ds_model_full = open_zarr_anonymous(model_zarr_path)

    ds_ref_full = open_zarr_anonymous(ref_zarr_path)

    ds_static_src = open_zarr_anonymous(static_zarr_path) if static_zarr_path else ds_ref_full
    ds_static = load_static_fields(ds_static_src)

    for var_name in list(ds_static.data_vars):
        v = ds_static[var_name]
        if "latitude" in v.dims and "longitude" in v.dims:
            if list(v.dims).index("longitude") < list(v.dims).index("latitude"):
                ds_static[var_name] = v.transpose("latitude", "longitude")

    z_sfc_name = _find_var(ds_static, ZSFC_NAMES)
    if z_sfc_name is None:
        raise ValueError(f"No surface geopotential in static dataset. Tried {ZSFC_NAMES}.")
    z_sfc = ds_static[z_sfc_name]

    area = get_grid_cell_area(ds_ref_full.isel(time=0, drop=True))
    lead_hours = int(lead_td / np.timedelta64(1, "h"))
    init_time = np.datetime64(date_str, "ns")
    valid_time = init_time + lead_td

    summary_rows: List[Dict[str, Any]] = []
    ts_rows: List[Dict[str, Any]] = []
    spectrum_rows: List[Dict[str, Any]] = []
    lr_dist_rows: List[Dict[str, Any]] = []

    _n_levels: Optional[int] = None
    _sp_method: str = "none"

    def _append_summary(
        metric_name: str,
        model_val: Any,
        ref_val: Any = None,
        ens_member: Any = 0,
    ) -> None:
        summary_rows.append({
            "date": date_str,
            "lead_time_hours": lead_hours,
            "metric_name": metric_name,
            "model_value": model_val,
            "ref_value": ref_val,
            "n_levels": _n_levels,
            "sp_method": _sp_method,
            "ensemble_member": ens_member,
        })

    try:
        ds_model_t = None
        ps_model = None
        area_model = area
        _lead_td_mismatch = False
        ens_dim = None
        ens_members: List[Any] = [0]

        if ds_model_full is not None:
            ds_model_t = ds_model_full.sel(time=init_time)
            pred_td_dim = _detect_pred_td_dim(ds_model_t)
            if pred_td_dim is not None and pred_td_dim in ds_model_t.dims:
                ds_model_t = ds_model_t.sel({pred_td_dim: lead_td}, method="nearest")
                actual_td = ds_model_t.coords.get(pred_td_dim)
                if actual_td is not None:
                    actual_td_val = actual_td.values
                    if isinstance(actual_td_val, np.timedelta64) and actual_td_val != lead_td:
                        _log(f"    [{counter}] Requested lead={lead_td}, nearest available={actual_td_val}")
                        _lead_td_mismatch = True

            ens_dim = _detect_ensemble_dim(ds_model_t)
            if ens_dim is not None and ens_dim in ds_model_t.dims:
                ens_members = list(ds_model_t[ens_dim].values)
                _log(f"    [{counter}] Detected ensemble dimension '{ens_dim}' with {len(ens_members)} members.")

            _NEEDED_VARS = set()
            for names in (T_NAMES, PHI_NAMES, U_NAMES, V_NAMES, Q_NAMES, MSL_NAMES, SP_NAMES, T2M_NAMES, ZSFC_NAMES):
                _NEEDED_VARS.update(names)
            drop_vars = [v for v in ds_model_t.data_vars if v.strip() not in _NEEDED_VARS]
            if drop_vars:
                ds_model_t = ds_model_t.drop_vars(drop_vars)

            if "time" in ds_model_t.dims:
                ds_model_t = ds_model_t.isel(time=0)

            if "latitude" in ds_model_t.dims and "longitude" in ds_model_t.dims:
                dims_list = list(ds_model_t.dims)
                idx_lon = dims_list.index("longitude")
                idx_lat = dims_list.index("latitude")
                if idx_lon < idx_lat:
                    dims_list[idx_lon], dims_list[idx_lat] = dims_list[idx_lat], dims_list[idx_lon]
                    ds_model_t = ds_model_t.transpose(*dims_list)

            ds_model_t = ds_model_t.load()
            model_grid_matches_ref = _grids_match(ds_model_t, ds_ref_full)

            ds_static_model = ds_static
            z_sfc_model = z_sfc
            if not model_grid_matches_ref:
                interp_coords = {
                    "latitude": ds_model_t["latitude"].values,
                    "longitude": ds_model_t["longitude"].values,
                }
                ds_static_model = ds_static.interp(interp_coords, method="nearest")
                z_sfc_name = _find_var(ds_static_model, ZSFC_NAMES)
                if z_sfc_name is not None:
                    z_sfc_model = ds_static_model[z_sfc_name]

            has_q = _find_var(ds_model_t, Q_NAMES) is not None
            model_level_dim = _detect_level_dim(ds_model_t)
            _has_sp = _find_var(ds_model_t, SP_NAMES) is not None
            _has_msl = _find_var(ds_model_t, MSL_NAMES) is not None
            _model_can_derive_sp = _has_sp or _has_msl
            _use_ref_sp = not _model_can_derive_sp

            if has_q and not _use_ref_sp:
                try:
                    ps_model = _get_ps(ds_model_t, ds_static_model, level_dim=model_level_dim)
                except (ValueError, KeyError, AttributeError) as exc:
                    _log(f"    [{counter}] Could not derive surface pressure: {exc}")
                    ps_model = None
            else:
                ps_model = None

            if model_level_dim in ds_model_t.dims:
                _n_levels = ds_model_t.sizes[model_level_dim]
            _sp_method = (
                ps_model.attrs.get("derivation_method", "unknown") if ps_model is not None else "none"
            )

            if "latitude" in ds_model_t.dims:
                n_model = ds_model_t.sizes["latitude"]
                n_area = area.sizes["latitude"]
                if not (n_model == n_area and model_grid_matches_ref):
                    area_model = get_grid_cell_area(ds_model_t)

        ds_ref_t = ds_ref_full.sel(time=valid_time)
        if "time" in ds_ref_t.dims:
            ds_ref_t = ds_ref_t.isel(time=0)
        ds_ref_t = ds_ref_t.load()

        ref_level_dim = _detect_level_dim(ds_ref_t)
        ps_ref = _get_ps(ds_ref_t, ds_static, level_dim=ref_level_dim)

        if _use_ref_sp and ps_ref is not None and has_q and ds_model_t is not None:
            if not model_grid_matches_ref:
                ps_model = ps_ref.interp(
                    latitude=ds_model_t.latitude, longitude=ds_model_t.longitude, method="linear"
                )
            else:
                ps_model = ps_ref
            ps_model.attrs["derivation_method"] = "ref_sp"
            _sp_method = "ref_sp"

    except Exception as exc:
        _log(f"    [{counter}] Data loading failed: {exc}")
        _append_summary("ERROR", None, None, ens_member=0)
        return summary_rows, ts_rows, spectrum_rows, lr_dist_rows

    # Evaluate per ensemble member
    if mode in ("joint", "prediction", "model") and ds_model_full is not None and not _lead_td_mismatch:
        try:
            td_start = np.timedelta64(12, "h")
            td_end = DRIFT_WINDOW_END.get(lead_hours, lead_td)

            ds_pred_init = ds_model_full.sel(time=init_time)
            pred_td_dim = _detect_pred_td_dim(ds_pred_init) or "prediction_timedelta"
            ds_pred_window = ds_pred_init.sel({pred_td_dim: slice(td_start, td_end)})

            avail_tds = ds_pred_window[pred_td_dim].values
            if len(avail_tds) >= 2:
                model_level_dim_d = _detect_level_dim(ds_pred_window)

                for m in ens_members:
                    ds_pred_m = ds_pred_window.sel({ens_dim: m}) if (ens_dim and ens_dim in ds_pred_window.dims) else ds_pred_window
                    hours_model, dry_vals, water_vals, energy_vals, pe_vals = [], [], [], [], []
                    hydro_vals, geo_vals = [], []

                    for td_val in avail_tds:
                        snap = ds_pred_m.sel({pred_td_dim: td_val}).load()
                        snap_valid_time = init_time + td_val

                        if "latitude" in snap.dims and "longitude" in snap.dims:
                            sdims = list(snap.dims)
                            si_lon = sdims.index("longitude")
                            si_lat = sdims.index("latitude")
                            if si_lon < si_lat:
                                sdims[si_lon], sdims[si_lat] = sdims[si_lat], sdims[si_lon]
                                snap = snap.transpose(*sdims)

                        try:
                            ps_snap = _get_ps(snap, ds_static_model, level_dim=model_level_dim_d) if not _use_ref_sp else ps_model
                            dry, water, energy = compute_conservation_scalars(
                                snap, ps_snap, area_model, z_sfc=z_sfc_model, level_dim=model_level_dim_d
                            )
                            step_sp_method = ps_snap.attrs.get("derivation_method", "unknown") if ps_snap is not None else "none"
                        except (ValueError, KeyError, AttributeError):
                            dry, water, energy = float("nan"), float("nan"), float("nan")
                            step_sp_method = "failed"

                        try:
                            hydro = float(compute_hydrostatic_imbalance(snap, area_model, level_dim=model_level_dim_d))
                        except (ValueError, KeyError, AttributeError):
                            hydro = float("nan")
                        try:
                            geo = float(compute_geostrophic_imbalance(snap, area_model, level_dim=model_level_dim_d))
                        except (ValueError, KeyError, AttributeError):
                            geo = float("nan")

                        h = float(td_val / np.timedelta64(1, "h"))
                        hours_model.append(h)
                        dry_vals.append(dry)
                        water_vals.append(water)
                        energy_vals.append(energy)
                        hydro_vals.append(hydro)
                        geo_vals.append(geo)

                        ts_rows.append({
                            "date": date_str,
                            "forecast_hour": h,
                            "dry_mass_Eg": dry,
                            "water_mass_kg": water,
                            "total_energy_J": energy,
                            "hydrostatic_rmse": hydro,
                            "geostrophic_rmse": geo,
                            "sp_method": step_sp_method,
                            "ensemble_member": m,
                        })

                    hours_model_arr = np.array(hours_model)
                    dry_vals_arr = np.array(dry_vals)
                    water_vals_arr = np.array(water_vals)
                    energy_vals_arr = np.array(energy_vals)

                    ref_hydro, ref_geo = None, None
                    if ds_ref_t is not None:
                        ref_ld = _detect_level_dim(ds_ref_t)
                        try:
                            ref_hydro = float(compute_hydrostatic_imbalance(ds_ref_t, area, level_dim=ref_ld))
                        except (ValueError, KeyError, AttributeError):
                            pass
                        try:
                            ref_geo = float(compute_geostrophic_imbalance(ds_ref_t, area, level_dim=ref_ld))
                        except (ValueError, KeyError, AttributeError):
                            pass

                    _append_summary("hydrostatic_rmse", hydro_vals[-1], ref_hydro, ens_member=m)
                    _append_summary("geostrophic_rmse", geo_vals[-1], ref_geo, ens_member=m)

                    slope_dry = compute_drift_slope(hours_model_arr, dry_vals_arr)
                    slope_water = compute_drift_slope(hours_model_arr, water_vals_arr)
                    slope_energy = compute_drift_slope(hours_model_arr, energy_vals_arr)
                    dry_ref = float(dry_vals_arr[0]) if len(dry_vals_arr) > 0 else 0.0

                    _append_summary(
                        "dry_mass_drift_pct_per_day",
                        (slope_dry / dry_ref * 100.0) if dry_ref != 0 and np.isfinite(slope_dry) else float("nan"),
                        ens_member=m,
                    )
                    _append_summary(
                        "water_mass_drift_pct_per_day",
                        (slope_water / water_vals_arr[0] * 100.0) if len(water_vals_arr) > 0 and water_vals_arr[0] != 0 and np.isfinite(slope_water) else float("nan"),
                        ens_member=m,
                    )
                    _append_summary(
                        "total_energy_drift_pct_per_day",
                        (slope_energy / energy_vals_arr[0] * 100.0) if len(energy_vals_arr) > 0 and energy_vals_arr[0] != 0 and np.isfinite(slope_energy) else float("nan"),
                        ens_member=m,
                    )
        except (ValueError, KeyError, AttributeError) as exc:
            _log(f"    [{counter}] Drift metrics failed: {exc}")

    # Spectral & Lapse Rate evaluation per ensemble member
    if mode in ("joint", "prediction", "model") and ds_model_t is not None and ds_ref_t is not None and not _lead_td_mismatch:
        try:
            ds_ref_aligned = _align_ref_to_model(ds_ref_t, ds_model_t)
        except ValueError as exc:
            _log(f"    [{counter}] Grid alignment failed: {exc}")
            ds_ref_aligned = None

        if ds_ref_aligned is not None:
            for m in ens_members:
                ds_model_m = ds_model_t.sel({ens_dim: m}) if (ens_dim and ens_dim in ds_model_t.dims) else ds_model_t

                try:
                    k_pred, e_pred = compute_ke_spectrum(ds_model_m, level=500.0)
                    k_ref, e_ref = compute_ke_spectrum(ds_ref_aligned, level=500.0)
                    n_min = min(len(e_pred), len(e_ref))

                    k_common = k_pred[:n_min]
                    e_pred_c = e_pred[:n_min]
                    e_ref_c = e_ref[:n_min]

                    eff_res_out = _find_effective_resolution(k_common, e_pred_c, e_ref_c)
                    L_eff, ratio = eff_res_out if isinstance(eff_res_out, tuple) else (eff_res_out, float("nan"))

                    s_div, s_res = compute_spectral_scores(e_pred_c, e_ref_c)

                    _append_summary("effective_resolution_km", L_eff, None, ens_member=m)
                    _append_summary("small_scale_ratio", ratio, None, ens_member=m)
                    _append_summary("spectral_divergence", s_div, None, ens_member=m)
                    _append_summary("spectral_residual", s_res, None, ens_member=m)

                    for wi in range(n_min):
                        spectrum_rows.append({
                            "date": date_str,
                            "lead_hours": lead_hours,
                            "variable": "KE",
                            "wavenumber": int(k_pred[wi]),
                            "power_pred": float(e_pred[wi]),
                            "power_ref": float(e_ref[wi]),
                            "ensemble_member": m,
                        })
                except (ValueError, KeyError, AttributeError) as exc:
                    _log(f"    [{counter}] Spectral evaluation failed for member {m}: {exc}")

    if not summary_rows:
        _append_summary("ALL_METRICS_FAILED", None, None, ens_member=0)

    return summary_rows, ts_rows, spectrum_rows, lr_dist_rows


# ============================================================================
# Core Batch Evaluation Loop
# ============================================================================

def run_evaluation(
    dates: List[str],
    output_csv: Path,
    mode: str = "joint",
    workers: int = DEFAULT_WORKERS,
    verbose: bool = True,
    prediction_zarr: str = DEFAULT_MODEL_ZARR,
    ref_zarr: str = REF_ZARR,
    model_name: str = "model",
    lead_times: Optional[List[Tuple[str, np.timedelta64]]] = None,
    static_zarr: Optional[str] = None,
    extended_spectra: bool = False,
    sp_ablation: str = "default",
) -> pd.DataFrame:
    """Run parallel physics metric evaluations across all requested dates and lead times.

    Args:
        dates: List of ISO date strings.
        output_csv: Path to output CSV file.
        mode: Evaluation mode ('joint', 'ref', 'model').
        workers: Number of parallel worker processes.
        verbose: Print progress logs.
        prediction_zarr: URL/Path to prediction dataset.
        ref_zarr: URL/Path to reference dataset.
        model_name: Name of evaluated model.
        lead_times: List of (label, timedelta) lead times.
        static_zarr: Path to static fields dataset.
        extended_spectra: Include Q and 850hPa spectra.
        sp_ablation: Surface pressure derivation ablation mode.

    Returns:
        pd.DataFrame containing long-format evaluation metric results.
    """
    _lead_times = lead_times if lead_times is not None else LEAD_TIMES
    work_items = [
        (date_str, lead_label, lead_td)
        for date_str in dates
        for lead_label, lead_td in _lead_times
    ]
    n_combos = len(work_items)

    if verbose:
        print("\n" + "=" * 70)
        print("  PHYSICS EVALUATION — WeatherBench 2 Zarr Streaming")
        print("=" * 70)
        print(f"  Prediction : {prediction_zarr}")
        print(f"  Reference  : {ref_zarr}")
        print(f"  Dates      : {len(dates)}")
        print(f"  Lead times : {[label for label, _ in _lead_times]}")
        print(f"  Total evals: {n_combos}")
        print(f"  Workers    : {workers}")
        print(f"  Mode       : {mode}")
        print(f"  Output     : {output_csv}")
        print("=" * 70)

    all_rows: List[Dict[str, Any]] = []
    all_ts_rows: List[Dict[str, Any]] = []
    all_spectrum_rows: List[Dict[str, Any]] = []
    all_lr_dist_rows: List[Dict[str, Any]] = []

    TASK_TIMEOUT = 600

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for idx, (date_str, lead_label, lead_td) in enumerate(work_items, 1):
            fut = pool.submit(
                _evaluate_one,
                prediction_zarr,
                ref_zarr,
                date_str,
                lead_label,
                lead_td,
                idx,
                n_combos,
                mode,
                verbose,
                static_zarr,
                model_name,
                extended_spectra,
                sp_ablation,
            )
            futures[fut] = (idx, date_str, lead_label)

        for fut in as_completed(futures):
            idx, date_str, lead_label = futures[fut]
            try:
                summary_rows, ts_rows, spectrum_rows, lr_dist_rows = fut.result(timeout=TASK_TIMEOUT)
                all_rows.extend(summary_rows)
                all_ts_rows.extend(ts_rows)
                all_spectrum_rows.extend(spectrum_rows)
                all_lr_dist_rows.extend(lr_dist_rows)
            except TimeoutError:
                if verbose:
                    print(f"  ⚠ Task {idx} ({date_str} {lead_label}) timed out after {TASK_TIMEOUT}s.")
            except Exception as exc:
                if verbose:
                    print(f"  ⚠ Worker exception (task {idx}): {exc}")

    if not all_rows:
        if verbose:
            print("\n  ⚠ No successful results obtained.")
        return pd.DataFrame()

    all_rows.sort(key=lambda r: (r["date"], r["lead_time_hours"], r["metric_name"], r.get("ensemble_member", 0)))
    df = pd.DataFrame(all_rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    if verbose:
        print(f"\n  ✓ Summary saved → {output_csv} ({len(df)} rows)")

    if all_ts_rows:
        year_str = dates[0][:4] if dates else "unknown"
        ts_csv = output_csv.parent / f"time_series_{model_name}_{year_str}.csv"
        df_ts = pd.DataFrame(all_ts_rows)
        df_ts.drop_duplicates(subset=["date", "forecast_hour", "ensemble_member"], inplace=True)
        df_ts.sort_values(["date", "forecast_hour", "ensemble_member"], inplace=True)
        df_ts.to_csv(ts_csv, index=False)
        if verbose:
            print(f"  ✓ Time series saved → {ts_csv} ({len(df_ts)} rows)")

    return df


# ============================================================================
# CLI Command Entrypoint
# ============================================================================

def main() -> None:
    """CLI entrypoint for physeval-run command."""
    parser = argparse.ArgumentParser(
        description="Physics evaluation for AI weather models (WB2 Zarr streaming)"
    )
    parser.add_argument("--year", type=int, default=2022, help="Year to evaluate (default: 2022).")
    parser.add_argument("--dates", nargs="+", default=None, help="Dates to evaluate (e.g. 2022-01-01 2022-01-15).")
    parser.add_argument("--month", type=str, default=None, help="Evaluate all days of month (e.g. 2022-01).")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help=f"Worker process count (default: {DEFAULT_WORKERS}).")
    parser.add_argument("--output", type=str, default=None, help="Output CSV file path.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory for generated CSV files.")
    parser.add_argument("--mode", type=str, choices=["joint", "ref", "reference", "prediction", "model"], default="joint", help="Evaluation mode.")
    parser.add_argument("--model", type=str, default="model", help="Model identifier name.")
    parser.add_argument("--prediction-zarr", type=str, default=DEFAULT_MODEL_ZARR, help="Path/URL to prediction Zarr.")
    parser.add_argument("--ref-zarr", type=str, default=REF_ZARR, help="Path/URL to reference Zarr.")
    parser.add_argument("--lead-times", type=str, default=None, help="Comma-separated lead times (e.g. '12h,5d,10d').")
    parser.add_argument("--static-zarr", type=str, default=None, help="Path/URL to static fields Zarr.")
    parser.add_argument("--quiet", action="store_true", help="Suppress output logging.")
    parser.add_argument("--extended-spectra", action="store_true", help="Compute additional spectra.")
    parser.add_argument("--sp-ablation", type=str, choices=["default", "hypsometric", "ref_sp", "dry_hydro"], default="default", help="SP derivation ablation mode.")

    args = parser.parse_args()
    dates = _resolve_dates(args)

    output_dir = Path(args.output_dir)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = output_dir / f"physics_evaluation_{args.model}_{args.year}.csv"

    lt = _parse_lead_times(args.lead_times) if args.lead_times else None

    run_evaluation(
        dates=dates,
        output_csv=output_path,
        mode=args.mode,
        workers=args.workers,
        verbose=not args.quiet,
        prediction_zarr=args.prediction_zarr,
        ref_zarr=args.ref_zarr,
        model_name=args.model,
        lead_times=lt,
        static_zarr=args.static_zarr,
        extended_spectra=args.extended_spectra,
        sp_ablation=args.sp_ablation,
    )


if __name__ == "__main__":
    main()
