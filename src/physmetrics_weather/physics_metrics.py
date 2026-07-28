"""Physics Metrics Library for Weather Model Evaluation.

This module provides diagnostic physical metrics for evaluating AI weather prediction
models, including mass, water, and energy conservation, spectral resolution, and
atmospheric balance metrics.

Supported Metrics:
    1. Global Dry Air Mass (conservation)
    2. Global Water Mass (stability)
    3. Global Total Energy (stability)
    4. Kinetic Energy / Humidity Spectra & Effective Resolution (spectral)
    5. Spectral Divergence & Residual (spectral)
    6. Hydrostatic Balance (balance)
    7. Geostrophic Balance (balance)
    8. Environmental Lapse Rate Wasserstein Distance (thermal structure)

Ensemble Support:
    All metric functions automatically detect extra ensemble dimensions
    (e.g., 'ens', 'realization', 'member', 'ensemble', 'number'). When an ensemble
    dimension is present, calculations are performed per ensemble member or across
    the member axis, returning dictionary mappings or data structures containing
    per-member metric evaluations.

Dependencies:
    numpy, pandas, xarray, scipy, pyshtools.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pyshtools as pysh
import xarray as xr
from scipy.stats import linregress, wasserstein_distance


# ============================================================================
# Physical Constants
# ============================================================================

GRAVITY: float = 9.80665           # m/s²  – standard gravity
EARTH_RADIUS: float = 6.371e6      # m     – mean Earth radius
C_PD: float = 1004.64              # J/(kg·K) – dry air specific heat (const. pressure)
C_PV: float = 1810.0               # J/(kg·K) – water vapour specific heat (const. pressure)
L_V: float = 2.501e6               # J/kg  – latent heat of vaporisation (at 0 °C)
R_DRY: float = 287.05              # J/(kg·K) – specific gas constant, dry air
LAPSE_RATE: float = 0.0065         # K/m   – standard tropospheric lapse rate
OMEGA: float = 7.2921e-5           # rad/s – Earth angular velocity
EXAGRAM: float = 1e18              # kg    – conversion factor to Exagrams
R_V: float = 461.5                 # J/(kg·K) – specific gas constant, water vapor


# ============================================================================
# Variable & Dimension Names
# ============================================================================

SP_NAMES: Tuple[str, ...] = ("surface_pressure", "sp", "ps")
MSL_NAMES: Tuple[str, ...] = ("mean_sea_level_pressure", "msl")
Q_NAMES: Tuple[str, ...] = ("specific_humidity", "q")
T_NAMES: Tuple[str, ...] = ("temperature", "t")
U_NAMES: Tuple[str, ...] = ("u_component_of_wind", "u")
V_NAMES: Tuple[str, ...] = ("v_component_of_wind", "v")
PHI_NAMES: Tuple[str, ...] = ("geopotential", "z")
T2M_NAMES: Tuple[str, ...] = ("2m_temperature", "t2m")
ZSFC_NAMES: Tuple[str, ...] = ("geopotential_at_surface", "z_sfc", "orography")

LEVEL_DIM_NAMES: Tuple[str, ...] = ("level", "pressure_level", "plev", "isobaricInhPa")
PRED_TD_NAMES: Tuple[str, ...] = ("prediction_timedelta", "lead_time", "step", "timedelta")
ENSEMBLE_DIM_NAMES: Tuple[str, ...] = ("ens", "realization", "member", "ensemble", "number")


# ============================================================================
# Structured Dataclasses & Domain Models
# ============================================================================

@dataclass
class MetricResult:
    """Structured container for metric evaluation results.

    Attributes:
        name: Name of the evaluated metric.
        value: Evaluation value (float for deterministic, dict for ensemble).
        units: Physical units of the metric value.
        description: Description of what the metric measures.
        ensemble_member: Optional ensemble member identifier.
    """

    name: str
    value: Union[float, Dict[Any, float]]
    units: str
    description: str
    ensemble_member: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metric result into a dictionary dictionary representation."""
        return {
            "name": self.name,
            "value": self.value,
            "units": self.units,
            "description": self.description,
            "ensemble_member": self.ensemble_member,
        }


class DatasetValidator:
    """Helper utility for dataset validation and variable discovery."""

    @staticmethod
    def find_variable(ds: xr.Dataset, candidates: Tuple[str, ...]) -> Optional[str]:
        """Find the first matching variable name in dataset."""
        for name in candidates:
            if name in ds.data_vars:
                return name
        return None

    @staticmethod
    def require_variable(ds: xr.Dataset, candidates: Tuple[str, ...], role: str) -> str:
        """Find matching variable or raise informative KeyError."""
        var = DatasetValidator.find_variable(ds, candidates)
        if var is None:
            raise KeyError(
                f"Required variable for '{role}' not found in dataset. "
                f"Tried candidates: {candidates}. Available variables: {list(ds.data_vars)}"
            )
        return var


def _find_var(ds: xr.Dataset, candidates: Tuple[str, ...]) -> Optional[str]:
    """Find the first variable name matching any candidate in the dataset."""
    return DatasetValidator.find_variable(ds, candidates)


def _detect_level_dim(ds: Union[xr.Dataset, xr.DataArray]) -> str:
    """Auto-detect the name of the pressure-level dimension."""
    for name in LEVEL_DIM_NAMES:
        if name in ds.dims:
            return name
    raise ValueError(
        f"Could not automatically detect the pressure level dimension. "
        f"Looked for: {LEVEL_DIM_NAMES}. Available dims: {list(ds.dims)}"
    )


def _detect_pred_td_dim(ds: Union[xr.Dataset, xr.DataArray]) -> Optional[str]:
    """Auto-detect the name of the prediction timedelta dimension."""
    for name in PRED_TD_NAMES:
        if name in ds.dims:
            return name
    return None


def _detect_ensemble_dim(ds: Union[xr.Dataset, xr.DataArray]) -> Optional[str]:
    """Auto-detect the name of the ensemble dimension."""
    for name in ENSEMBLE_DIM_NAMES:
        if name in ds.dims:
            return name
    return None


# ============================================================================
# Grid & Coordinate Utilities
# ============================================================================

def get_grid_cell_area(
    ds: Union[xr.Dataset, xr.DataArray],
    lat_name: str = "latitude",
    lon_name: str = "longitude",
    earth_radius: float = EARTH_RADIUS,
) -> xr.DataArray:
    """Compute the area of each grid cell on a regular latitude/longitude grid.

    Formula:
        A_i = R² × Δλ × abs(sin(φ_north) − sin(φ_south))

    Args:
        ds: xarray Dataset or DataArray containing latitude and longitude coordinates.
        lat_name: Name of the latitude coordinate.
        lon_name: Name of the longitude coordinate.
        earth_radius: Mean Earth radius in meters.

    Returns:
        xr.DataArray containing cell areas in square meters (m²).

    Raises:
        ValueError: If latitude or longitude coordinates are missing.
    """
    if lat_name not in ds.coords or lon_name not in ds.coords:
        raise ValueError(f"Dataset missing '{lat_name}' or '{lon_name}' coordinates.")

    lat = ds[lat_name].values
    lon = ds[lon_name].values

    dlon = np.abs(np.diff(lon).mean())
    dlon_rad = np.deg2rad(dlon)

    lat_rad = np.deg2rad(lat)
    midpoints = (lat_rad[:-1] + lat_rad[1:]) / 2.0
    lat_s = np.empty_like(lat_rad)
    lat_n = np.empty_like(lat_rad)
    lat_s[0] = np.clip(2 * lat_rad[0] - midpoints[0], -np.pi / 2, np.pi / 2)
    lat_s[1:] = midpoints
    lat_n[:-1] = midpoints
    lat_n[-1] = np.clip(2 * lat_rad[-1] - midpoints[-1], -np.pi / 2, np.pi / 2)
    lat_s = np.clip(lat_s, -np.pi / 2, np.pi / 2)
    lat_n = np.clip(lat_n, -np.pi / 2, np.pi / 2)

    area_1d = earth_radius**2 * dlon_rad * np.abs(np.sin(lat_n) - np.sin(lat_s))
    area_2d = np.broadcast_to(area_1d[:, None], (len(lat), len(lon)))

    return xr.DataArray(
        area_2d,
        dims=[lat_name, lon_name],
        coords={lat_name: lat, lon_name: lon},
        name="grid_cell_area",
        attrs={"units": "m²", "long_name": "Grid cell area"},
    )


def derive_surface_pressure(
    ds: xr.Dataset,
    ds_static: xr.Dataset,
    msl_names: Tuple[str, ...] = MSL_NAMES,
    z_names: Tuple[str, ...] = ZSFC_NAMES,
    gravity: float = GRAVITY,
    r_dry: float = R_DRY,
    lapse_rate: float = LAPSE_RATE,
    lat_name: str = "latitude",
) -> xr.DataArray:
    """Derive surface pressure using the U.S. Standard Atmosphere (1976) profile.

    Formula:
        z_sfc = Φ_s / g
        P_s = P_MSL × (1 - (Γ × z_sfc) / T_0)^(g / (R_d × Γ))
        where T_0 = 288.15 K.

    Ensemble Support:
        If `ds` contains an ensemble dimension (e.g. 'ens'), the derived surface
        pressure array retains that dimension.

    Args:
        ds: xarray Dataset containing mean sea level pressure (MSL).
        ds_static: xarray Dataset containing surface geopotential (orography).
        msl_names: Candidates for MSL variable name.
        z_names: Candidates for surface geopotential variable name.
        gravity: Gravitational acceleration in m/s².
        r_dry: Gas constant for dry air in J/(kg·K).
        lapse_rate: Standard tropospheric lapse rate in K/m.
        lat_name: Name of latitude coordinate.

    Returns:
        xr.DataArray: Derived surface pressure in Pascals (Pa).

    Raises:
        ValueError: If MSL or surface geopotential variables are missing or grid mismatch > 1.
    """
    msl_name = DatasetValidator.require_variable(ds, msl_names, "Mean Sea Level Pressure")
    msl = ds[msl_name]

    z_name = DatasetValidator.require_variable(ds_static, z_names, "Surface Geopotential")
    z_sfc = ds_static[z_name]

    for tdim in ("time", "valid_time"):
        if tdim in z_sfc.dims:
            z_sfc = z_sfc.isel({tdim: 0}, drop=True)

    if lat_name in z_sfc.dims and lat_name in msl.dims:
        n_static = z_sfc.sizes[lat_name]
        n_target = msl.sizes[lat_name]

        if abs(n_static - n_target) > 1:
            raise ValueError("Latitude size mismatch between static and MSL grid is > 1.")
        if n_static == n_target + 1:
            z_sfc = z_sfc.sel({lat_name: msl[lat_name].values}, method="nearest")
        elif n_target == n_static + 1:
            msl = msl.sel({lat_name: z_sfc[lat_name].values}, method="nearest")

        if z_sfc.sizes[lat_name] == msl.sizes[lat_name]:
            z_sfc = z_sfc.assign_coords({lat_name: msl[lat_name]})

    lon_name_cands = [d for d in msl.dims if "lon" in d.lower()]
    lon_name = lon_name_cands[0] if lon_name_cands else "longitude"

    if lon_name in z_sfc.dims and lon_name in msl.dims:
        if z_sfc.sizes[lon_name] != msl.sizes[lon_name]:
            raise ValueError("Longitude size mismatch between surface geopotential and MSL.")
        z_sfc = z_sfc.assign_coords({lon_name: msl[lon_name]})

    t_0 = 288.15
    exponent = gravity / (r_dry * lapse_rate)
    z = z_sfc / gravity

    sp = msl * np.power((1.0 - (lapse_rate * z) / t_0), exponent)
    sp.name = "surface_pressure"
    sp.attrs = {
        "units": "Pa",
        "long_name": "Surface pressure (US Standard Atmosphere derivation)",
    }
    return sp


# ============================================================================
# Column Integration Helpers
# ============================================================================

def _integrate_column(
    field_3d: np.ndarray,
    levels_hpa: np.ndarray,
    ps_2d: np.ndarray,
    gravity: float = GRAVITY,
) -> np.ndarray:
    """Trapezoidal column integration with surface-pressure masking.

    Args:
        field_3d: 3D numpy array of vertical field values (level, lat, lon).
        levels_hpa: 1D numpy array of pressure levels in hPa.
        ps_2d: 2D numpy array of surface pressure in Pa (lat, lon).
        gravity: Gravitational acceleration in m/s².

    Returns:
        2D numpy array of vertical column integrated values (lat, lon).
    """
    levels_pa = levels_hpa.astype(np.float64) * 100.0
    sort_idx = np.argsort(levels_pa)
    levels_sorted = levels_pa[sort_idx]
    field_sorted = field_3d[sort_idx]

    n = len(levels_sorted)
    col = np.zeros_like(ps_2d, dtype=np.float64)

    dp_top = np.minimum(ps_2d, levels_sorted[0]) - 0.0
    col += field_sorted[0] * dp_top

    for k in range(n - 1):
        p_top = levels_sorted[k]
        p_bot = levels_sorted[k + 1]

        eff_top = np.minimum(ps_2d, p_top)
        eff_bot = np.minimum(ps_2d, p_bot)
        dp = np.maximum(0.0, eff_bot - eff_top)

        field_avg = 0.5 * (field_sorted[k] + field_sorted[k + 1])
        col += field_avg * dp

    dp_bottom = np.maximum(0.0, ps_2d - levels_sorted[-1])
    col += field_sorted[-1] * dp_bottom

    col /= gravity
    return col


def _ensure_ps_2d(ps: xr.DataArray) -> np.ndarray:
    """Extract a 2D numpy array (lat, lon) from a surface pressure DataArray.

    Args:
        ps: Input surface pressure DataArray.

    Returns:
        2D numpy array of surface pressure.

    Raises:
        ValueError: If surface pressure DataArray cannot be reduced to 2D.
    """
    arr = np.asarray(ps.values).squeeze()
    if arr.ndim == 0:
        return np.array([[float(arr)]])
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        warnings.warn(
            f"ps array has 3D shape {arr.shape} after squeeze; taking slice [0]."
        )
        return arr[0]
    raise ValueError(f"Cannot reduce ps shape {arr.shape} to 2D.")


def _compute_tcwv(
    ds: xr.Dataset,
    ps: xr.DataArray,
    q_name: str = "q",
    level_dim: str = "level",
    levels: Optional[np.ndarray] = None,
) -> xr.DataArray:
    """Integrate specific humidity to Total Column Water Vapour (TCWV).

    Args:
        ds: Input Dataset containing specific humidity.
        ps: Surface pressure DataArray.
        q_name: Specific humidity variable name string.
        level_dim: Pressure level dimension name string.
        levels: Optional array of pressure levels in hPa.

    Returns:
        2D DataArray containing TCWV values in kg/m².
    """
    if q_name not in ds.data_vars:
        q_name = DatasetValidator.require_variable(ds, Q_NAMES, "Specific Humidity")

    q = ds[q_name]
    if levels is None:
        levels = ds[level_dim].values

    ps_np = _ensure_ps_2d(ps)
    lat_dim = [d for d in q.dims if d != level_dim][0]
    lon_dim = [d for d in q.dims if d != level_dim][1]

    tcwv_np = _integrate_column(q.transpose(level_dim, ...).values, levels, ps_np)

    return xr.DataArray(
        tcwv_np,
        dims=[lat_dim, lon_dim],
        coords={lat_dim: q[lat_dim], lon_dim: q[lon_dim]},
        name="tcwv",
        attrs={"units": "kg/m²", "long_name": "Total Column Water Vapour"},
    )


# ============================================================================
# Metric 1 — Global Dry Air Mass
# ============================================================================

def compute_dry_air_mass(
    ds: xr.Dataset,
    ps: xr.DataArray,
    area: xr.DataArray,
    q_name: str = "q",
    level_dim: str = "level",
    levels: Optional[np.ndarray] = None,
    tcwv: Optional[xr.DataArray] = None,
) -> Union[float, Dict[Any, float]]:
    """Compute global dry air mass in Exagrams (10¹⁸ kg).

    Formula:
        M_d = Σ A_i × (P_s,i / g − TCWV_i)

    Ensemble Support:
        If `ds` or `ps` contains an ensemble dimension (e.g. 'ens', 'member',
        'realization'), the function computes the dry air mass for each member and
        returns a dictionary mapping member IDs to their respective values.

    Args:
        ds: Dataset containing specific humidity field.
        ps: Surface pressure DataArray.
        area: Grid cell area DataArray (m²).
        q_name: Variable name for specific humidity.
        level_dim: Dimension name for pressure levels.
        levels: Optional array of pressure levels in hPa.
        tcwv: Optional pre-computed TCWV DataArray.

    Returns:
        Float (if deterministic) or Dict[member, float] (if ensemble) representing
        global dry air mass in Eg.

    Raises:
        ValueError: If area and data grid shapes mismatch.
    """
    ens_dim = _detect_ensemble_dim(ds) or _detect_ensemble_dim(ps)
    if ens_dim is not None and ens_dim in ds.dims:
        results = {}
        for m in ds[ens_dim].values:
            ds_m = ds.sel({ens_dim: m})
            ps_m = ps.sel({ens_dim: m}) if ens_dim in ps.dims else ps
            tcwv_m = tcwv.sel({ens_dim: m}) if (tcwv is not None and ens_dim in tcwv.dims) else None
            results[m] = float(compute_dry_air_mass(
                ds_m, ps_m, area, q_name=q_name, level_dim=level_dim, levels=levels, tcwv=tcwv_m
            ))
        return results

    if tcwv is None:
        tcwv = _compute_tcwv(ds, ps, q_name=q_name, level_dim=level_dim, levels=levels)

    col_dry = ps / GRAVITY - tcwv
    if area.shape != col_dry.shape:
        raise ValueError(f"Area shape {area.shape} and data shape {col_dry.shape} do not match.")

    dry_mass_kg = float((area * col_dry).sum())
    return dry_mass_kg / EXAGRAM


# ============================================================================
# Metric 2 — Global Water Mass
# ============================================================================

def compute_water_mass(
    ds: xr.Dataset,
    ps: xr.DataArray,
    area: xr.DataArray,
    q_name: str = "q",
    level_dim: str = "level",
    levels: Optional[np.ndarray] = None,
    tcwv: Optional[xr.DataArray] = None,
) -> Union[float, Dict[Any, float]]:
    """Compute global total atmospheric water mass in kg.

    Formula:
        M_w = Σ A_i × TCWV_i

    Ensemble Support:
        If `ds` or `ps` contains an ensemble dimension, evaluates the water mass per
        member and returns a dictionary mapping member IDs to values.

    Args:
        ds: Dataset containing specific humidity.
        ps: Surface pressure DataArray.
        area: Grid cell area DataArray (m²).
        q_name: Variable name for specific humidity.
        level_dim: Dimension name for pressure levels.
        levels: Optional array of pressure levels in hPa.
        tcwv: Optional pre-computed TCWV DataArray.

    Returns:
        Float (if deterministic) or Dict[member, float] (if ensemble) representing
        global water mass in kg.

    Raises:
        ValueError: If area and TCWV shapes mismatch.
    """
    ens_dim = _detect_ensemble_dim(ds) or _detect_ensemble_dim(ps)
    if ens_dim is not None and ens_dim in ds.dims:
        results = {}
        for m in ds[ens_dim].values:
            ds_m = ds.sel({ens_dim: m})
            ps_m = ps.sel({ens_dim: m}) if ens_dim in ps.dims else ps
            tcwv_m = tcwv.sel({ens_dim: m}) if (tcwv is not None and ens_dim in tcwv.dims) else None
            results[m] = float(compute_water_mass(
                ds_m, ps_m, area, q_name=q_name, level_dim=level_dim, levels=levels, tcwv=tcwv_m
            ))
        return results

    if tcwv is None:
        tcwv = _compute_tcwv(ds, ps, q_name=q_name, level_dim=level_dim, levels=levels)
    if area.shape != tcwv.shape:
        raise ValueError(f"Area shape {area.shape} and TCWV shape {tcwv.shape} do not match!")
    return float((area * tcwv).sum())


# ============================================================================
# Metric 3 — Global Total Energy
# ============================================================================

def compute_total_energy(
    ds: xr.Dataset,
    ps: xr.DataArray,
    area: xr.DataArray,
    z_sfc: xr.DataArray,
    t_name: str = "temperature",
    q_name: str = "q",
    u_names: Tuple[str, ...] = U_NAMES,
    v_names: Tuple[str, ...] = V_NAMES,
    level_dim: str = "level",
    levels: Optional[np.ndarray] = None,
    c_pd: float = C_PD,
    c_pv: float = C_PV,
    l_v: float = L_V,
) -> Union[float, Dict[Any, float]]:
    """Compute global total atmospheric energy in Joules (J).

    Formula:
        TE = (1/g) Σ A_i ∫ (c_p T + Φ_s + L_v q + ½(u² + v²)) dp
        where c_p = c_pd (1 − q) + c_pv q.

    Ensemble Support:
        If `ds` or `ps` contains an ensemble dimension, evaluates total energy for
        each ensemble member and returns a dictionary of results.

    Args:
        ds: Dataset containing temperature, humidity, and wind components.
        ps: Surface pressure DataArray.
        area: Grid cell area DataArray (m²).
        z_sfc: Surface geopotential DataArray.
        t_name: Temperature variable name.
        q_name: Specific humidity variable name.
        u_names: Candidate variable names for zonal wind.
        v_names: Candidate variable names for meridional wind.
        level_dim: Pressure level dimension name.
        levels: Optional array of pressure levels in hPa.
        c_pd: Specific heat capacity of dry air in J/(kg·K).
        c_pv: Specific heat capacity of water vapor in J/(kg·K).
        l_v: Latent heat of vaporization in J/kg.

    Returns:
        Float (if deterministic) or Dict[member, float] (if ensemble) of total energy in J.

    Raises:
        ValueError: If required variables (u, v, T, q) are missing or grids mismatch.
    """
    ens_dim = _detect_ensemble_dim(ds) or _detect_ensemble_dim(ps)
    if ens_dim is not None and ens_dim in ds.dims:
        results = {}
        for m in ds[ens_dim].values:
            ds_m = ds.sel({ens_dim: m})
            ps_m = ps.sel({ens_dim: m}) if ens_dim in ps.dims else ps
            results[m] = float(compute_total_energy(
                ds_m, ps_m, area, z_sfc=z_sfc, t_name=t_name, q_name=q_name,
                u_names=u_names, v_names=v_names, level_dim=level_dim, levels=levels,
                c_pd=c_pd, c_pv=c_pv, l_v=l_v,
            ))
        return results

    if levels is None:
        levels = ds[level_dim].values

    t_var = (
        t_name
        if t_name in ds.data_vars
        else DatasetValidator.require_variable(ds, T_NAMES, "Temperature")
    )
    q_var = (
        q_name
        if q_name in ds.data_vars
        else DatasetValidator.require_variable(ds, Q_NAMES, "Specific Humidity")
    )

    T = ds[t_var].transpose(level_dim, ...).values
    q = ds[q_var].transpose(level_dim, ...).values

    c_p = c_pd * (1.0 - q) + c_pv * q

    ref_var = ds[t_var]
    lat_dim_ = [d for d in ref_var.dims if d != level_dim][0]
    lon_dim_ = [d for d in ref_var.dims if d != level_dim][1]

    lat_diff = abs(z_sfc.sizes[lat_dim_] - ref_var.sizes[lat_dim_])
    lon_diff = abs(z_sfc.sizes[lon_dim_] - ref_var.sizes[lon_dim_])

    if lat_diff <= 1 and lon_diff == 0:
        z_aligned = z_sfc.reindex_like(ref_var.isel({level_dim: 0}), method="nearest")
    else:
        raise ValueError(
            f"Grid mismatch too large: z_sfc={z_sfc.shape}, model={ref_var.shape}."
        )

    nlevels = T.shape[0]
    z_sfc_np = z_aligned.values
    z_sfc_3d = np.broadcast_to(z_sfc_np[None, :, :], (nlevels,) + z_sfc_np.shape)

    u_var = DatasetValidator.require_variable(ds, u_names, "Zonal Wind (u)")
    v_var = DatasetValidator.require_variable(ds, v_names, "Meridional Wind (v)")

    u_val = ds[u_var].transpose(level_dim, ...).values
    v_val = ds[v_var].transpose(level_dim, ...).values
    wspd_sq = u_val**2 + v_val**2

    energy_density = c_p * T + z_sfc_3d + l_v * q + 0.5 * wspd_sq
    ps_np = _ensure_ps_2d(ps)
    col_energy = _integrate_column(energy_density, levels, ps_np)
    col_da = xr.DataArray(
        col_energy,
        dims=[lat_dim_, lon_dim_],
        coords={lat_dim_: ds[lat_dim_], lon_dim_: ds[lon_dim_]},
    )
    if area.shape != col_da.shape:
        raise ValueError(f"Area shape {area.shape} and ENERGY shape {col_da.shape} do not match!")
    return float((area * col_da).sum())


# ============================================================================
# Metric 4 — Spectral Analysis & Effective Resolution
# ============================================================================

def _ke_spectrum_spharm(
    u: np.ndarray,
    v: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute spherical-harmonic kinetic energy spectrum E(l) using pyshtools.

    Args:
        u: 2D numpy array of zonal wind component (lat, lon).
        v: 2D numpy array of meridional wind component (lat, lon).

    Returns:
        Tuple of (wavenumber_array, energy_spectrum_array).

    Raises:
        ValueError: If input wind arrays are not 2D or grid is invalid Driscoll-Healy.
    """
    u = np.asarray(u, dtype=np.float64).squeeze()
    v = np.asarray(v, dtype=np.float64).squeeze()

    if u.ndim != 2 or v.ndim != 2:
        raise ValueError(f"Expected 2D wind fields, got u shape = {u.shape}, v shape = {v.shape}.")

    nlat, nlon = u.shape

    if nlat % 2 != 0:
        u, v = u[:-1, :], v[:-1, :]
        nlat -= 1
    if nlon % 2 != 0:
        u, v = u[:, :-1], v[:, :-1]
        nlon -= 1

    if nlon == 2 * nlat:
        sampling = 2
        lmax = nlat // 2 - 1
    elif nlon == nlat:
        sampling = 1
        lmax = nlat // 2 - 1
    else:
        raise ValueError(
            f"Grid ({nlat}, {nlon}) is not a valid Driscoll-Healy grid "
            f"(requires nlon == 2*nlat or nlon == nlat)."
        )

    u_c = pysh.expand.SHExpandDH(u, sampling=sampling, lmax_calc=lmax)
    v_c = pysh.expand.SHExpandDH(v, sampling=sampling, lmax_calc=lmax)

    wavenumber = np.arange(lmax + 1)
    energy = np.zeros(lmax + 1)
    for l in range(lmax + 1):
        for m in range(l + 1):
            pw = u_c[0, l, m] ** 2 + v_c[0, l, m] ** 2
            if m > 0:
                pw += u_c[1, l, m] ** 2 + v_c[1, l, m] ** 2
            energy[l] += 0.5 * pw

    return wavenumber, energy


def _scalar_spectrum_spharm(
    field: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute spherical-harmonic power spectrum S(l) of a 2D scalar field.

    Args:
        field: 2D numpy array of scalar values (lat, lon).

    Returns:
        Tuple of (wavenumber_array, power_spectrum_array).

    Raises:
        ValueError: If field array is not 2D or grid is invalid Driscoll-Healy.
    """
    field = np.asarray(field, dtype=np.float64).squeeze()

    if field.ndim != 2:
        raise ValueError(f"Expected 2D field, got shape = {field.shape}")

    nlat, nlon = field.shape

    if nlat % 2 != 0:
        field = field[:-1, :]
        nlat -= 1
    if nlon % 2 != 0:
        field = field[:, :-1]
        nlon -= 1

    if nlon == 2 * nlat:
        sampling = 2
        lmax = nlat // 2 - 1
    elif nlon == nlat:
        sampling = 1
        lmax = nlat // 2 - 1
    else:
        raise ValueError(
            f"Grid ({nlat}, {nlon}) is not a valid Driscoll-Healy grid "
            f"(requires nlon == 2*nlat or nlon == nlat)."
        )

    coeffs = pysh.expand.SHExpandDH(field, sampling=sampling, lmax_calc=lmax)

    wavenumber = np.arange(lmax + 1)
    power = np.zeros(lmax + 1)
    for l in range(lmax + 1):
        for m in range(l + 1):
            pw = coeffs[0, l, m] ** 2
            if m > 0:
                pw += coeffs[1, l, m] ** 2
            power[l] += pw

    return wavenumber, power


def compute_ke_spectrum(
    ds: xr.Dataset,
    level: float = 500.0,
    u_names: Tuple[str, ...] = U_NAMES,
    v_names: Tuple[str, ...] = V_NAMES,
    level_dim: str = "level",
) -> Union[Tuple[np.ndarray, np.ndarray], Dict[Any, Tuple[np.ndarray, np.ndarray]]]:
    """Compute the kinetic energy spectrum E(l) at a target pressure level.

    Args:
        ds: Input Dataset containing wind components.
        level: Target pressure level in hPa (default: 500.0).
        u_names: Candidate variable names for zonal wind.
        v_names: Candidate variable names for meridional wind.
        level_dim: Pressure level dimension name.

    Returns:
        Tuple of (wavenumber_array, energy_spectrum_array) for deterministic input,
        or Dict[member_id, Tuple[wavenumber, energy]] for ensemble datasets.

    Raises:
        KeyError: If wind variables cannot be found.
        ValueError: If pressure level dimension cannot be detected.
    """
    ens_dim = _detect_ensemble_dim(ds)
    if ens_dim is not None and ens_dim in ds.dims:
        results = {}
        for m in ds[ens_dim].values:
            results[m] = compute_ke_spectrum(
                ds.sel({ens_dim: m}),
                level=level,
                u_names=u_names,
                v_names=v_names,
                level_dim=level_dim,
            )
        return results

    if level_dim not in ds.dims:
        level_dim = _detect_level_dim(ds)

    u_var = DatasetValidator.require_variable(ds, u_names, "Zonal Wind (u)")
    v_var = DatasetValidator.require_variable(ds, v_names, "Meridional Wind (v)")

    u = ds[u_var]
    v = ds[v_var]

    if level_dim in u.dims:
        lvls = ds[level_dim].values
        idx = int(np.abs(lvls - level).argmin())
        u = u.isel({level_dim: idx})
        v = v.isel({level_dim: idx})

    lat_dims = [d for d in u.dims if "lat" in d.lower()]
    lon_dims = [d for d in u.dims if "lon" in d.lower()]
    if lat_dims and lon_dims:
        u = u.transpose(..., lat_dims[0], lon_dims[0])
        v = v.transpose(..., lat_dims[0], lon_dims[0])

    return _ke_spectrum_spharm(u.values, v.values)


def compute_q_spectrum(
    ds: xr.Dataset,
    level: float = 500.0,
    q_names: Tuple[str, ...] = Q_NAMES,
    level_dim: str = "level",
) -> Union[Tuple[np.ndarray, np.ndarray], Dict[Any, Tuple[np.ndarray, np.ndarray]]]:
    """Compute specific humidity power spectrum S_q(l) at a single pressure level.

    Args:
        ds: Input Dataset containing specific humidity field.
        level: Target pressure level in hPa (default: 500.0).
        q_names: Candidate variable names for specific humidity.
        level_dim: Pressure level dimension name.

    Returns:
        Tuple of (wavenumber_array, power_spectrum_array) for deterministic input,
        or Dict[member_id, Tuple[wavenumber, power]] for ensemble datasets.

    Raises:
        KeyError: If specific humidity variable cannot be found.
    """
    ens_dim = _detect_ensemble_dim(ds)
    if ens_dim is not None and ens_dim in ds.dims:
        results = {}
        for m in ds[ens_dim].values:
            results[m] = compute_q_spectrum(
                ds.sel({ens_dim: m}), level=level, q_names=q_names, level_dim=level_dim
            )
        return results

    if level_dim not in ds.dims:
        level_dim = _detect_level_dim(ds)

    q_var = DatasetValidator.require_variable(ds, q_names, "Specific Humidity")

    q = ds[q_var]
    if level_dim in q.dims:
        lvls = ds[level_dim].values
        idx = int(np.abs(lvls - level).argmin())
        q = q.isel({level_dim: idx})

    lat_dims = [d for d in q.dims if "lat" in d.lower()]
    lon_dims = [d for d in q.dims if "lon" in d.lower()]
    if lat_dims and lon_dims:
        q = q.transpose(..., lat_dims[0], lon_dims[0])

    return _scalar_spectrum_spharm(q.values)


def _find_effective_resolution(
    k: np.ndarray,
    e_pred: np.ndarray,
    e_true: np.ndarray,
    threshold: float = 0.5,
    k_min: int = 10,
    n_consecutive: int = 5,
    earth_radius: float = EARTH_RADIUS,
) -> Tuple[float, float]:
    """Calculate effective spatial resolution L_eff (km) where E_pred / E_true < threshold.

    Args:
        k: 1D numpy array of spherical harmonic wavenumbers.
        e_pred: Predicted spectrum energy array.
        e_true: Reference ground-truth spectrum energy array.
        threshold: Energy ratio threshold (default: 0.5).
        k_min: Minimum wavenumber to start checking.
        n_consecutive: Number of consecutive wavenumbers below threshold.
        earth_radius: Mean Earth radius in meters.

    Returns:
        Tuple of (effective_resolution_km, small_scale_energy_ratio).

    Raises:
        ValueError: If spectrum length is shorter than n_consecutive.
    """
    mask = (k >= k_min) & (e_true > 1e-12)
    k_sel = k[mask]
    ratio = e_pred[mask] / e_true[mask]

    below = ratio < threshold
    n = len(ratio)
    if n < n_consecutive:
        raise ValueError(
            f"Spectrum too short ({n} wavenumbers). "
            f"Need at least {n_consecutive} for effective resolution."
        )

    idx = None
    run = 0
    for i in range(n):
        if below[i]:
            run += 1
            if run >= n_consecutive:
                idx = i - n_consecutive + 1
                break
        else:
            run = 0

    if idx is None:
        l_max = float(k_sel[-1]) if len(k_sel) > 0 else float(k[-1])
        L_grid_km = (2.0 * np.pi * earth_radius / l_max) / 1000.0
        return L_grid_km, float(np.mean(ratio))

    k_c = float(k_sel[idx])
    L_km = (2.0 * np.pi * earth_radius / k_c) / 1000.0
    small_scale_ratio = float(np.mean(ratio[idx:])) if idx < len(ratio) else float(ratio[-1])

    return L_km, small_scale_ratio


def compute_spectral_scores(
    e_pred: np.ndarray,
    e_true: np.ndarray,
    eps: float = 1e-12,
) -> Tuple[float, float]:
    """Compute spectral divergence (1-Wasserstein) and spectral residual (log RMSE).

    Args:
        e_pred: 1D numpy array of predicted spectrum power.
        e_true: 1D numpy array of reference ground-truth spectrum power.
        eps: Small epsilon numerical stabilizer for log calculations.

    Returns:
        Tuple of (spectral_divergence_w1, spectral_residual_log_rmse).
    """
    k = np.arange(len(e_pred))
    spec_div = float(
        wasserstein_distance(
            u_values=k, v_values=k, u_weights=e_true, v_weights=e_pred
        )
    )

    log_diff = np.log(e_pred + eps) - np.log(e_true + eps)
    spec_res = float(np.sqrt(np.mean(log_diff**2)))

    return spec_div, spec_res


# ============================================================================
# Metric 5 — Hydrostatic Balance RMSE
# ============================================================================

def compute_hydrostatic_imbalance(
    ds: xr.Dataset,
    area: xr.DataArray,
    phi_name: Optional[str] = None,
    t_name: Optional[str] = None,
    q_name: str = "q",
    level_dim: str = "level",
    lat_name: str = "latitude",
    p_top: float = 500.0,
    p_bot: float = 850.0,
    r_dry: float = R_DRY,
) -> Union[float, Dict[Any, float]]:
    """Compute hydrostatic balance error RMSE (m²/s²) between p_top and p_bot.

    Formula:
        Error = abs((Φ_top − Φ_bot) − R_d T̄_v ln(p_bot/p_top))

    Ensemble Support:
        If `ds` contains an ensemble dimension, evaluates the hydrostatic error per
        member and returns a dictionary mapping member IDs to RMSE values.

    Args:
        ds: Dataset containing geopotential and temperature fields.
        area: Grid cell area DataArray (m²).
        phi_name: Geopotential variable name.
        t_name: Temperature variable name.
        q_name: Specific humidity variable name.
        level_dim: Pressure level dimension name.
        lat_name: Latitude dimension name.
        p_top: Upper pressure level in hPa.
        p_bot: Lower pressure level in hPa.
        r_dry: Gas constant for dry air in J/(kg·K).

    Returns:
        Float (if deterministic) or Dict[member, float] (if ensemble) representing
        area-weighted hydrostatic balance RMSE in m²/s².

    Raises:
        ValueError: If geopotential or temperature variables are missing or shapes mismatch.
    """
    ens_dim = _detect_ensemble_dim(ds)
    if ens_dim is not None and ens_dim in ds.dims:
        results = {}
        for m in ds[ens_dim].values:
            results[m] = float(compute_hydrostatic_imbalance(
                ds.sel({ens_dim: m}), area, phi_name=phi_name, t_name=t_name,
                q_name=q_name, level_dim=level_dim, lat_name=lat_name,
                p_top=p_top, p_bot=p_bot, r_dry=r_dry,
            ))
        return results

    if level_dim not in ds.dims:
        level_dim = _detect_level_dim(ds)
    levels = ds[level_dim].values

    phi_var = phi_name or DatasetValidator.require_variable(ds, PHI_NAMES, "Geopotential")
    t_var = t_name or DatasetValidator.require_variable(ds, T_NAMES, "Temperature")
    q_var = q_name if q_name in ds.data_vars else _find_var(ds, Q_NAMES)

    def _sel_level(var: str, p: float) -> xr.DataArray:
        idx = int(np.abs(levels - p).argmin())
        return ds[var].isel({level_dim: idx})

    phi_top = _sel_level(phi_var, p_top)
    phi_bot = _sel_level(phi_var, p_bot)
    T_top = _sel_level(t_var, p_top)
    T_bot = _sel_level(t_var, p_bot)

    if q_var is not None and q_var in ds.data_vars:
        q_top = _sel_level(q_var, p_top)
        q_bot = _sel_level(q_var, p_bot)
        Tv_top = T_top * (1.0 + (R_V / R_DRY - 1) * q_top)
        Tv_bot = T_bot * (1.0 + (R_V / R_DRY - 1) * q_bot)
    else:
        Tv_top = T_top
        Tv_bot = T_bot

    Tv_mean = 0.5 * (Tv_top + Tv_bot)
    lhs = phi_top - phi_bot
    rhs = r_dry * Tv_mean * np.log(p_bot / p_top)
    error = lhs - rhs

    lat_dim_e = next((d for d in error.dims if "lat" in d.lower()), None)
    lon_dim_e = next((d for d in error.dims if "lon" in d.lower()), None)
    if lat_dim_e and lon_dim_e and error.dims != (lat_dim_e, lon_dim_e):
        error = error.transpose(lat_dim_e, lon_dim_e)

    if area.shape != error.shape:
        raise ValueError(f"Area shape {area.shape} and error shape {error.shape} do not match.")

    weights = area / float(area.sum())
    mse = float((weights.values * error.values**2).sum())
    return float(np.sqrt(mse))


# ============================================================================
# Metric 6 — Geostrophic Balance RMSE
# ============================================================================

def compute_geostrophic_imbalance(
    ds: xr.Dataset,
    area: xr.DataArray,
    phi_name: str = "geopotential",
    u_names: Tuple[str, ...] = U_NAMES,
    v_names: Tuple[str, ...] = V_NAMES,
    level: float = 500.0,
    level_dim: str = "level",
    lat_name: str = "latitude",
    lon_name: str = "longitude",
    lat_cutoff: float = 10.0,
    earth_radius: float = EARTH_RADIUS,
    omega: float = OMEGA,
) -> Union[float, Dict[Any, float]]:
    """Compute geostrophic wind balance RMSE (m/s) at a given pressure level.

    Formula:
        f u_g = -∂Φ/∂y,  f v_g = ∂Φ/∂x
        Error = sqrt(area_weighted_mean((u - u_g)² + (v - v_g)²))

    Args:
        ds: Input Dataset containing geopotential and wind components.
        area: Grid cell area DataArray (m²).
        phi_name: Geopotential variable name.
        u_names: Zonal wind variable name candidates.
        v_names: Meridional wind variable name candidates.
        level: Target pressure level in hPa (default: 500.0).
        level_dim: Pressure level dimension name.
        lat_name: Latitude coordinate name.
        lon_name: Longitude coordinate name.
        lat_cutoff: Latitude cutoff in degrees (excluding tropics near equator).
        earth_radius: Mean Earth radius in meters.
        omega: Earth angular rotation velocity in rad/s.

    Returns:
        Float (if deterministic) or Dict[member, float] (if ensemble) representing
        area-weighted geostrophic balance RMSE in m/s.

    Raises:
        KeyError: If geopotential or wind variables are missing.
    """
    ens_dim = _detect_ensemble_dim(ds)
    if ens_dim is not None and ens_dim in ds.dims:
        results = {}
        for m in ds[ens_dim].values:
            results[m] = float(compute_geostrophic_imbalance(
                ds.sel({ens_dim: m}), area, phi_name=phi_name, u_names=u_names,
                v_names=v_names, level=level, level_dim=level_dim,
                lat_name=lat_name, lon_name=lon_name, lat_cutoff=lat_cutoff,
                earth_radius=earth_radius, omega=omega,
            ))
        return results

    if level_dim not in ds.dims:
        level_dim = _detect_level_dim(ds)

    levels = ds[level_dim].values
    idx = int(np.abs(levels - level).argmin())

    phi_var = (
        phi_name
        if phi_name in ds.data_vars
        else DatasetValidator.require_variable(ds, PHI_NAMES, "Geopotential")
    )
    phi = ds[phi_var]
    if level_dim in phi.dims:
        phi = phi.isel({level_dim: idx})

    u_var = DatasetValidator.require_variable(ds, u_names, "Zonal Wind (u)")
    v_var = DatasetValidator.require_variable(ds, v_names, "Meridional Wind (v)")

    u_actual = ds[u_var]
    v_actual = ds[v_var]
    if level_dim in u_actual.dims:
        u_actual = u_actual.isel({level_dim: idx})
        v_actual = v_actual.isel({level_dim: idx})

    if lat_name in phi.dims and lon_name in phi.dims:
        phi = phi.transpose(lat_name, lon_name)
        u_actual = u_actual.transpose(lat_name, lon_name)
        v_actual = v_actual.transpose(lat_name, lon_name)

    lat = ds[lat_name].values
    lon = ds[lon_name].values

    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)

    f_1d = 2.0 * omega * np.sin(lat_rad)
    f_2d = f_1d[:, None] * np.ones((1, len(lon)))
    cos_lat = np.cos(lat_rad)
    cos_2d = cos_lat[:, None] * np.ones((1, len(lon)))

    phi_np = phi.values
    dPhi_dphi = np.gradient(phi_np, lat_rad, axis=0, edge_order=2)
    phi_padded = np.pad(phi_np, pad_width=((0, 0), (1, 1)), mode="wrap")
    dlon_rad = np.abs(lon_rad[1] - lon_rad[0])
    dPhi_dlam = np.gradient(phi_padded, dlon_rad, axis=1)[:, 1:-1]

    with np.errstate(divide="ignore", invalid="ignore"):
        u_g = -dPhi_dphi / (f_2d * earth_radius)
        v_g = dPhi_dlam / (f_2d * earth_radius * cos_2d)

    du = u_actual.values - u_g
    dv = v_actual.values - v_g
    vec_err_sq = du**2 + dv**2

    lat_mask = (np.abs(lat) >= lat_cutoff) & (np.abs(lat) < 89.9)
    mask_2d = lat_mask[:, None] * np.ones((1, len(lon)), dtype=bool)
    vec_err_sq = np.where(mask_2d, np.nan_to_num(vec_err_sq, nan=0.0), 0.0)

    area_vals = area.values
    if area_vals.shape != vec_err_sq.shape:
        area_da = get_grid_cell_area(ds)
        if lat_name in area_da.dims and lon_name in area_da.dims:
            area_da = area_da.transpose(lat_name, lon_name)
        area_vals = area_da.values

    w = area_vals.copy()
    w[~mask_2d] = 0.0
    w_sum = w.sum()
    if w_sum == 0:
        return float("nan")

    w_norm = w / w_sum
    mse = float((w_norm * vec_err_sq).sum())
    return float(np.sqrt(mse))


# ============================================================================
# Metric 7 — Environmental Lapse Rate Wasserstein Distance
# ============================================================================

def compute_lapse_rate_wasserstein(
    ds_pred: xr.Dataset,
    ds_ref: xr.Dataset,
    area: xr.DataArray,
    t_name: Optional[str] = None,
    phi_name: Optional[str] = None,
    level_dim_pred: Optional[str] = None,
    level_dim_ref: Optional[str] = None,
) -> Dict[str, float]:
    """Compute 1D Wasserstein distance of lapse rate distribution for geographic bands.

    Args:
        ds_pred: Model prediction Dataset.
        ds_ref: Ground-truth reference Dataset.
        area: Grid cell area DataArray (m²).
        t_name: Temperature variable name.
        phi_name: Geopotential variable name.
        level_dim_pred: Pressure level dimension name for model prediction.
        level_dim_ref: Pressure level dimension name for reference.

    Returns:
        Dictionary mapping regional band keys to 1-Wasserstein distances (K/km).
    """
    ens_dim_p = _detect_ensemble_dim(ds_pred)
    if ens_dim_p is not None and ens_dim_p in ds_pred.dims:
        ds_pred = ds_pred.mean(dim=ens_dim_p)

    t_name_p = t_name or _find_var(ds_pred, T_NAMES)
    phi_name_p = phi_name or _find_var(ds_pred, PHI_NAMES)
    t_name_r = t_name or _find_var(ds_ref, T_NAMES)
    phi_name_r = phi_name or _find_var(ds_ref, PHI_NAMES)

    ld_p = level_dim_pred or _detect_level_dim(ds_pred)
    ld_r = level_dim_ref or _detect_level_dim(ds_ref)

    def _calc_gamma(ds: xr.Dataset, t_var: str, phi_var: str, ld: str) -> xr.DataArray:
        t_500 = ds[t_var].sel({ld: 500})
        t_850 = ds[t_var].sel({ld: 850})
        phi_500 = ds[phi_var].sel({ld: 500})
        phi_850 = ds[phi_var].sel({ld: 850})
        return -GRAVITY * (t_500 - t_850) / (phi_500 - phi_850) * 1000.0

    gamma_p = _calc_gamma(ds_pred, t_name_p, phi_name_p, ld_p)
    gamma_r = _calc_gamma(ds_ref, t_name_r, phi_name_r, ld_r)

    lat_p = ds_pred.latitude
    lat_r = ds_ref.latitude

    bands = {
        "tropics": ((lat_p >= -30) & (lat_p <= 30), (lat_r >= -30) & (lat_r <= 30)),
        "nh_mid": ((lat_p > 30) & (lat_p <= 60), (lat_r > 30) & (lat_r <= 60)),
        "sh_mid": ((lat_p >= -60) & (lat_p < -30), (lat_r >= -60) & (lat_r < -30)),
    }

    area_p = get_grid_cell_area(ds_pred)
    area_r = get_grid_cell_area(ds_ref)

    results = {}
    for band_name, (mask_p, mask_r) in bands.items():
        gp_band = gamma_p.where(mask_p, drop=True).values.flatten()
        gr_band = gamma_r.where(mask_r, drop=True).values.flatten()

        area_p_band = area_p.where(mask_p, drop=True).values.flatten()
        area_r_band = area_r.where(mask_r, drop=True).values.flatten()

        valid_p = ~np.isnan(gp_band) & ~np.isnan(area_p_band)
        valid_r = ~np.isnan(gr_band) & ~np.isnan(area_r_band)

        w1 = wasserstein_distance(
            u_values=gp_band[valid_p],
            v_values=gr_band[valid_r],
            u_weights=area_p_band[valid_p],
            v_weights=area_r_band[valid_r],
        )
        results[f"lapse_rate_w1_{band_name}"] = float(w1)

    return results


# ============================================================================
# Drift & Conservation Helpers
# ============================================================================

def compute_drift_slope(
    hours: np.ndarray,
    values: np.ndarray,
) -> float:
    """Compute linear regression slope expressed as change per day.

    Args:
        hours: 1D numpy array of forecast lead times in hours.
        values: 1D numpy array of evaluated metric values.

    Returns:
        Linear drift slope value per day. Returns nan if fewer than 2 valid points.
    """
    days = np.asarray(hours, dtype=np.float64) / 24.0
    vals = np.asarray(values, dtype=np.float64)

    mask = np.isfinite(days) & np.isfinite(vals)
    if mask.sum() < 2:
        return float("nan")

    result = linregress(days[mask], vals[mask])
    return float(result.slope)


def compute_drift_percentages(
    hours_model: np.ndarray,
    dry_model: np.ndarray,
    water_model: np.ndarray,
    energy_model: np.ndarray,
    hours_ref: np.ndarray,
    water_ref: np.ndarray,
    energy_ref: np.ndarray,
) -> Dict[str, float]:
    """Compute percentage drift rates per day for mass, water, and energy.

    Args:
        hours_model: Model forecast hours array.
        dry_model: Model global dry air mass values array (Eg).
        water_model: Model global water mass values array (kg).
        energy_model: Model global total energy values array (J).
        hours_ref: Reference ground-truth hours array.
        water_ref: Reference ground-truth water mass values array (kg).
        energy_ref: Reference ground-truth total energy values array (J).

    Returns:
        Dictionary mapping drift metric names to relative %/day drift rates.
    """
    slope_dry = compute_drift_slope(hours_model, dry_model)
    slope_water = compute_drift_slope(hours_model, water_model)
    slope_energy = compute_drift_slope(hours_model, energy_model)

    slope_water_ref = compute_drift_slope(hours_ref, water_ref)
    slope_energy_ref = compute_drift_slope(hours_ref, energy_ref)

    dry_ref_0 = float(dry_model[0]) if len(dry_model) > 0 else 0.0
    water_model_0 = float(water_model[0]) if len(water_model) > 0 else 0.0
    water_ref_0 = float(water_ref[0]) if len(water_ref) > 0 else 0.0
    energy_model_0 = float(energy_model[0]) if len(energy_model) > 0 else 0.0
    energy_ref_0 = float(energy_ref[0]) if len(energy_ref) > 0 else 0.0

    def _safe_rel_pct(slope: float, ref: float) -> float:
        if ref != 0 and np.isfinite(slope) and np.isfinite(ref):
            return (slope / ref) * 100.0
        return float("nan")

    return {
        "dry_mass_drift_pct_per_day": _safe_rel_pct(slope_dry, dry_ref_0),
        "water_mass_drift_pct_per_day": _safe_rel_pct(slope_water, water_model_0)
        - _safe_rel_pct(slope_water_ref, water_ref_0),
        "total_energy_drift_pct_per_day": _safe_rel_pct(slope_energy, energy_model_0)
        - _safe_rel_pct(slope_energy_ref, energy_ref_0),
    }


def compute_conservation_scalars(
    ds: xr.Dataset,
    ps: xr.DataArray,
    area: xr.DataArray,
    z_sfc: xr.DataArray,
    level_dim: str = "level",
    levels: Optional[np.ndarray] = None,
) -> Union[Tuple[float, float, float], Dict[Any, Tuple[float, float, float]]]:
    """Compute the three conservation scalars (dry mass, water mass, total energy).

    Args:
        ds: Input Dataset containing atmospheric state variables.
        ps: Surface pressure DataArray.
        area: Grid cell area DataArray (m²).
        z_sfc: Surface geopotential DataArray.
        level_dim: Pressure level dimension name.
        levels: Optional array of pressure levels in hPa.

    Returns:
        Tuple of (dry_mass_Eg, water_mass_kg, total_energy_J) for deterministic inputs,
        or Dict[member, Tuple[dry, water, energy]] for ensemble datasets.
    """
    ens_dim = _detect_ensemble_dim(ds)
    if ens_dim is not None and ens_dim in ds.dims:
        results = {}
        for m in ds[ens_dim].values:
            ds_m = ds.sel({ens_dim: m})
            ps_m = ps.sel({ens_dim: m}) if ens_dim in ps.dims else ps
            results[m] = compute_conservation_scalars(
                ds_m, ps_m, area, z_sfc=z_sfc, level_dim=level_dim, levels=levels
            )
        return results

    if level_dim not in ds.dims:
        level_dim = _detect_level_dim(ds)
    if levels is None and level_dim in ds.coords:
        levels = ds[level_dim].values

    q_name = _find_var(ds, Q_NAMES)
    t_name = _find_var(ds, T_NAMES) or "temperature"
    has_q = q_name is not None

    if has_q:
        tcwv = _compute_tcwv(ds, ps, q_name=q_name, level_dim=level_dim, levels=levels)
        dry = float(compute_dry_air_mass(
            ds, ps, area, q_name=q_name, level_dim=level_dim, levels=levels, tcwv=tcwv
        ))
        water = float(compute_water_mass(
            ds, ps, area, q_name=q_name, level_dim=level_dim, levels=levels, tcwv=tcwv
        ))
        try:
            energy = float(compute_total_energy(
                ds, ps, area, z_sfc=z_sfc, t_name=t_name, q_name=q_name,
                level_dim=level_dim, levels=levels
            ))
        except (ValueError, KeyError, AttributeError):
            energy = float("nan")
    else:
        dry = float("nan")
        water = float("nan")
        energy = float("nan")

    return dry, water, energy


def compute_pure_tcwv(
    ds: xr.Dataset,
    q_name: str = "q",
    level_dim: str = "level",
) -> xr.DataArray:
    """Integrate specific humidity purely over fixed pressure levels without ps masking.

    Args:
        ds: Input Dataset containing specific humidity.
        q_name: Variable name for specific humidity.
        level_dim: Pressure level dimension name.

    Returns:
        2D DataArray containing unmasked TCWV values in kg/m².

    Raises:
        KeyError: If specific humidity variable is not found in dataset.
    """
    if q_name not in ds.data_vars:
        q_name = DatasetValidator.require_variable(ds, Q_NAMES, "Specific Humidity")

    q = ds[q_name]
    levels = ds[level_dim].values
    levels_pa = levels.astype(np.float64) * 100.0
    sort_idx = np.argsort(levels_pa)
    levels_sorted = levels_pa[sort_idx]
    q_sorted = q.transpose(level_dim, ...).values[sort_idx]

    col = np.zeros_like(q_sorted[0], dtype=np.float64)
    for k in range(len(levels_sorted) - 1):
        dp = levels_sorted[k + 1] - levels_sorted[k]
        q_avg = 0.5 * (q_sorted[k] + q_sorted[k + 1])
        col += q_avg * dp

    lat_dim = [d for d in q.dims if d != level_dim][0]
    lon_dim = [d for d in q.dims if d != level_dim][1]

    return xr.DataArray(
        col / GRAVITY,
        dims=[lat_dim, lon_dim],
        coords={lat_dim: q[lat_dim], lon_dim: q[lon_dim]},
        name="tcwv_pure",
        attrs={
            "units": "kg/m²",
            "long_name": "TCWV (fixed pressure levels, no surface pressure)",
        },
    )
