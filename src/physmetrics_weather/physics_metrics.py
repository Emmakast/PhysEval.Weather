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
# Default Evaluation Metric Levels & Threshold Constants
# ============================================================================

DEFAULT_SPECTRUM_LEVEL: float = 500.0        # hPa – default pressure level for spectra
DEFAULT_KE_850_LEVEL: float = 850.0          # hPa – secondary level for boundary layer KE
HYDROSTATIC_P_TOP: float = 500.0            # hPa – upper level for hydrostatic error
HYDROSTATIC_P_BOT: float = 850.0            # hPa – lower level for hydrostatic error
GEOSTROPHIC_LEVEL: float = 500.0            # hPa – pressure level for geostrophic balance
GEOSTROPHIC_LAT_CUTOFF: float = 20.0        # deg – equatorial exclusion zone latitude
LAPSE_RATE_P_TOP: float = 500.0             # hPa – upper pressure level for lapse rate
LAPSE_RATE_P_BOT: float = 850.0             # hPa – lower pressure level for lapse rate
EFFECTIVE_RES_THRESHOLD: float = 0.5        # spectral power ratio cutoff threshold
EFFECTIVE_RES_K_MIN: int = 10               # minimum wavenumber for effective resolution


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
    field: xr.DataArray,
    levels_hpa: np.ndarray,
    ps: xr.DataArray,
    gravity: float = GRAVITY,
    level_dim: str = "level",
) -> xr.DataArray:
    levels_pa = np.asarray(levels_hpa, dtype=np.float64) * 100.0
    sort_idx = np.argsort(levels_pa)
    p = xr.DataArray(levels_pa[sort_idx], dims=[level_dim], coords={level_dim: levels_pa[sort_idx]})
    f = field.isel({level_dim: sort_idx})
    f = f.assign_coords({level_dim: p})
    
    p_top = p.isel({level_dim: slice(0, -1)})
    p_bot = p.isel({level_dim: slice(1, None)})
    p_bot = p_bot.assign_coords({level_dim: p_top[level_dim]})
    
    f_top = f.isel({level_dim: slice(0, -1)})
    f_bot = f.isel({level_dim: slice(1, None)})
    f_bot = f_bot.assign_coords({level_dim: p_top[level_dim]})
    
    f_avg = 0.5 * (f_top + f_bot)
    
    eff_top = xr.where(ps < p_top, ps, p_top)
    eff_bot = xr.where(ps < p_bot, ps, p_bot)
    dp = xr.where((eff_bot - eff_top) > 0, eff_bot - eff_top, 0.0)
    
    col = (f_avg * dp).sum(dim=level_dim)
    
    dp_top = xr.where(ps < p[0], ps, p[0]) - 0.0
    col += f.isel({level_dim: 0}).drop_vars(level_dim) * dp_top
    
    dp_bot = xr.where(ps > p[-1], ps - p[-1], 0.0)
    col += f.isel({level_dim: -1}).drop_vars(level_dim) * dp_bot
    
    return col / gravity


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
        N-D DataArray containing TCWV values in kg/m².
    """
    if q_name not in ds.data_vars:
        q_name = DatasetValidator.require_variable(ds, Q_NAMES, "Specific Humidity")

    q = ds[q_name]
    if levels is None:
        levels = ds[level_dim].values

    tcwv = _integrate_column(q, levels, ps, level_dim=level_dim)

    tcwv.name = "tcwv"
    tcwv.attrs = {"units": "kg/m²", "long_name": "Total Column Water Vapour"}
    return tcwv


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
) -> Union[xr.DataArray, float, Dict[Any, float]]:
    if tcwv is None:
        tcwv = _compute_tcwv(ds, ps, q_name=q_name, level_dim=level_dim, levels=levels)
    col_dry = ps / GRAVITY - tcwv
    lat_dim = [d for d in col_dry.dims if "lat" in d.lower()][0]
    lon_dim = [d for d in col_dry.dims if "lon" in d.lower()][0]
    
    area_aligned = xr.DataArray(area.values, dims=[lat_dim, lon_dim], coords={lat_dim: col_dry[lat_dim], lon_dim: col_dry[lon_dim]})
    dry_mass_kg = (area_aligned * col_dry).sum(dim=[lat_dim, lon_dim], skipna=True, min_count=1)
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
) -> Union[xr.DataArray, float, Dict[Any, float]]:
    if tcwv is None:
        tcwv = _compute_tcwv(ds, ps, q_name=q_name, level_dim=level_dim, levels=levels)
    lat_dim = [d for d in tcwv.dims if "lat" in d.lower()][0]
    lon_dim = [d for d in tcwv.dims if "lon" in d.lower()][0]
    
    area_aligned = xr.DataArray(area.values, dims=[lat_dim, lon_dim], coords={lat_dim: tcwv[lat_dim], lon_dim: tcwv[lon_dim]})
    return (area_aligned * tcwv).sum(dim=[lat_dim, lon_dim], skipna=True, min_count=1)



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
) -> Union[xr.DataArray, float, Dict[Any, float]]:
    if levels is None:
        levels = ds[level_dim].values
    t_var = t_name if t_name in ds.data_vars else DatasetValidator.require_variable(ds, T_NAMES, "Temperature")
    q_var = q_name if q_name in ds.data_vars else DatasetValidator.require_variable(ds, Q_NAMES, "Specific Humidity")
    T = ds[t_var]
    q = ds[q_var]
    c_p = c_pd * (1.0 - q) + c_pv * q
    ref_var = ds[t_var]
    lat_dim_ = [d for d in ref_var.dims if "lat" in d.lower()][0]
    lon_dim_ = [d for d in ref_var.dims if "lon" in d.lower()][0]
    
    lat_diff = abs(z_sfc.sizes[lat_dim_] - ref_var.sizes[lat_dim_])
    lon_diff = abs(z_sfc.sizes[lon_dim_] - ref_var.sizes[lon_dim_])
    if lat_diff <= 1 and lon_diff == 0:
        z_aligned = z_sfc.reindex({lat_dim_: ref_var[lat_dim_], lon_dim_: ref_var[lon_dim_]}, method="nearest")
    else:
        raise ValueError(f"Grid mismatch too large: z_sfc={z_sfc.shape}, model={ref_var.shape}.")
        
    u_var = DatasetValidator.require_variable(ds, u_names, "Zonal Wind (u)")
    v_var = DatasetValidator.require_variable(ds, v_names, "Meridional Wind (v)")
    u_val = ds[u_var]
    v_val = ds[v_var]
    wspd_sq = u_val**2 + v_val**2
    
    energy_density = c_p * T + z_aligned + l_v * q + 0.5 * wspd_sq
    col_energy = _integrate_column(energy_density, levels, ps, level_dim=level_dim)
    
    area_aligned = xr.DataArray(area.values, dims=[lat_dim_, lon_dim_], coords={lat_dim_: col_energy[lat_dim_], lon_dim_: col_energy[lon_dim_]})
    return (area_aligned * col_energy).sum(dim=[lat_dim_, lon_dim_], skipna=True, min_count=1)



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
    
    u = np.nan_to_num(u, nan=0.0)
    v = np.nan_to_num(v, nan=0.0)

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

    field = np.nan_to_num(field, nan=0.0)

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
    level: float = DEFAULT_SPECTRUM_LEVEL,
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
    level: float = DEFAULT_SPECTRUM_LEVEL,
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
    return _scalar_spectrum_spharm(q.values)


VAR_ALIASES: Dict[str, Tuple[str, ...]] = {
    "Q": Q_NAMES,
    "T": T_NAMES,
    "Z": PHI_NAMES,
    "PHI": PHI_NAMES,
    "SP": SP_NAMES,
    "MSL": MSL_NAMES,
    "T2M": T2M_NAMES,
}


def compute_scalar_spectrum(
    ds: xr.Dataset,
    var_name: str,
    level: float = DEFAULT_SPECTRUM_LEVEL,
    level_dim: str = "level",
) -> Union[Tuple[np.ndarray, np.ndarray], Dict[Any, Tuple[np.ndarray, np.ndarray]]]:
    """Compute spherical-harmonic power spectrum S(l) of any scalar variable at a level.

    Args:
        ds: Input xarray Dataset.
        var_name: Name or variable alias candidate string (e.g. 'T', 'Q', 'Z', 'temperature').
        level: Target pressure level in hPa (default: 500.0).
        level_dim: Pressure level dimension name.

    Returns:
        Tuple of (wavenumber_array, power_spectrum_array) for deterministic input,
        or Dict[member_id, Tuple[wavenumber, power]] for ensemble datasets.
    """
    ens_dim = _detect_ensemble_dim(ds)
    if ens_dim is not None and ens_dim in ds.dims:
        results = {}
        for m in ds[ens_dim].values:
            results[m] = compute_scalar_spectrum(
                ds.sel({ens_dim: m}), var_name=var_name, level=level, level_dim=level_dim
            )
        return results

    if level_dim not in ds.dims:
        level_dim = _detect_level_dim(ds)

    var_key = var_name.upper()
    actual_var = None
    if var_key in VAR_ALIASES:
        actual_var = _find_var(ds, VAR_ALIASES[var_key])
    if actual_var is None and var_name in ds.data_vars:
        actual_var = var_name
    if actual_var is None:
        actual_var = DatasetValidator.require_variable(ds, (var_name,), f"Scalar ({var_name})")

    da = ds[actual_var]
    if level_dim in da.dims:
        lvls = ds[level_dim].values
        idx = int(np.abs(lvls - level).argmin())
        da = da.isel({level_dim: idx})

    lat_dims = [d for d in da.dims if "lat" in d.lower()]
    lon_dims = [d for d in da.dims if "lon" in d.lower()]
    if lat_dims and lon_dims:
        da = da.transpose(..., lat_dims[0], lon_dims[0])

    return _scalar_spectrum_spharm(da.values)


def _find_effective_resolution(
    k: np.ndarray,
    e_pred: np.ndarray,
    e_true: np.ndarray,
    threshold: float = EFFECTIVE_RES_THRESHOLD,
    k_min: int = EFFECTIVE_RES_K_MIN,
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
    p_top: float = HYDROSTATIC_P_TOP,
    p_bot: float = HYDROSTATIC_P_BOT,
    r_dry: float = R_DRY,
) -> Union[xr.DataArray, float, Dict[Any, float]]:
    if level_dim not in ds.dims:
        level_dim = _detect_level_dim(ds)
    levels = ds[level_dim].values
    phi_var = phi_name or DatasetValidator.require_variable(ds, PHI_NAMES, "Geopotential")
    t_var = t_name or DatasetValidator.require_variable(ds, T_NAMES, "Temperature")
    q_var = q_name if q_name in ds.data_vars else _find_var(ds, Q_NAMES)
    
    def _sel_level(var: str, p: float) -> xr.DataArray:
        idx = int(np.abs(levels - p).argmin())
        return ds[var].isel({level_dim: idx}).drop_vars(level_dim, errors="ignore")
        
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
    if lat_dim_e and lon_dim_e and list(error.dims)[-2:] != [lat_dim_e, lon_dim_e]:
        error = error.transpose(..., lat_dim_e, lon_dim_e)
        
    area_aligned = xr.DataArray(area.values, dims=[lat_dim_e, lon_dim_e], coords={lat_dim_e: error[lat_dim_e], lon_dim_e: error[lon_dim_e]})
    weights = area_aligned / area_aligned.sum(dim=[lat_dim_e, lon_dim_e])
    mse = (weights * error**2).sum(dim=[lat_dim_e, lon_dim_e], skipna=True, min_count=1)
    return mse ** 0.5



# ============================================================================
# Metric 6 — Geostrophic Balance RMSE
# ============================================================================

def compute_geostrophic_imbalance(
    ds: xr.Dataset,
    area: xr.DataArray,
    phi_name: Optional[str] = None,
    u_names: Tuple[str, ...] = U_NAMES,
    v_names: Tuple[str, ...] = V_NAMES,
    level: float = GEOSTROPHIC_LEVEL,
    level_dim: str = "level",
    lat_name: str = "latitude",
    lon_name: str = "longitude",
    lat_cutoff: float = GEOSTROPHIC_LAT_CUTOFF,
    earth_radius: float = EARTH_RADIUS,
    omega: float = OMEGA,
) -> Union[xr.DataArray, float, Dict[Any, float]]:
    if level_dim not in ds.dims:
        level_dim = _detect_level_dim(ds)
    levels = ds[level_dim].values
    idx = int(np.abs(levels - level).argmin())
    
    phi_var = phi_name if phi_name in ds.data_vars else DatasetValidator.require_variable(ds, PHI_NAMES, "Geopotential")
    phi = ds[phi_var]
    if level_dim in phi.dims:
        phi = phi.isel({level_dim: idx}).drop_vars(level_dim, errors="ignore")
        
    u_var = DatasetValidator.require_variable(ds, u_names, "Zonal Wind (u)")
    v_var = DatasetValidator.require_variable(ds, v_names, "Meridional Wind (v)")
    u_actual = ds[u_var]
    v_actual = ds[v_var]
    if level_dim in u_actual.dims:
        u_actual = u_actual.isel({level_dim: idx}).drop_vars(level_dim, errors="ignore")
        v_actual = v_actual.isel({level_dim: idx}).drop_vars(level_dim, errors="ignore")
        
    if lat_name in phi.dims and lon_name in phi.dims:
        if list(phi.dims)[-2:] != [lat_name, lon_name]:
            phi = phi.transpose(..., lat_name, lon_name)
            u_actual = u_actual.transpose(..., lat_name, lon_name)
            v_actual = v_actual.transpose(..., lat_name, lon_name)
            
    lat = ds[lat_name]
    lon = ds[lon_name]
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    
    f = 2.0 * omega * np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    
    f_safe = xr.where(f == 0.0, 1e-10, f)
    cos_lat_safe = xr.where(cos_lat == 0.0, 1e-10, cos_lat)

    phi = phi.chunk({lat_name: -1, lon_name: -1})
    dPhi_dphi = phi.differentiate(lat_name) * (180.0 / np.pi)
    
    phi_east = phi.shift({lon_name: -1})
    phi_west = phi.shift({lon_name: 1})

    phi_east = xr.where(phi_east.isnull(), phi.isel({lon_name: 0}), phi_east)
    phi_west = xr.where(phi_west.isnull(), phi.isel({lon_name: -1}), phi_west)
    dlon_rad = np.deg2rad(np.abs(lon.values[1] - lon.values[0]))
    dPhi_dlam = (phi_east - phi_west) / (2.0 * dlon_rad)
    
    u_g = -dPhi_dphi / (f_safe * earth_radius)
    v_g = dPhi_dlam / (f_safe * earth_radius * cos_lat_safe)
    
    du = u_actual - u_g
    dv = v_actual - v_g
    vec_err_sq = du**2 + dv**2
    
    lat_mask = (np.abs(lat) >= lat_cutoff) & (np.abs(lat) < 89.9)
    vec_err_sq = xr.where(lat_mask, vec_err_sq.fillna(0.0), 0.0)
    
    area_aligned = xr.DataArray(area.values, dims=[lat_name, lon_name], coords={lat_name: ds[lat_name], lon_name: ds[lon_name]})
    w = xr.where(lat_mask, area_aligned, 0.0)
    w_sum = w.sum(dim=[lat_name, lon_name])
    w_norm = w / w_sum
    
    mse = (w_norm * vec_err_sq).sum(dim=[lat_name, lon_name], skipna=True, min_count=1)
    return mse ** 0.5



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
    p_top: float = LAPSE_RATE_P_TOP,
    p_bot: float = LAPSE_RATE_P_BOT,
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
        p_top: Upper pressure level in hPa (default: LAPSE_RATE_P_TOP = 500.0).
        p_bot: Lower pressure level in hPa (default: LAPSE_RATE_P_BOT = 850.0).

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
        levels = ds[ld].values
        idx_top = int(np.abs(levels - p_top).argmin())
        idx_bot = int(np.abs(levels - p_bot).argmin())
        t_top = ds[t_var].isel({ld: idx_top})
        t_bot = ds[t_var].isel({ld: idx_bot})
        phi_top = ds[phi_var].isel({ld: idx_top})
        phi_bot = ds[phi_var].isel({ld: idx_bot})
        return -GRAVITY * (t_top - t_bot) / (phi_top - phi_bot) * 1000.0

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
) -> Union[Tuple[xr.DataArray, xr.DataArray, xr.DataArray], Tuple[float, float, float], Dict[Any, Tuple[float, float, float]]]:
    if level_dim not in ds.dims:
        level_dim = _detect_level_dim(ds)
    if levels is None and level_dim in ds.coords:
        levels = ds[level_dim].values
    q_name = _find_var(ds, Q_NAMES)
    t_name = _find_var(ds, T_NAMES) or "temperature"
    has_q = q_name is not None
    if has_q:
        tcwv = _compute_tcwv(ds, ps, q_name=q_name, level_dim=level_dim, levels=levels)
        dry = compute_dry_air_mass(ds, ps, area, q_name=q_name, level_dim=level_dim, levels=levels, tcwv=tcwv)
        water = compute_water_mass(ds, ps, area, q_name=q_name, level_dim=level_dim, levels=levels, tcwv=tcwv)
        try:
            energy = compute_total_energy(ds, ps, area, z_sfc=z_sfc, t_name=t_name, q_name=q_name, level_dim=level_dim, levels=levels)
        except (ValueError, KeyError, AttributeError):
            energy = xr.full_like(dry, np.nan)
    else:
        dry = xr.DataArray(np.nan)
        water = xr.DataArray(np.nan)
        energy = xr.DataArray(np.nan)
    return dry, water, energy



def compute_pure_tcwv(
    ds: xr.Dataset,
    q_name: str = "q",
    level_dim: str = "level",
) -> xr.DataArray:
    if q_name not in ds.data_vars:
        q_name = DatasetValidator.require_variable(ds, Q_NAMES, "Specific Humidity")
    q = ds[q_name]
    levels = ds[level_dim].values
    levels_pa = levels.astype(np.float64) * 100.0
    
    sort_idx = np.argsort(levels_pa)
    p = xr.DataArray(levels_pa[sort_idx], dims=[level_dim], coords={level_dim: levels_pa[sort_idx]})
    q_sorted = q.isel({level_dim: sort_idx}).assign_coords({level_dim: p})
    
    p_top = p.isel({level_dim: slice(0, -1)})
    p_bot = p.isel({level_dim: slice(1, None)})
    p_bot = p_bot.assign_coords({level_dim: p_top[level_dim]})
    
    q_top = q_sorted.isel({level_dim: slice(0, -1)})
    q_bot = q_sorted.isel({level_dim: slice(1, None)})
    q_bot = q_bot.assign_coords({level_dim: p_top[level_dim]})
    
    q_avg = 0.5 * (q_top + q_bot)
    dp = p_bot - p_top
    
    col = (q_avg * dp).sum(dim=level_dim)
    
    col.name = "tcwv_pure"
    col.attrs = {
        "units": "kg/m²",
        "long_name": "TCWV (fixed pressure levels, no surface pressure)",
    }
    return col / GRAVITY


