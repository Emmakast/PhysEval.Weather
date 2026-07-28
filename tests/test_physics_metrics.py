"""Unit tests for physics_metrics.py using pytest and mock xarray datasets."""

import numpy as np
import pytest
import xarray as xr

from physmetrics_weather.physics_metrics import (
    _detect_ensemble_dim,
    _detect_level_dim,
    _integrate_column,
    compute_conservation_scalars,
    compute_drift_percentages,
    compute_drift_slope,
    compute_dry_air_mass,
    compute_geostrophic_imbalance,
    compute_hydrostatic_imbalance,
    compute_ke_spectrum,
    compute_lapse_rate_wasserstein,
    compute_pure_tcwv,
    compute_q_spectrum,
    compute_scalar_spectrum,
    compute_spectral_scores,
    compute_total_energy,
    compute_water_mass,
    derive_surface_pressure,
    get_grid_cell_area,
)


@pytest.fixture
def mock_grid_coords():
    """Create standard Driscoll-Healy latitude/longitude grid coordinates (32x64)."""
    lats = np.linspace(90, -90, 32)
    lons = np.linspace(0, 354.375, 64)
    levels = np.array([1000.0, 850.0, 500.0, 250.0, 100.0])
    return lats, lons, levels


@pytest.fixture
def mock_deterministic_ds(mock_grid_coords):
    """Create a deterministic mock xarray dataset without ensemble dimension."""
    lats, lons, levels = mock_grid_coords
    nlat, nlon, nlev = len(lats), len(lons), len(levels)

    np.random.seed(42)
    shape_3d = (nlev, nlat, nlon)
    shape_2d = (nlat, nlon)

    temp = 280.0 + np.random.randn(*shape_3d) * 5.0
    q = np.abs(np.random.randn(*shape_3d) * 0.005)
    u = np.random.randn(*shape_3d) * 10.0
    v = np.random.randn(*shape_3d) * 10.0
    phi = (
        np.broadcast_to(levels[:, None, None] * 100.0, shape_3d)
        + np.random.randn(*shape_3d) * 10.0
    )
    msl = 101325.0 + np.random.randn(*shape_2d) * 500.0
    sp = 100000.0 + np.random.randn(*shape_2d) * 1000.0

    ds = xr.Dataset(
        data_vars={
            "temperature": (["level", "latitude", "longitude"], temp),
            "specific_humidity": (["level", "latitude", "longitude"], q),
            "u_component_of_wind": (["level", "latitude", "longitude"], u),
            "v_component_of_wind": (["level", "latitude", "longitude"], v),
            "geopotential": (["level", "latitude", "longitude"], phi),
            "mean_sea_level_pressure": (["latitude", "longitude"], msl),
            "surface_pressure": (["latitude", "longitude"], sp),
        },
        coords={
            "latitude": lats,
            "longitude": lons,
            "level": levels,
        },
    )
    return ds


@pytest.fixture
def mock_ensemble_ds(mock_deterministic_ds):
    """Create an ensemble mock xarray dataset with an 'ens' dimension (3 members)."""
    ds = mock_deterministic_ds
    members = [0, 1, 2]

    data_vars = {}
    for var_name, da in ds.data_vars.items():
        shape = (len(members),) + da.shape
        var_data = np.stack([da.values + i * 0.1 for i in members], axis=0)
        dims = ["ens"] + list(da.dims)
        data_vars[var_name] = (dims, var_data)

    coords = dict(ds.coords)
    coords["ens"] = members

    return xr.Dataset(data_vars=data_vars, coords=coords)


@pytest.fixture
def mock_static_ds(mock_grid_coords):
    """Create a static dataset containing surface geopotential."""
    lats, lons, _ = mock_grid_coords
    shape_2d = (len(lats), len(lons))
    z_sfc = np.clip(np.random.randn(*shape_2d) * 500.0, 0, None)

    return xr.Dataset(
        data_vars={"geopotential_at_surface": (["latitude", "longitude"], z_sfc)},
        coords={"latitude": lats, "longitude": lons},
    )


# ============================================================================
# Unit Tests
# ============================================================================

def test_detect_ensemble_dim(mock_deterministic_ds, mock_ensemble_ds):
    """Test auto-detection of ensemble dimension."""
    assert _detect_ensemble_dim(mock_deterministic_ds) is None
    assert _detect_ensemble_dim(mock_ensemble_ds) == "ens"


def test_detect_level_dim(mock_deterministic_ds):
    """Test auto-detection of pressure level dimension."""
    assert _detect_level_dim(mock_deterministic_ds) == "level"


def test_get_grid_cell_area(mock_deterministic_ds):
    """Test computation of grid cell areas."""
    area = get_grid_cell_area(mock_deterministic_ds)
    assert isinstance(area, xr.DataArray)
    assert area.shape == (32, 64)
    assert (area.values > 0).all()
    # Total surface area of Earth ≈ 5.1e14 m²
    assert np.isclose(float(area.sum()), 4 * np.pi * (6.371e6) ** 2, rtol=1e-2)


def test_derive_surface_pressure(mock_deterministic_ds, mock_static_ds):
    """Test surface pressure derivation from MSL and surface geopotential."""
    sp = derive_surface_pressure(mock_deterministic_ds, mock_static_ds)
    assert isinstance(sp, xr.DataArray)
    assert sp.shape == (32, 64)
    assert (sp.values > 50000.0).all()


def test_integrate_column(mock_grid_coords):
    """Test 3D column integration helper."""
    lats, lons, levels = mock_grid_coords
    nlat, nlon, nlev = len(lats), len(lons), len(levels)
    field_3d = np.ones((nlev, nlat, nlon))
    ps_2d = np.full((nlat, nlon), 100000.0)

    integrated = _integrate_column(field_3d, levels, ps_2d)
    assert integrated.shape == (nlat, nlon)
    assert (integrated > 0).all()


def test_compute_dry_air_mass_deterministic(mock_deterministic_ds):
    """Test dry air mass calculation for deterministic dataset."""
    area = get_grid_cell_area(mock_deterministic_ds)
    ps = mock_deterministic_ds["surface_pressure"]

    dry_mass = compute_dry_air_mass(mock_deterministic_ds, ps, area)
    assert isinstance(dry_mass, float)
    assert 4.0 < dry_mass < 6.0  # Earth dry air mass ≈ 5.1 Eg


def test_compute_dry_air_mass_ensemble(mock_ensemble_ds):
    """Test dry air mass calculation for ensemble dataset."""
    area = get_grid_cell_area(mock_ensemble_ds)
    ps = mock_ensemble_ds["surface_pressure"]

    dry_mass = compute_dry_air_mass(mock_ensemble_ds, ps, area)
    assert isinstance(dry_mass, dict)
    assert len(dry_mass) == 3
    for m, val in dry_mass.items():
        assert isinstance(val, float)
        assert 4.0 < val < 6.0


def test_compute_water_mass(mock_deterministic_ds, mock_ensemble_ds):
    """Test global water mass calculation for deterministic and ensemble datasets."""
    area = get_grid_cell_area(mock_deterministic_ds)
    ps = mock_deterministic_ds["surface_pressure"]

    water_det = compute_water_mass(mock_deterministic_ds, ps, area)
    assert isinstance(water_det, float)
    assert water_det > 0

    water_ens = compute_water_mass(mock_ensemble_ds, mock_ensemble_ds["surface_pressure"], area)
    assert isinstance(water_ens, dict)
    assert len(water_ens) == 3


def test_compute_total_energy(mock_deterministic_ds, mock_static_ds, mock_ensemble_ds):
    """Test total energy calculation for deterministic and ensemble datasets."""
    area = get_grid_cell_area(mock_deterministic_ds)
    ps = mock_deterministic_ds["surface_pressure"]
    z_sfc = mock_static_ds["geopotential_at_surface"]

    te_det = compute_total_energy(mock_deterministic_ds, ps, area, z_sfc=z_sfc)
    assert isinstance(te_det, float)
    assert te_det > 0

    te_ens = compute_total_energy(
        mock_ensemble_ds, mock_ensemble_ds["surface_pressure"], area, z_sfc=z_sfc
    )
    assert isinstance(te_ens, dict)
    assert len(te_ens) == 3


def test_compute_ke_spectrum(mock_deterministic_ds, mock_ensemble_ds):
    """Test KE spectrum calculation."""
    res_det = compute_ke_spectrum(mock_deterministic_ds, level=500.0)
    assert isinstance(res_det, tuple)
    k, e = res_det
    assert len(k) == len(e)

    res_ens = compute_ke_spectrum(mock_ensemble_ds, level=500.0)
    assert isinstance(res_ens, dict)
    assert len(res_ens) == 3


def test_compute_q_spectrum(mock_deterministic_ds, mock_ensemble_ds):
    """Test Q spectrum calculation."""
    res_det = compute_q_spectrum(mock_deterministic_ds, level=500.0)
    assert isinstance(res_det, tuple)

    res_ens = compute_q_spectrum(mock_ensemble_ds, level=500.0)
    assert isinstance(res_ens, dict)
    assert len(res_ens) == 3


def test_compute_spectral_scores():
    """Test spectral divergence and residual calculations."""
    e_true = np.exp(-np.linspace(0, 5, 20))
    e_pred = e_true * 1.1

    s_div, s_res = compute_spectral_scores(e_pred, e_true)
    assert isinstance(s_div, float)
    assert isinstance(s_res, float)
    assert s_div >= 0
    assert s_res >= 0


def test_compute_hydrostatic_imbalance(mock_deterministic_ds, mock_ensemble_ds):
    """Test hydrostatic balance RMSE calculation."""
    area = get_grid_cell_area(mock_deterministic_ds)

    hydro_det = compute_hydrostatic_imbalance(mock_deterministic_ds, area)
    assert isinstance(hydro_det, float)
    assert hydro_det >= 0

    hydro_ens = compute_hydrostatic_imbalance(mock_ensemble_ds, area)
    assert isinstance(hydro_ens, dict)
    assert len(hydro_ens) == 3


def test_compute_geostrophic_imbalance(mock_deterministic_ds, mock_ensemble_ds):
    """Test geostrophic balance RMSE calculation."""
    area = get_grid_cell_area(mock_deterministic_ds)

    geo_det = compute_geostrophic_imbalance(mock_deterministic_ds, area)
    assert isinstance(geo_det, float)

    geo_ens = compute_geostrophic_imbalance(mock_ensemble_ds, area)
    assert isinstance(geo_ens, dict)
    assert len(geo_ens) == 3


def test_compute_lapse_rate_wasserstein(mock_deterministic_ds):
    """Test lapse rate Wasserstein distance calculation."""
    area = get_grid_cell_area(mock_deterministic_ds)
    res = compute_lapse_rate_wasserstein(mock_deterministic_ds, mock_deterministic_ds, area)
    assert isinstance(res, dict)
    assert "lapse_rate_w1_tropics" in res


def test_compute_drift_slope_and_percentages():
    """Test drift slope and drift percentage calculations."""
    hours = np.array([12, 24, 36, 48])
    values = np.array([100.0, 101.0, 102.0, 103.0])

    slope = compute_drift_slope(hours, values)
    assert np.isclose(slope, 2.0)  # 1 unit per 12h = 2 units per 24h (day)

    drift = compute_drift_percentages(
        hours, values, values, values,
        hours, values, values
    )
    assert isinstance(drift, dict)
    assert "dry_mass_drift_pct_per_day" in drift


def test_compute_conservation_scalars(mock_deterministic_ds, mock_static_ds, mock_ensemble_ds):
    """Test conservation scalars wrapper function."""
    area = get_grid_cell_area(mock_deterministic_ds)
    ps = mock_deterministic_ds["surface_pressure"]
    z_sfc = mock_static_ds["geopotential_at_surface"]

    scalars_det = compute_conservation_scalars(mock_deterministic_ds, ps, area, z_sfc=z_sfc)
    assert isinstance(scalars_det, tuple)
    assert len(scalars_det) == 3

    scalars_ens = compute_conservation_scalars(
        mock_ensemble_ds, mock_ensemble_ds["surface_pressure"], area, z_sfc=z_sfc
    )
    assert isinstance(scalars_ens, dict)
    assert len(scalars_ens) == 3


def test_compute_pure_tcwv(mock_deterministic_ds):
    """Test fixed-level pure TCWV integration."""
    tcwv_pure = compute_pure_tcwv(mock_deterministic_ds)
    assert isinstance(tcwv_pure, xr.DataArray)
    assert tcwv_pure.shape == (32, 64)


def test_compute_scalar_spectrum(mock_deterministic_ds):
    """Test spherical harmonic spectrum calculation for arbitrary scalar fields."""
    k, spec = compute_scalar_spectrum(mock_deterministic_ds, var_name="temperature", level=850.0)
    assert isinstance(k, np.ndarray)
    assert isinstance(spec, np.ndarray)
    assert len(k) == len(spec)
    assert len(k) > 0
