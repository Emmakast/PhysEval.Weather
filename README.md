# PhysMetrics.Weather

An open-source, unified framework for evaluating the **physical consistency**, **spectral resolution**, and **atmospheric balance** of Machine Learning Weather Prediction (MLWP) models.

It computes diagnostic physical metrics against ERA5 or IFS HRES reference datasets from [WeatherBench 2](https://weatherbench2.readthedocs.io/), supporting both **deterministic** and **probabilistic / ensemble** weather models.

---

## Authors

* **Emma Kasteleyn**
* **Timo Maier** (`timo.maier@tum.de`)

---

## Key Metrics Evaluated

| Metric Category | Metric | What it measures |
|---|---|---|
| **Conservation & Stability** | **Dry Air Mass drift** | Is global dry-air mass conserved over time? (Exagrams, %/day) |
| | **Water Mass drift** | Is global total atmospheric water mass conserved? (kg, %/day) |
| | **Total Energy drift** | Is global total atmospheric energy conserved? (Joules, %/day) |
| **Spectral Skill** | **Kinetic Energy Spectrum** | Spherical harmonic KE spectrum at 500 hPa & 850 hPa |
| | **Humidity Spectrum** | Power spectrum of specific humidity at 500 hPa |
| | **Effective Resolution ($L_{eff}$)** | Spatial scale (km) where model loses skill vs. reference |
| | **Spectral Divergence** | 1-Wasserstein distance between true and predicted spectra |
| | **Spectral Residual** | Log-RMSE difference between energy spectra |
| **Atmospheric Balance** | **Hydrostatic Balance** | Hypsometric balance error RMSE ($m^2/s^2$) between 500 & 850 hPa |
| | **Geostrophic Balance** | Area-weighted wind balance RMSE ($m/s$) at 500 hPa |
| **Thermal Structure** | **Lapse Rate Wasserstein** | Environmental lapse rate distribution distance across geographical bands |

---

## Installation

The package supports Python `>= 3.10` up to `3.14`. You can manage dependencies using [uv](https://docs.astral.sh/uv/) or standard `pip`.

### Using `uv` (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/Emmakast/PhysMetrics.Weather.git
cd PhysMetrics.Weather

# 2. Sync dependencies and create environment
uv sync --extra dev --extra docs

# 3. Activate the virtual environment
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows
```

### Using standard `pip`

```bash
pip install -e .
```

After installation, the CLI commands `physmetrics-run` and `physmetrics-plot` are available on your system `PATH`:

| Command | Description |
|---|---|
| `physmetrics-run` | Stream WeatherBench 2 Zarr data, run physics metrics, output long-format CSV |
| `physmetrics-plot` | Generate publication-ready figures and visualization plots from output CSVs |

---

## Probabilistic & Ensemble Model Support

`physmetrics-weather` automatically detects extra ensemble dimensions in your datasets (e.g. `ens`, `realization`, `member`, `ensemble`, `number`).

* **Per-Member Evaluation**: Metrics are computed for each individual ensemble member.
* **Output Format**: The resulting long-format CSV includes an `ensemble_member` column (`0` for deterministic models, or member ID `0, 1, 2, ...` for ensemble realizations).

---

## Data Inputs & Usage

### 1. WeatherBench 2 Streaming (No Download Required)

Stream data directly from public WeatherBench 2 Google Cloud Storage buckets:

```bash
# Evaluate Pangu-Weather predictions against ERA5 for 2022
physmetrics-run \
  --model pangu \
  --prediction-zarr gs://weatherbench2/datasets/pangu/2018-2022_0012_0p25.zarr \
  --year 2022 \
  --workers 4 \
  --output-dir ./results
```

### 2. Custom Zarr Dataset Evaluation

Point `--prediction-zarr` at any local or cloud Zarr store:

```bash
physmetrics-run \
  --model my_model \
  --prediction-zarr /path/to/my_model_forecasts.zarr \
  --output-dir ./results
```

Required variable names (or standard aliases):
* **Surface Pressure**: `surface_pressure`, `sp`, `ps`
* **Temperature**: `temperature`, `t`
* **Zonal Wind**: `u_component_of_wind`, `u`
* **Meridional Wind**: `v_component_of_wind`, `v`
* **Specific Humidity**: `specific_humidity`, `q`
* **Geopotential**: `geopotential`, `z`

---

## CLI Options

### `physmetrics-run`

| Option | Default | Description |
|---|---|---|
| `--year` | `2022` | Year to evaluate |
| `--dates` | — | Specific dates, e.g. `2022-01-01 2022-01-15` |
| `--month` | — | Specific month, e.g. `2022-01` |
| `--model` | `model` | Model name identifier |
| `--prediction-zarr` | WB2 path | Zarr store URL or local path for model predictions |
| `--ref-zarr` | ERA5 WB2 path | Zarr store URL or local path for reference dataset |
| `--lead-times` | `12h,5d,10d` | Comma-separated forecast lead times |
| `--workers` | `4` | Parallel worker process count |
| `--output-dir` | `./results` | Directory for output CSV results |
| `--extended-spectra` | off | Also compute 850 hPa KE spectrum and Q spectrum |
| `--quiet` | off | Suppress verbose logging |

### `physmetrics-plot`

```bash
physmetrics-plot --results-dir ./results --outdir ./plots
```

| Option | Default | Description |
|---|---|---|
| `--results-dir` | `./results` | Path to directory containing output CSV files |
| `--outdir` | `./plots` | Path to directory for saving generated figures |
| `--reference-label` | auto | Reference label override for legends (`ERA5` or `IFS`) |

---

## Output CSV Format

`physmetrics-run` outputs long-format CSV files with the following structure:

```csv
date,lead_time_hours,metric_name,model_value,ref_value,n_levels,sp_method,ensemble_member
2022-01-01,12,hydrostatic_rmse,1.23,0.98,13,direct_sp,0
2022-01-01,12,geostrophic_rmse,2.45,2.10,13,direct_sp,0
2022-01-01,120,dry_mass_drift_pct_per_day,-0.002,,13,direct_sp,0
```

---

## Documentation

Full Sphinx documentation is available in the `docs/` directory.

### Build and View Locally

```bash
# Build Sphinx HTML documentation
uv run sphinx-build -b html docs docs/_build/html

# Serve locally at http://localhost:8000
python3 -m http.server 8000 --directory docs/_build/html
```

---

## Testing & Verification

Run the comprehensive unit test suite:

```bash
# Run pytest test suite
uv run pytest

# Verify wheel and source distribution build
uv build
```

---

## License

This project is licensed under the [MIT License](LICENCE).
