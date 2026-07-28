"""Unit tests for plot_neurips_metrics.py visualization utilities."""

from pathlib import Path
import pandas as pd
import pytest

from physmetrics_weather.plot_neurips_metrics import (
    get_model_baselines,
    infer_reference_label,
    load_summaries,
    pretty_region_name,
)


def test_infer_reference_label(tmp_path):
    """Test reference label inference from directory path name."""
    era5_dir = tmp_path / "results_era5"
    ifs_dir = tmp_path / "results_ifs"

    assert infer_reference_label(era5_dir) == "ERA5"
    assert infer_reference_label(ifs_dir) == "IFS"


def test_pretty_region_name():
    """Test region name formatting."""
    assert pretty_region_name("tropics") == "Tropics"
    assert pretty_region_name("nh_mid") == "Nor. HS"
    assert pretty_region_name("custom_region") == "Custom Region"


def test_load_summaries_and_baselines(tmp_path):
    """Test loading model summary CSV files and calculating baselines."""
    df_aurora = pd.DataFrame({
        "date": ["2022-01-01"],
        "lead_time_hours": [12],
        "metric_name": ["hydrostatic_rmse"],
        "model_value": [10.5],
        "ref_value": [8.0],
        "ensemble_member": [0],
    })

    file_path = tmp_path / "physics_evaluation_aurora_2022.csv"
    df_aurora.to_csv(file_path, index=False)

    summaries = load_summaries(tmp_path)
    assert "aurora" in summaries
    assert len(summaries["aurora"]) == 1

    baselines = get_model_baselines(summaries, "hydrostatic_rmse")
    assert "aurora" in baselines
    assert baselines["aurora"] == 8.0
