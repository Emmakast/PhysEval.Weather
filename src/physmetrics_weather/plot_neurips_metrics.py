#!/usr/bin/env python3
"""Visualisation Script for Physics Metrics Benchmark.

Plots evaluation graphics for AI weather prediction model comparisons:
1. Kinetic energy & humidity spectra across lead times (12h, 120h, 240h).
2. Summary metric tables and comparative figures.
3. Timeseries of dry air mass, water mass, total energy, and balance RMSEs.
4. Environmental lapse rate distributions across tropical and mid-latitude regions.

Ensemble Support:
    Handles long-format CSV inputs containing an optional `ensemble_member` column,
    aggregating statistics (mean and standard deviation) across ensemble members or
    initialization dates.

Usage:
    physmetrics-plot --results-dir ./results --outdir ./plots
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wasserstein_distance


# ============================================================================
# Plotting Configuration & Model Styling
# ============================================================================

MODELS: List[str] = ["hres", "pangu", "graphcast", "neuralgcm", "fuxi", "aurora"]

NICE: Dict[str, str] = {
    "hres": "HRES",
    "pangu": "Pangu",
    "graphcast": "GraphCast",
    "neuralgcm": "NeuralGCM",
    "fuxi": "FuXi",
    "aurora": "Aurora",
}

MODEL_STYLES: Dict[str, Dict[str, str]] = {
    "aurora": {"color": "#0072B2", "marker": "o"},     # Blue
    "pangu": {"color": "#D55E00", "marker": "s"},      # Vermilion
    "fuxi": {"color": "#009E73", "marker": "^"},       # Bluish Green
    "graphcast": {"color": "#000000", "marker": "D"},  # Black
    "neuralgcm": {"color": "#E69F00", "marker": "v"},  # Orange
    "hres": {"color": "#56B4E9", "marker": "P"},       # Sky Blue
}

EARTH_RADIUS_KM: float = 6371.0


# ============================================================================
# Helper Functions
# ============================================================================

def infer_reference_label(results_dir: Path) -> str:
    """Infer the reference dataset label based on results directory naming.

    Args:
        results_dir: Path to directory containing results CSV files.

    Returns:
        Reference label string ("IFS" or "ERA5").
    """
    name = results_dir.name.lower()
    if "ifs" in name:
        return "IFS"
    return "ERA5"


def pretty_region_name(region: str) -> str:
    """Format geographical region key into a human-readable display string.

    Args:
        region: Region identifier key (e.g., "tropics", "nh_mid", "sh_mid").

    Returns:
        Formatted region string.
    """
    mapping = {
        "tropics": "Tropics",
        "nh_mid": "Nor. HS",
        "sh_mid": "Sou. HS",
    }
    return mapping.get(region, region.replace("_", " ").title())


def load_summaries(results_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load evaluation summary CSV files from the results directory.

    Args:
        results_dir: Path to directory containing CSV files.

    Returns:
        Dict mapping model name to summary DataFrame.
    """
    summaries = {}
    for path in results_dir.glob("physics_evaluation_*.csv"):
        try:
            model = path.stem.replace("physics_evaluation_", "").split("_")[0]
            df = pd.read_csv(path)
            summaries[model] = df
        except (ValueError, KeyError, FileNotFoundError) as exc:
            print(f"Warning: Failed to load summary {path}: {exc}")
    return summaries


def get_model_baselines(summaries: Dict[str, pd.DataFrame], metric: str) -> Dict[str, float]:
    """Extract reference baseline metric values for each model.

    Args:
        summaries: Dictionary of model summary DataFrames.
        metric: Target metric name string.

    Returns:
        Dict mapping model name to baseline reference float value.
    """
    bases = {}
    for m, df in summaries.items():
        metric_col = next((c for c in ["metric_name", "metric", "variable", "name"] if c in df.columns), None)
        if metric_col:
            sub = df[df[metric_col] == metric]
            if not sub.empty:
                ref_col = next((c for c in ["ref_value", "mean_ref"] if c in sub.columns), None)
                if ref_col:
                    vals = pd.to_numeric(sub[ref_col], errors="coerce")
                    if vals.notna().any():
                        bases[m] = float(vals.mean())
    return bases


# ============================================================================
# Plotting Modules
# ============================================================================

def plot_timeseries(results_dir: Path, outdir: Path) -> None:
    """Generate timeseries figures for conservation and balance metrics.

    Args:
        results_dir: Path to results directory containing time_series_*.csv files.
        outdir: Output directory path to save generated PNG plots.
    """
    csv_paths = list(results_dir.glob("time_series_*.csv"))
    if not csv_paths:
        print("No time_series_*.csv files found.")
        return

    summaries = load_summaries(results_dir)
    frames = []
    for path in csv_paths:
        try:
            model = path.stem.replace("time_series_", "").split("_")[0]
            if model not in MODELS:
                continue
            df = pd.read_csv(path)
            df["model"] = model
            frames.append(df)
        except (ValueError, KeyError, FileNotFoundError) as exc:
            print(f"Warning: Error reading {path}: {exc}")

    if not frames:
        return
    df_all = pd.concat(frames, ignore_index=True)

    metrics = {
        "dry_mass_Eg": "Dry Air Mass (Eg)",
        "water_mass_kg": "Water Mass (kg)",
        "total_energy_J": "Total Energy (J)",
        "hydrostatic_rmse": "Hydrostatic RMSE Δ",
        "geostrophic_rmse": "Geostrophic RMSE Δ",
    }

    outdir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    for col, title in metrics.items():
        if col not in df_all.columns:
            continue

        bases = get_model_baselines(summaries, col) if "rmse" in col else {}
        fig, ax = plt.subplots(figsize=(10, 5))
        ylabel = title

        for model in MODELS:
            mdf = df_all[df_all["model"] == model].copy()
            if mdf.empty:
                continue
            style = MODEL_STYLES.get(model, {"color": "grey", "marker": "."})

            if col in ["dry_mass_Eg", "water_mass_kg", "total_energy_J"]:
                rel_df = mdf[["date", "forecast_hour", col]].dropna().copy()
                if rel_df.empty:
                    continue

                base = (
                    rel_df.sort_values("forecast_hour")
                    .groupby("date", as_index=False)
                    .first()[["date", col]]
                    .rename(columns={col: "base_val"})
                )
                rel_df = rel_df.merge(base, on="date", how="left")
                rel_df = rel_df[rel_df["base_val"].abs() > 0]
                if rel_df.empty:
                    continue

                rel_df["rel_pct"] = (rel_df[col] - rel_df["base_val"]) / rel_df["base_val"] * 100.0
                agg = rel_df.groupby("forecast_hour")["rel_pct"].agg(["mean", "std"]).reset_index()

                ax.plot(
                    agg["forecast_hour"],
                    agg["mean"],
                    label=NICE.get(model, model),
                    color=style["color"],
                    marker=style["marker"],
                    markersize=3,
                )
                ax.fill_between(
                    agg["forecast_hour"],
                    agg["mean"] - agg["std"].fillna(0),
                    agg["mean"] + agg["std"].fillna(0),
                    color=style["color"],
                    alpha=0.18,
                    linewidth=0,
                )
            else:
                mdf_clean = mdf.copy()
                if "rmse" in col:
                    base_val = bases.get(model, 0.0)
                    mdf_clean[col] = pd.to_numeric(mdf_clean[col], errors="coerce") - base_val

                agg = mdf_clean.groupby("forecast_hour")[col].agg(["mean", "std"]).reset_index()
                if agg.empty:
                    continue

                x = agg["forecast_hour"].values
                y = agg["mean"].values
                y_sigma = agg["std"].fillna(0.0).values

                if col == "hydrostatic_rmse":
                    ylabel = "Δ RMSE (m²/s²)"
                elif col == "geostrophic_rmse":
                    ylabel = "Δ RMSE (m/s)"

                ax.plot(
                    x,
                    y,
                    label=NICE.get(model, model),
                    color=style["color"],
                    marker=style["marker"],
                    markersize=3,
                )
                ax.fill_between(
                    x, y - y_sigma, y + y_sigma, color=style["color"], alpha=0.18, linewidth=0
                )

        ax.set_title(title, fontsize=24)
        ax.set_xlabel("Forecast Hour", fontsize=18)
        ax.set_ylabel(ylabel, fontsize=18)
        ax.tick_params(axis="both", which="major", labelsize=14)

        if col == "geostrophic_rmse":
            ax.legend(fontsize=12, loc="upper left")
        elif col == "total_energy_J":
            ax.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc="upper left")

        fig.savefig(outdir / f"ts_{col}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved timeseries plot: {outdir / f'ts_{col}.png'}")


def plot_spectra(
    results_dir: Path,
    outdir: Path,
    leads: List[int] = [12, 120, 240],
    reference_label: Optional[str] = None,
) -> None:
    """Plot kinetic energy spectra across target lead times.

    Args:
        results_dir: Directory containing spectra_*.csv files.
        outdir: Directory to save generated plot images.
        leads: List of target lead times in hours.
        reference_label: Reference dataset label.
    """
    if reference_label is None:
        reference_label = infer_reference_label(results_dir)

    csv_paths = list(results_dir.glob("spectra_*.csv"))
    if not csv_paths:
        print("No spectra_*.csv files found.")
        return

    frames = []
    for path in csv_paths:
        try:
            model = path.stem.replace("spectra_", "").split("_")[0]
            if model not in MODELS:
                continue
            df = pd.read_csv(path)
            df["model"] = model
            frames.append(df)
        except (ValueError, KeyError, FileNotFoundError) as exc:
            print(f"Warning: Error reading {path}: {exc}")

    if not frames:
        return
    df_all = pd.concat(frames, ignore_index=True)
    outdir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    for lt in leads:
        sub = df_all[(df_all["lead_hours"] == lt) & (df_all["variable"] == "KE") & (df_all["wavenumber"] > 0)]
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))

        ref_agg = sub.groupby("wavenumber")["power_ref"].mean().reset_index()
        if not ref_agg.empty:
            wl = 2.0 * np.pi * EARTH_RADIUS_KM / ref_agg["wavenumber"].values
            ax.loglog(wl, ref_agg["power_ref"].values, color="black", linewidth=2, label=reference_label, zorder=5)

        for model in MODELS:
            msub = sub[sub["model"] == model]
            if msub.empty:
                continue
            msub_agg = msub.groupby("wavenumber")["power_pred"].mean().reset_index()
            style = MODEL_STYLES.get(model, {"color": "grey"})
            wl = 2.0 * np.pi * EARTH_RADIUS_KM / msub_agg["wavenumber"].values
            ax.loglog(wl, msub_agg["power_pred"].values, color=style["color"], linewidth=1.5, label=NICE.get(model, model))

        ax.set_title(f"KE Spectrum - {lt}h", fontsize=24)
        ax.set_xlabel("Wavelength (km)", fontsize=18)
        ax.set_ylabel("Kinetic Energy", fontsize=18)
        ax.set_xlim(40000, 100)
        ax.tick_params(axis="both", which="major", labelsize=14)

        if lt == 240:
            ax.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc="upper left")

        fig.savefig(outdir / f"spectra_ke_{lt}h.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved spectra plot: {outdir / f'spectra_ke_{lt}h.png'}")


# ============================================================================
# CLI Command Entrypoint
# ============================================================================

def main() -> None:
    """CLI entrypoint for physmetrics-plot command."""
    parser = argparse.ArgumentParser(description="Generate diagnostic plots for physics metrics")
    parser.add_argument("--results-dir", type=str, default=str(Path.cwd() / "results"), help="Path to results directory.")
    parser.add_argument("--outdir", type=str, default=str(Path.cwd() / "plots"), help="Path to output plots directory.")
    parser.add_argument("--reference-label", type=str, default=None, help="Reference label override (ERA5 or IFS).")

    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    outdir = Path(args.outdir)

    print(f"Generating plots from: {results_dir}")
    print(f"Saving figures to: {outdir}")

    plot_timeseries(results_dir, outdir)
    plot_spectra(results_dir, outdir, reference_label=args.reference_label)
    print("Plotting complete.")


if __name__ == "__main__":
    main()
