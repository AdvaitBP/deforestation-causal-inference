"""Run the propensity-score matching workflow using the packaged utilities."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import geopandas as gpd

from deforestation.matching import MatchSummary, run_matching_pipeline


DATA_DIR = Path("..") / "data"


def load_grid() -> gpd.GeoDataFrame:
    """Load the pre-processed grid with forest loss and protection status."""

    grid_fp = DATA_DIR / "processed" / "grid_stats_with_protection.geojson"
    return gpd.read_file(grid_fp)


def add_population_covariate(grid: gpd.GeoDataFrame, pop_fp: Path) -> gpd.GeoDataFrame:
    """Append population density (mean per cell) from a raster file."""

    from rasterstats import zonal_stats

    stats = zonal_stats(grid.geometry, pop_fp, stats="mean", nodata=-3.4028230607370965e38)
    grid["pop_density"] = [s["mean"] for s in stats]
    return grid


def add_road_distance_covariate(grid: gpd.GeoDataFrame, roads_fp: Path) -> gpd.GeoDataFrame:
    """Compute centroid distance to the nearest road for each grid cell."""

    roads = gpd.read_file(roads_fp)
    roads_union = roads.unary_union
    grid_proj = grid.to_crs(epsg=3857)
    grid_proj["centroid"] = grid_proj.geometry.centroid
    grid_proj["road_dist"] = grid_proj["centroid"].apply(lambda pt: pt.distance(roads_union))
    grid["road_dist"] = grid_proj["road_dist"].values
    return grid


def assemble_covariates(grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    pop_fp = DATA_DIR / "population" / "gpw_v4_popdensity_2020.tif"
    roads_fp = DATA_DIR / "roads" / "roads.geojson"
    grid = add_population_covariate(grid, pop_fp)
    grid = add_road_distance_covariate(grid, roads_fp)
    grid = grid.dropna(subset=["lossyear_mean"]).copy()
    return grid


def run(caliper: float, covariates: Sequence[str]) -> MatchSummary:
    grid = load_grid()
    grid = assemble_covariates(grid)
    summary = run_matching_pipeline(
        grid,
        treatment_col="is_protected",
        outcome_col="lossyear_mean",
        covariates=covariates,
        caliper=caliper,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--caliper", type=float, default=0.1, help="Caliper for matching")
    parser.add_argument(
        "--covariates",
        nargs="*",
        default=["pop_density", "road_dist"],
        help="Covariate columns used in the propensity model",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for matched data and covariate balance summaries.",
    )
    args = parser.parse_args()

    summary = run(caliper=args.caliper, covariates=args.covariates)

    print(f"Propensity score model AUC: {summary.auc:.3f}")
    print(f"ATT (lossyear_mean): {summary.att:.4f}")
    print("Covariate balance after matching:")
    print(summary.balance.to_string(index=False))

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        matched_fp = args.output_dir / "matched_pairs.geojson"
        balance_fp = args.output_dir / "covariate_balance.csv"
        summary.matched_data.to_file(matched_fp, driver="GeoJSON")
        summary.balance.to_csv(balance_fp, index=False)
        print(f"Saved matched data to {matched_fp}")
        print(f"Saved balance table to {balance_fp}")


if __name__ == "__main__":
    main()
