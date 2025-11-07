"""Command-line interface for the deforestation causal inference toolkit."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd
import typer

from .matching import MatchSummary, run_matching_pipeline

app = typer.Typer(help="Causal inference utilities for the deforestation project")


def _load_tabular(path: Path) -> pd.DataFrame:
    """Load CSV or Parquet data based on file extension."""

    if not path.exists():
        raise typer.BadParameter(f"Data file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv", ".tsv"}:
        sep = "," if suffix == ".csv" else "\t"
        return pd.read_csv(path, sep=sep)
    if suffix in {".geojson", ".json", ".gpkg"}:
        try:
            import geopandas as gpd
        except ImportError as exc:
            raise typer.BadParameter("GeoPandas is required to read geospatial files.") from exc
        gdf = gpd.read_file(path)
        return pd.DataFrame(gdf)

    raise typer.BadParameter("Only CSV, Parquet, and GeoJSON inputs are supported.")


def _echo_summary(summary: MatchSummary, *, outcome_col: str) -> None:
    """Pretty-print the main outputs of the matching pipeline."""

    typer.secho("\nPropensity score model", fg=typer.colors.CYAN)
    typer.echo(f"AUC: {summary.auc:.3f}")

    typer.secho("\nAverage treatment effect on the treated", fg=typer.colors.CYAN)
    typer.echo(f"ATT ({outcome_col}): {summary.att:.4f}")

    typer.secho("\nCovariate balance after matching", fg=typer.colors.CYAN)
    typer.echo(summary.balance.to_string(index=False))


def _save_outputs(summary: MatchSummary, *, output_dir: Optional[Path], prefix: str) -> None:
    if output_dir is None:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    matched_path = output_dir / f"{prefix}_matched.csv"
    balance_path = output_dir / f"{prefix}_balance.csv"
    summary.matched_data.to_csv(matched_path, index=False)
    summary.balance.to_csv(balance_path, index=False)
    typer.echo(f"Saved matched data to {matched_path}")
    typer.echo(f"Saved balance table to {balance_path}")


@app.command()
def match(
    data_path: Path = typer.Argument(..., help="Input dataset (CSV or Parquet)."),
    treatment_col: str = typer.Option("is_protected", help="Binary treatment indicator column."),
    outcome_col: str = typer.Option("lossyear_mean", help="Outcome column for ATT."),
    covariate: List[str] = typer.Option(
        None,
        help="Covariate columns to include in the propensity model. Provide multiple --covariate entries.",
    ),
    caliper: float = typer.Option(0.1, help="Maximum propensity-score difference for matching."),
    allow_replacement: bool = typer.Option(False, help="Allow control reuse during matching."),
    output_dir: Optional[Path] = typer.Option(
        None, help="Optional directory where matched data and balance diagnostics are saved."
    ),
    prefix: str = typer.Option("matching", help="Prefix for saved artefacts."),
) -> None:
    """Run the propensity-score matching pipeline on a tabular dataset."""

    df = _load_tabular(data_path)
    if not covariate:
        raise typer.BadParameter("At least one --covariate must be provided.")

    summary = run_matching_pipeline(
        df,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        covariates=covariate,
        caliper=caliper,
        allow_replacement=allow_replacement,
    )
    _echo_summary(summary, outcome_col=outcome_col)
    _save_outputs(summary, output_dir=output_dir, prefix=prefix)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
