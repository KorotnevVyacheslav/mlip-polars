import json

import polars as pl
from pymatgen.analysis.structure_matcher import (
    ElementComparator,
    StructureMatcher,
)
from pymatgen.core import Structure


def load_result_table(path, energy_column, structure_column):
    return pl.scan_parquet(path).select(
        pl.col("material_id"),
        pl.col(energy_column).alias("energy"),
        pl.col(structure_column).alias("structure"),
    )


def calculate_energy_metrics(comparison_df):
    difference = pl.col("energy_difference_per_atom")

    return comparison_df.select(
        pl.len().alias("n"),
        difference.abs().sum().alias("L1"),
        difference.pow(2).sum().sqrt().alias("L2"),
        difference.abs().mean().alias("MAE"),
        difference.pow(2).mean().alias("MSE"),
        difference.pow(2).mean().sqrt().alias("RMSE"),
    ).row(0, named=True)


def calculate_geometry_metrics(mlip_structure, pbe_structure, matcher):
    mlip = Structure.from_dict(json.loads(mlip_structure))
    pbe = Structure.from_dict(json.loads(pbe_structure))

    distances = matcher.get_rms_dist(pbe, mlip)

    return {
        "rmsd": distances[0] if distances is not None else None,
        "volume_difference_pct": 100 * (mlip.volume / pbe.volume - 1),
        "num_sites": mlip.num_sites,
        "num_sites_pbe": pbe.num_sites,
    }


def compare_tables(mlip_df, pbe_df):
    matcher = StructureMatcher(
        ltol=0.2,
        stol=0.3,
        angle_tol=5,
        primitive_cell=False,
        scale=False,
        attempt_supercell=False,
        allow_subset=False,
        comparator=ElementComparator(),
    )

    comparison_df = (
        mlip_df.lazy()
        .join(
            pbe_df.lazy(),
            on="material_id",
            how="inner",
            suffix="_pbe",
        )
        .with_columns(
            (
                pl.col("energy").cast(pl.Float64)
                - pl.col("energy_pbe").cast(pl.Float64)
            ).alias("energy_difference"),
            # pymatgen работает с Python-объектами:
            # здесь остаётся вызов Python-функции для каждой пары.
            pl.struct("structure", "structure_pbe")
            .map_elements(
                lambda row: calculate_geometry_metrics(
                    row["structure"],
                    row["structure_pbe"],
                    matcher,
                ),
                return_dtype=pl.Struct(
                    {
                        "rmsd": pl.Float64,
                        "volume_difference_pct": pl.Float64,
                        "num_sites": pl.Int64,
                        "num_sites_pbe": pl.Int64,
                    }
                ),
            )
            .alias("geometry"),
        )
        .unnest("geometry")
        .with_columns(
            (
                pl.col("energy").cast(pl.Float64) / pl.col("num_sites")
                - pl.col("energy_pbe").cast(pl.Float64) / pl.col("num_sites_pbe")
            ).alias("energy_difference_per_atom")
        )
        .collect()
    )

    energy_metrics = calculate_energy_metrics(comparison_df)

    geometry_metrics = comparison_df.select(
        pl.len().alias("n"),
        pl.col("rmsd").count().alias("n_matched"),
        pl.col("rmsd").mean().alias("mean_rmsd"),
        pl.when(pl.len() > 0)
        .then(pl.col("rmsd").count() / pl.len())
        .otherwise(None)
        .alias("match_fraction"),
        pl.col("volume_difference_pct")
        .abs()
        .mean()
        .alias("mean_absolute_volume_difference_pct"),
    ).row(0, named=True)

    return comparison_df, energy_metrics, geometry_metrics
