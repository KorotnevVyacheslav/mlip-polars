import math
import polars as pl
import json
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import (
    ElementComparator,
    StructureMatcher,
)

def load_result_table(path, energy_column, structure_column):
    df = pl.read_parquet(
        path,
        columns=["material_id", energy_column, structure_column],
    )

    return df.select(
        pl.col("material_id"),
        pl.col(energy_column).alias("energy"),
        pl.col(structure_column).alias("structure"),
    )


def calculate_energy_metrics(mlip_energies, pbe_energies):
    n = len(mlip_energies)

    differences = [
        mlip-pbe
        for mlip, pbe in zip(mlip_energies, pbe_energies)
    ]

    sum_absolute = sum(abs(value) for value in differences)
    sum_squared = sum(value**2 for value in differences)

    return {
        "n": n,
        "L1": sum_absolute,
        "L2": math.sqrt(sum_squared),
        "MAE": sum_absolute/n,
        "MSE": sum_squared/n,
        "RMSE": math.sqrt(sum_squared/n),
    }

def calculate_geometry_metrics(mlip_structure, pbe_structure, matcher):
    mlip = Structure.from_dict(json.loads(mlip_structure))
    pbe = Structure.from_dict(json.loads(pbe_structure))

    distances = matcher.get_rms_dist(pbe, mlip)

    return {
        "rmsd": distances[0] if distances is not None else None,
        "volume_difference_pct": 100*(mlip.volume/pbe.volume-1),
    }

def compare_tables(mlip_df, pbe_df):
    comparison_df = mlip_df.join(
        pbe_df,
        on="material_id",
        how="inner",
        suffix="_pbe",
    )

    comparison_df = comparison_df.with_columns(
        (pl.col("energy") - pl.col("energy_pbe"))
        .alias("energy_difference")
    )

    energy_metrics = calculate_energy_metrics(
        mlip_energies=comparison_df["energy"].to_list(),
        pbe_energies=comparison_df["energy_pbe"].to_list(),
    )

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

    geometry_rows = [
        calculate_geometry_metrics(mlip, pbe, matcher)
        for mlip, pbe in zip(
            comparison_df["structure"],
            comparison_df["structure_pbe"],
        )
    ]

    comparison_df = comparison_df.with_columns(
        pl.Series(
            "rmsd",
            [row["rmsd"] for row in geometry_rows],
            dtype=pl.Float64,
        ),
        pl.Series(
            "volume_difference_pct",
            [row["volume_difference_pct"] for row in geometry_rows],
            dtype=pl.Float64,
        ),
    )

    geometry_metrics = {
        "n": comparison_df.height,
        "n_matched": comparison_df["rmsd"].count(),
        "mean_rmsd": comparison_df["rmsd"].mean(),
        "match_fraction": (
            comparison_df["rmsd"].count() / comparison_df.height
        ),
        "mean_absolute_volume_difference_pct": (
            comparison_df["volume_difference_pct"].abs().mean()
        ),
    }

    return comparison_df, energy_metrics, geometry_metrics

