import math
import polars as pl


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

def compare_energy_tables(mlip_df, pbe_df):
    comparison_df = mlip_df.join(
        pbe_df,
        on="material_id",
        how="inner",
        suffix="_pbe",
    )

    comparison_df = comparison_df.with_columns(
        (pl.col("energy")-pl.col("energy_pbe"))
        .alias("energy_difference")
    )

    metrics = calculate_energy_metrics(
        mlip_energies=comparison_df["energy"].to_list(),
        pbe_energies=comparison_df["energy_pbe"].to_list(),
    )

    return comparison_df, metrics

