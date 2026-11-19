import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    from mlip_polars.Batch_class import BatchRelaxer, RelaxationConfig
    from mlip_polars.meas_func import compare_tables, load_result_table

    return (
        BatchRelaxer,
        Path,
        RelaxationConfig,
        compare_tables,
        load_result_table,
    )


@app.cell
def _(Path, RelaxationConfig):
    project_dir = Path("/home/yaroslav/projects/mlip-polars")

    config = RelaxationConfig(
        dataset_path=(
            project_dir / "data" / "energy_comparision" / "df_initial.parquet"
        ),
        model_path=project_dir / "models" / "model.pt",
        output_path=project_dir / "results" / "combined_results.parquet",
        cache_dir=project_dir / "results" / "batches_experiment_1",
        batch_size=32,
        device="cpu",
    )
    return config, project_dir


@app.cell
def _(config):
    import upet

    config.model_path.parent.mkdir(parents=True, exist_ok=True)

    if not config.model_path.exists():
        upet.save_upet(
            model="pet-mad",
            size="m",
            version="1.6.0",
            output=str(config.model_path),
        )


@app.cell
def _(BatchRelaxer, config):
    relaxer = BatchRelaxer(config)
    output_path = relaxer.run()
    print(f"Результаты сохранены: {output_path}")
    return output_path, relaxer


@app.cell
def _(relaxer):
    for record in relaxer.timer.records:
        print(f"{record.name}: {record.elapsed_s:.3f} s")


@app.cell
def _(load_result_table, output_path, project_dir):
    mlip_df = load_result_table(
        path=output_path,
        energy_column="energies",
        structure_column="structure_out",
    )

    pbe_df = load_result_table(
        path=(project_dir / "data" / "energy_comparision" / "df_pbe_high.parquet"),
        energy_column="energy",
        structure_column="structure_final",
    )
    return mlip_df, pbe_df


@app.cell
def _(compare_tables, mlip_df, pbe_df):
    comparison_df, energy_metrics, geometry_metrics = compare_tables(
        mlip_df,
        pbe_df,
    )
    return comparison_df, energy_metrics, geometry_metrics


@app.cell
def _(energy_metrics, geometry_metrics):
    print("Энергетические метрики:")
    print(energy_metrics)

    print("Геометрические метрики:")
    print(geometry_metrics)


@app.cell
def _(comparison_df):
    comparison_df


if __name__ == "__main__":
    app.run()
