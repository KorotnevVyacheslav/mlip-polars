import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import time
    from functools import wraps


    def measure_time(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            wrapper.last_elapsed = elapsed
            print(f"{func.__name__}: {elapsed:.3f} s")
            return result

        return wrapper

    return (measure_time,)


@app.cell
def _():
    import json

    import marimo as mo
    import polars as pl
    from pymatgen.core import Structure

    file_path: str = "/media/korotnev/7D7F-C362/lcdm/data/materials_project/mp_energies.parquet"

    # df = pl.scan_parquet(
    #     "../databases/data/datasets/dft_structures_20261012.parquet"
    # ).select(
    #     [
    #         "material_id",
    #         "structure_dft",
    #         "structure_initial",
    #         "structure_mattersim",
    #         "relaxation_energy_dft",
    #         "relaxation_energy_mattersim",
    #     ]
    # ).head(128)

    df = pl.scan_parquet(file_path).sort("num_sites").head(128)
    return Structure, df, json, pl


@app.cell
def _():
    import upet

    upet.save_upet(
        model="pet-mad",
        size="m",
        version="1.6.0",
        output="model.pt",
    )
    return


@app.cell
def _():
    import ase.build
    import torch_sim as ts
    from metatomic_torchsim import MetatomicModel
    from metatomic.torch import load_atomistic_model

    atomistic_model = load_atomistic_model("model.pt")
    model = MetatomicModel(atomistic_model, device="cpu")
    return model, ts


@app.cell
def _(Structure, df, json, measure_time, model, pl, ts):
    from pathlib import Path

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    @measure_time
    def optimize_batch(sim_state, model):
        return ts.optimize(
            system=sim_state,
            model=model,
            optimizer=ts.Optimizer.lbfgs,
            max_steps=1000,
        )

    for batch_idx, df_batch in enumerate(df.collect_batches(chunk_size=32)):
        structures = [
            Structure.from_dict(json.loads(s))
            for s in df_batch["structure"]
        ]

        sim_state = ts.initialize_state(
            structures,
            device=model.device,
            dtype=model.dtype,
        )

        relaxed_state = optimize_batch(sim_state, model)
        batch_time = optimize_batch.last_elapsed

        energies = relaxed_state.energy.detach().cpu().numpy()
        relaxed_structures = relaxed_state.to_structures()

        result = pl.DataFrame({
            "material_id": df_batch["material_id"],
            "energies": energies,
            "structure_out": [
                structure.to_json()
                for structure in relaxed_structures
            ],
            "batch_time_s": [batch_time] * len(energies),
        })

        result.write_parquet(
            output_dir / f"batch_{batch_idx:04d}.parquet"
        )

        del structures, sim_state, relaxed_state
        del energies, relaxed_structures, result
    return (output_dir,)


@app.cell
def _(df, output_dir, pl):
    results = pl.scan_parquet(str(output_dir / "batch_*.parquet"))

    combined = df.join(
        results,
        on="material_id",
        how="left",
    )

    combined.sink_parquet("combined_results.parquet")
    return


if __name__ == "__main__":
    app.run()
