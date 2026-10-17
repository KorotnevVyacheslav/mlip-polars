import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


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
def _(Structure, df, json, pl):
    import ase.build
    import torch_sim as ts
    from metatomic_torchsim import MetatomicModel
    from metatomic.torch import load_atomistic_model

    atomistic_model = load_atomistic_model("model.pt")
    model = MetatomicModel(atomistic_model, device="cpu")

    dfs = []

    for df_batch in df.collect_batches(chunk_size=32):
        structures = [
            Structure.from_dict(json.loads(_i)) for _i in df_batch["structure"]
        ]

        sim_state = ts.initialize_state(
            structures,
            device=model.device,
            dtype=model.dtype,
        )

        relaxed_state = ts.optimize(
            system=sim_state,
            model=model,
            optimizer=ts.Optimizer.lbfgs,
            max_steps=1000,

        )


        energies = relaxed_state.energy.cpu().numpy()

        for i, e in enumerate(energies):
            print(f"Structure {i}: E = {e:.4f} eV")

        relaxed_atoms_list = relaxed_state.to_structures()

        dfs.append(
            pl.DataFrame(
                {
                    "material_id": df_batch["material_id"],
                    "energies" : energies,
                    "structure_out" : [_struct.to_json() for _struct in relaxed_atoms_list],
                }
            )
        )
    return (dfs,)


@app.cell
def _(df, dfs, pl):
    df.collect().join(
        pl.concat(dfs), how="left", on="material_id"
    ).with_columns(
        (pl.col("relaxed_energy") - pl.col("energies")).alias("d_energy")
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
