import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import time
    from dataclasses import dataclass
    from pathlib import Path
    import json
    import polars as pl
    from pymatgen.core import Structure
    import upet
    import ase.build
    import torch_sim as ts
    from metatomic_torchsim import MetatomicModel
    from metatomic.torch import load_atomistic_model

    return Path, dataclass, time, upet


@app.cell
def _(time):
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

    return


@app.cell
def _(Path, dataclass):
    @dataclass
    class RelaxationConfig:
        dataset_path: Path
        model_path: Path
        output_path: Path
        cache_dir: Path

        batch_size: int = 32
        max_steps: int = 1000
        optimizer: str = "lbfgs"
        device: str = "cpu"

    return (RelaxationConfig,)


app._unparsable_cell(
    r"""
    @measure_time
    def optimize_batch(sim_state, model, optimizer, max_steps):
        return ts.optimize(
            system=sim_state,
            model=model,
            optimizer=optimizer,
            max_steps=max_steps,
        )


    class BatchRelaxer:
        def __init__(self, config: RelaxationConfig):
            self.config = config

            atomistic_model = load_atomistic_model(
                str(self.config.model_path)
            )

            self.model = MetatomicModel(
                atomistic_model,
                device=self.config.device,
            )

        def run(self) -> Path:
            df = (
                pl.scan_parquet(self.config.dataset_path)
                .sort("num_sites")
                .head(128)
            )
        
            output_dir = self.config.cache_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            batch_files = []

            for batch_idx, df_batch in enumerate(
                df.collect_batches(chunk_size=self.config.batch_size)
            ):
                cache_file = output_dir / f"batch_{batch_idx:04d}.parquet"
        batch_files.append(str(cache_file))

        if cache_file.exists():
            print(f"Batch {batch_idx} already completed, skipping")
            continue
                structures = [
                    Structure.from_dict(json.loads(s))
                    for s in df_batch["structure"]
                ]

                sim_state = ts.initialize_state(
                    structures,
                    device=self.model.device,
                    dtype=self.model.dtype,
                )

                relaxed_state = optimize_batch(
                    sim_state,
                    self.model,
                    getattr(ts.Optimizer, self.config.optimizer),
                    self.config.max_steps,
                )

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

                result.write_parquet(cache_file)

                del structures, sim_state, relaxed_state
                del energies, relaxed_structures, result

            results = pl.scan_parquet(batch_files)

            combined = df.join(
                results,
                on="material_id",
                how="left",
            )

            self.config.output_path.parent.mkdir(
                parents=True, exist_ok=True
            )

            combined.sink_parquet(self.config.output_path)

            return self.config.output_path
    """,
    name="_"
)


@app.cell
def _(config, upet):
    upet.save_upet(
        model="pet-mad",
        size="m",
        version="1.6.0",
        output=str(config.model_path),
    )
    return


@app.cell
def _(Path, RelaxationConfig):
    config = RelaxationConfig(
        dataset_path=Path("data/materials.parquet"),
        model_path=Path("model.pt"),
        output_path=Path("combined_results.parquet"),
        cache_dir=Path("results"),
        batch_size=32,
        max_steps=1000,
        optimizer="lbfgs",
        device="cpu",
    )
    return (config,)


@app.cell
def _(BatchRelaxer, config):
    relaxer = BatchRelaxer(config)
    output_path = relaxer.run()
    return


if __name__ == "__main__":
    app.run()
