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
    from comparison import load_result_table, compare_tables

    return (
        MetatomicModel,
        Path,
        Structure,
        compare_tables,
        dataclass,
        json,
        load_atomistic_model,
        load_result_table,
        pl,
        time,
        ts,
        upet,
    )


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
    #декоратор позволяющий измерять время работы функций 
    return (measure_time,)


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
        fire_max_steps: int = 200
        fire_fmax: float = 0.1
        lbfgs_max_steps: int = 1000
        lbfgs_fmax: float = 0.01
        device: str = "cpu"

    return (RelaxationConfig,)


@app.cell
def _(
    MetatomicModel,
    Path,
    RelaxationConfig,
    Structure,
    json,
    load_atomistic_model,
    measure_time,
    pl,
    ts,
):
    @measure_time
    def optimize_batch(sim_state, model, config: RelaxationConfig):
        fire_state = ts.optimize(
        system=sim_state,
        model=model,
        optimizer=ts.Optimizer.fire,
        max_steps=config.fire_max_steps,
        convergence_fn=ts.generate_force_convergence_fn(
            force_tol=config.fire_fmax,
        ) 
    )

        lbfgs_input = ts.SimState.from_state(fire_state)
        del fire_state

        return ts.optimize(
            system=lbfgs_input,
            model=model,
            optimizer=ts.Optimizer.lbfgs,
            max_steps=config.lbfgs_max_steps,
            convergence_fn=ts.generate_force_convergence_fn(
                force_tol=config.lbfgs_fmax,
            ),
        )
    #оптимизирование одного батча


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
                pl.scan_parquet(self.config.dataset_path).head(128)
            )
            #создание папки для файлов батчей:
            output_dir = self.config.cache_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            batch_files = []

            for batch_idx, df_batch in enumerate(
                df.collect_batches(chunk_size=self.config.batch_size)
            ):
                #проверка нужно ли рассчитывать батч:
                cache_file = output_dir / f"batch_{batch_idx:04d}.parquet"
                batch_files.append(str(cache_file))

                if cache_file.exists():
                    print(f"Batch {batch_idx} already completed, skipping")
                    continue
    # подготовка структуры:
                structures = [
                    Structure.from_dict(json.loads(s))
                    for s in df_batch["structure"]
                ]

                sim_state = ts.initialize_state(
                    structures,
                    device=self.model.device,
                    dtype=self.model.dtype,
                )
    # релаксация:
                relaxed_state = optimize_batch(
                    sim_state,
                    self.model,
                    self.config,
                )
    # получаем данные
                batch_time = optimize_batch.last_elapsed

                energies = relaxed_state.energy.detach().cpu().numpy()
                relaxed_structures = relaxed_state.to_structures()
    # сохраняем датафрейм:
                result = pl.DataFrame({
                    "material_id": df_batch["material_id"],
                    "energies": energies,
                    "structure_out": [
                        structure.to_json()
                        for structure in relaxed_structures
                    ],
                    "batch_time_s": [batch_time]*len(energies),
                })

                result.write_parquet(cache_file)
    # удаляем временыне данные 
                del structures, sim_state, relaxed_state
                del energies, relaxed_structures, result
    #собираем общую таблицу:
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

    return (BatchRelaxer,)


@app.cell
def _(Path, RelaxationConfig):
    config = RelaxationConfig(
        dataset_path=Path("energy_comparision/df_initial.parquet"),
        model_path=Path("model.pt"),
        output_path=Path("combined_results_fire_lbfgs.parquet"),
        cache_dir=Path("results_fire_lbfgs"),
        batch_size=32,
        max_steps=1000,
        device="cpu",
    )
    # пути и параметры расчета 
    return (config,)


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
def _(BatchRelaxer, config):
    relaxer = BatchRelaxer(config)
    output_path = relaxer.run()
    return (output_path,)


@app.cell
def _(Path, compare_tables, load_result_table, output_path):
    mlip_df = load_result_table(
        output_path,
        energy_column="energies",
        structure_column="structure_out",
    )

    pbe_df = load_result_table( 
        Path("energy_comparision/df_pbe_high.parquet"),
        energy_column="energy",
        structure_column="structure_final",
    )

    comparison_df, energy_metrics, geometry_metrics = compare_tables(
        mlip_df,
        pbe_df,
    )
    return energy_metrics, geometry_metrics


@app.cell
def _(energy_metrics, geometry_metrics):
    print("Энергия:", energy_metrics)
    print("Геометрия:", geometry_metrics)
    return


if __name__ == "__main__":
    app.run()
