import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from typing import Any
    import logging
    from dataclasses import dataclass, field
    from ase.calculators.calculator import Calculator
    from torch_sim.models.mattersim import MatterSimModel
    from mattersim.torchsim import get_torchsim_wrapper

    return Calculator, dataclass, field, logging


@app.cell
def _():
    from mlip_polars.utils.log import setup_logger
    logger = setup_logger()
    return (logger,)


@app.cell
def _(dataclass, logging, pl):
    @dataclass
    class DataLoaderConfig:
        path: str = ""

    class DataLoader:
        def __init__(self, config: DataLoaderConfig, logger: logging.Logger):
            self.config = config
            self.logger = logger

        def run(self) -> pl.LazyFrame:
            return pl.scan_parquet(self.config.path)

    return DataLoader, DataLoaderConfig


@app.cell
def _(dataclass, field, logging):
    @dataclass
    class ModelLoaderConfig:
        model_name: str = ""
        model_params: dict = field(default_factory=lambda: {})

    class ModelLoader:
        def __init__(self, config: ModelLoaderConfig, logger: logging.Logger):
            self.config = config
            self.logger = logger

        def run(self):
            match self.config.model_name:
                case "mattersim":
                    self.logger.info("Trying to load MatterSim model")
                    from mattersim.torchsim import get_torchsim_wrapper

                    return get_torchsim_wrapper(
                        **self.config.model_params
                    )
                case _:
                    self.logger.error(f"Unknown model name: {self.config.name}")
                    raise ValueError("Unknown Model")

    return ModelLoader, ModelLoaderConfig


@app.cell
def _(Calculator, dataclass, field, logging):
    import json
    # from dataclasses import dataclass
    from pathlib import Path

    import polars as pl
    import torch_sim as ts
    # from metatomic.torch import load_atomistic_model
    # from metatomic_torchsim import MetatomicModel
    from pymatgen.core import Structure

    from mlip_polars.timer_class import Timer

    @dataclass
    class RelaxationConfig:
        cache_dir: Path = field(default_factory=lambda : Path("data/cache", mkdir=True))
        batch_size: int = 32
        max_steps: int = 1000
        fire_max_steps: int = 200
        fire_fmax: float = 0.1
        lbfgs_max_steps: int = 1000
        lbfgs_fmax: float = 0.01
        device: str = "cpu"

    class BatchRelaxer:
        def __init__(self, config: RelaxationConfig, logger: logging.Logger):
            self.config = config
            self.logger = logger
            self.timer = Timer()
            self.optimize_batch = self.timer.measure(self.optimize_batch)

        def optimize_batch(self, sim_state: ts.state, model : Calculator):
            fire_state = ts.optimize(
                system=sim_state,
                model=model,
                optimizer=ts.Optimizer.fire,
                max_steps=self.config.fire_max_steps,
                convergence_fn=ts.generate_force_convergence_fn(
                    force_tol=self.config.fire_fmax,
                ),
            )

            lbfgs_input = ts.SimState.from_state(fire_state)
            del fire_state

            return ts.optimize(
                system=lbfgs_input,
                model=model,
                optimizer=ts.Optimizer.lbfgs,
                max_steps=self.config.lbfgs_max_steps,
                convergence_fn=ts.generate_force_convergence_fn(
                    force_tol=self.config.lbfgs_fmax,
                ),
            )

        def run(self, model : Calculator, data : pl.DataFrame) -> pl.LazyFrame:
            # df = pl.scan_parquet(self.config.dataset_path).head(128)

            df = data
            # создание папки для файлов батчей:
            output_dir = self.config.cache_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            batch_files = []

            for batch_idx, df_batch in enumerate(
                df.collect_batches(chunk_size=self.config.batch_size)
            ):
                # проверка нужно ли рассчитывать батч:
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
                    device=model.device,
                    dtype=model.dtype,
                )

                # релаксация:
                relaxed_state = self.optimize_batch(sim_state, model)

                # получаем данные
                batch_time = self.timer.records[-1].elapsed_s

                energies = relaxed_state.energy.detach().cpu().numpy()
                relaxed_structures = relaxed_state.to_structures()

                # сохраняем датафрейм:
                result = pl.DataFrame(
                    {
                        "material_id": df_batch["material_id"],
                        "energies": energies,
                        "structure_out": [
                            structure.to_json()
                            for structure in relaxed_structures
                        ],
                        "batch_time_s": [batch_time] * len(energies),
                    }
                )

                result.write_parquet(cache_file)

                # удаляем временыне данные
                del structures, sim_state, relaxed_state
                del energies, relaxed_structures, result

            # собираем общую таблицу:
            results = pl.scan_parquet(batch_files)

            combined = df.join(
                results,
                on="material_id",
                how="left",
            )

            return combined

    return BatchRelaxer, Path, RelaxationConfig, pl


@app.cell
def _(Path, dataclass, logging, pl):
    @dataclass
    class DataSaverConfig:
        path : str = "data/results/combined_results.parquet"

    class DataSaver:
        def __init__(self, config: DataSaverConfig, logger: logging.Logger):
            self.config = config
            self.logger = logger

        def run(self, data:pl.LazyFrame):
            output_path = Path(self.config.path, mkdir=True)
            data.sink_parquet(output_path)

    return DataSaver, DataSaverConfig


@app.cell
def _(DataLoader, DataLoaderConfig, logger):
    config_dataloader = DataLoaderConfig(path="data/energy_comparision/df_initial.parquet")
    dataloader = DataLoader(config=config_dataloader, logger=logger)
    data = dataloader.run().head(32)
    return (data,)


@app.cell
def _(ModelLoader, ModelLoaderConfig, logger):
    config_model_loader = ModelLoaderConfig(
        model_name="mattersim",
        model_params={"potential": "mattersim-v1.0.0-1M", "device": "cpu"},
    )
    model_loader = ModelLoader(config=config_model_loader, logger=logger)

    model = model_loader.run()
    return (model,)


@app.cell
def _(BatchRelaxer, RelaxationConfig, data, logger, model):
    config_relaxer = RelaxationConfig()
    relaxer = BatchRelaxer(config=config_relaxer, logger=logger)

    output_data = relaxer.run(model=model, data=data)
    return (output_data,)


@app.cell
def _(DataSaver, DataSaverConfig, logger, output_data):
    config_datasaver = DataSaverConfig()
    datasaver = DataSaver(config=config_datasaver, logger=logger)

    datasaver.run(data=output_data)
    return


@app.cell
def _(
    DataLoaderConfig,
    DataSaverConfig,
    ModelLoaderConfig,
    RelaxationConfig,
    dataclass,
):
    @dataclass
    class PipelineConfig:
        config_dataloader: DataLoaderConfig
        config_model_loader: ModelLoaderConfig
        config_relaxer: RelaxationConfig
        config_datasaver: DataSaverConfig

    class Pipeline:
        pass

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
