import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import torch_sim as ts
from metatomic.torch import load_atomistic_model
from metatomic_torchsim import MetatomicModel
from pymatgen.core import Structure

from .timer_class import Timer


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


class BatchRelaxer:
    def __init__(self, config: RelaxationConfig):
        self.config = config

        atomistic_model = load_atomistic_model(str(self.config.model_path))
        self.model = MetatomicModel(
            atomistic_model,
            device=self.config.device,
        )

        self.timer = Timer()
        self.optimize_batch = self.timer.measure(self.optimize_batch)

    def optimize_batch(self, sim_state):
        fire_state = ts.optimize(
            system=sim_state,
            model=self.model,
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
            model=self.model,
            optimizer=ts.Optimizer.lbfgs,
            max_steps=self.config.lbfgs_max_steps,
            convergence_fn=ts.generate_force_convergence_fn(
                force_tol=self.config.lbfgs_fmax,
            ),
        )

    def run(self) -> Path:
        df = pl.scan_parquet(self.config.dataset_path).head(128)

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
                Structure.from_dict(json.loads(s)) for s in df_batch["structure"]
            ]

            sim_state = ts.initialize_state(
                structures,
                device=self.model.device,
                dtype=self.model.dtype,
            )

            # релаксация:
            relaxed_state = self.optimize_batch(sim_state)

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
                        structure.to_json() for structure in relaxed_structures
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

        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)

        combined.sink_parquet(self.config.output_path)

        return self.config.output_path
