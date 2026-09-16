import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import polars as pl

    return (pl,)


@app.cell
def _(pl):
    df = pl.read_parquet(
        "/media/korotnev/7D7F-C362/lcdm/data/materials_project/mp_energies.parquet"
    )
    return (df,)


@app.cell
def _(df):
    import json

    from pymatgen.core import Structure

    structure_ase = Structure.from_dict(
        json.loads(df[0]["structure"].item())
    ).to_ase_atoms()


@app.cell
def _(structure):
    # from upet.ase import UPETCalculator
    from upet.calculator import UPETCalculator

    calculator = UPETCalculator(model="pet-mad-s", version="1.5.0", device="cpu")
    structure.calc = calculator

    energy = structure.get_potential_energy()
    forces = structure.get_forces()
    return (UPETCalculator,)


@app.cell
def _(ModelConfig, UPETCalculator, UPETConfig, pl):
    import dataclasses

    @dataclasses.dataclass(slots=True)
    class CalculatorConfig:
        pass

    class CalculatorBase:
        def __init__(self, config: ModelConfig):
            self.config = config

        def calc_df(self, df: pl.DataFrame) -> pl.DataFrame:
            pass

    @dataclasses.dataclass(slots=True)
    class CalculatorUPETConfig(CalculatorConfig):
        model: str = "pet-mad-s"
        version: str = "1.5.0"
        device: str = "cpu"

    class CalculatorUPETCalculator(CalculatorBase):
        def __init__(self, config: UPETConfig):
            self.config = config
            self.calculator = UPETCalculator(model=config.model)

        def calc_df(self, df: pl.DataFrame):
            pass


if __name__ == "__main__":
    app.run()
