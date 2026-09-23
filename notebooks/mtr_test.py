import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import polars as pl
    df = pl.read_parquet("mp_energies (1).parquet")
    df[0]
    return df, pl


@app.cell
def _(df):
    import json 
    from pymatgen.core import Structure
    structure_ase = Structure.from_dict(json.loads(df[0]["structure"].item())).to_ase_atoms()
    return Structure, json


@app.cell
def _(Structure, df, json, pl):
    import dataclasses
    from upet.ase import UPETCalculator

    @dataclasses.dataclass(slots=True)
    class CalculatorConfig:
        pass

    class CalculatorBase:
        def __init__(self, config: CalculatorConfig):
            self.config = config

        def calc_df(self, df: pl.DataFrame) -> pl.DataFrame:
            pass
    
    @dataclasses.dataclass(slots=True)
    class CalculatorUPETConfig(CalculatorConfig):
        model: str = "pet-mad-s"
        version: str = "1.5.0"
        device: str = "cpu"

    class CalculatorUPETCalculator(CalculatorBase):
        def __init__(self, config: CalculatorUPETConfig):
            super().__init__(config)
            self.calculator = UPETCalculator(
                model=config.model,
                version=config.version,
                device=config.device
            )
        def calc_structure(self, structure_json: str) -> float:
            structure = Structure.from_dict(json.loads(structure_json))
            atoms = structure.to_ase_atoms()
            atoms.calc = self.calculator
            energy = atoms.get_potential_energy()
            return energy

        def calc_df(self, df: pl.DataFrame) -> pl.DataFrame:
            energies = []
            for structure_json in df["structure"]:
                energy = self.calc_structure(structure_json)
                structure = Structure.from_dict(json.loads(structure_json))
                n_atoms = len(structure)
                energies.append(energy)
            return df.with_columns(pl.Series("upet_energy", energies))

    config = CalculatorUPETConfig()
    calc = CalculatorUPETCalculator(config)
    
    result = calc.calc_df(df.head(5))
    result
    return


if __name__ == "__main__":
    app.run()
