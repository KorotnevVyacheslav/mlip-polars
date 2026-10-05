import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Import and dataloading
    """)
    return


@app.cell
def _():
    import json

    import marimo as mo
    import polars as pl
    from pymatgen.core import Structure

    file_path: str = (
        "/mnt/c/Users/Huawei/Downloads/mp_energies (2).parquet"
    )

    df = pl.read_parquet(file_path).head(10)
    ldf = pl.scan_parquet(file_path).head(2)
    return Structure, df, json, ldf, mo, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## UPET calculator
    """)
    return


@app.cell
def _(Structure, json, pl):
    import dataclasses

    import ase
    from upet.calculator import UPETCalculator

    @dataclasses.dataclass(slots=True)
    class CalculatorConfig:
        pass

    class CalculatorBase:
        def __init__(self, config: CalculatorConfig):
            self.config = config

        def calculate_df(self, df: pl.DataFrame) -> pl.DataFrame:
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
                device=config.device,
            )

        @staticmethod
        def calc_structure(
            structure_json: str,
            calculator: ase.calculators.calculator.Calculator,
        ) -> float:
            structure = Structure.from_dict(json.loads(structure_json))
            atoms = structure.to_ase_atoms()
            atoms.calc = calculator
            energy = atoms.get_potential_energy()

            return {
                "energy": energy,
                "structure_output": structure.to_json(),
            }

        def calculate_df(self, df: pl.DataFrame) -> pl.DataFrame:
            return df.with_columns(
                pl.col("structure")
                .map_elements(
                    lambda json_str: self.calc_structure(json_str, self.calculator),
                    return_dtype=pl.Struct(
                        {"energy": pl.Float64, "structure_output": pl.String}
                    ),
                )
                .struct.unnest()
            )

    return CalculatorUPETCalculator, CalculatorUPETConfig


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Calculator initialization
    """)
    return


@app.cell
def _(CalculatorUPETCalculator, CalculatorUPETConfig):
    config = CalculatorUPETConfig()
    calculator = CalculatorUPETCalculator(config)
    return (calculator,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## LazyFrame transform
    """)
    return


@app.cell
def _(calculator, ldf):
    ldf_calculated = calculator.calculate_df(ldf)
    ldf_calculated
    return (ldf_calculated,)


@app.cell
def _(ldf_calculated):
    ldf_calculated.sink_parquet("result.parquet")
    return


@app.cell
def _(pl):
    result = pl.read_parquet("result.parquet")
    result
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dataframe transform
    """)
    return


@app.cell
def _(calculator, df):
    df_calculated = calculator.calculate_df(df.head(5))
    df_calculated
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
