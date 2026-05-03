import argparse
from pathlib import Path

import polars as pl

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default="textual_scalar.csv")
    args = parser.parse_args()

    df = pl.read_json(args.input)
    df = df.with_columns(
        pl.when(pl.col("order") == pl.lit(["scalar", "textual"]))
        .then(4 - pl.col("choice"))
        .otherwise(pl.col("choice"))
        .alias("result")
    )
    df = df.select(pl.col("result").value_counts()).unnest().sort("result")
    df.write_csv(args.output)
