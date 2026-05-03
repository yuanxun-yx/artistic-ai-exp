import argparse
from pathlib import Path

import polars as pl

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default="textual_artist")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    df = pl.read_json(args.input)

    df = df.with_columns(
        pl.when(pl.col("reversed"))
        .then(4 - pl.col("choice"))
        .otherwise(pl.col("choice"))
        .alias("result")
    )
    df = df.select(["seed", "result"]).sort("seed")
    df.write_csv(args.output / "run.csv")

    df = df.select(pl.col("result").value_counts()).unnest().sort("result")
    df.write_csv(args.output / "count.csv")
