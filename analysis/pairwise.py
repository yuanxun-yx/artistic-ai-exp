import argparse
from pathlib import Path

import polars as pl

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default="analysis.csv")
    args = parser.parse_args()

    df = pl.read_csv(args.input)
    df = df.with_columns(
        pl.when(pl.col("swap"))
        .then(4 - pl.col("choice"))
        .otherwise(pl.col("choice"))
        .alias("choice")
    ).drop("swap")

    df = (
        df.filter(pl.col("step").is_null())
        .group_by(["mode", "choice"])
        .agg(pl.len().alias("count"))
        .pivot("mode", index="choice", values="count")
        .fill_null(0)
        .sort("choice")
    )
    df.write_csv(args.output)
