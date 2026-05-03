import argparse
from pathlib import Path

import polars as pl

CHOICE_TYPE = ["best", "worst"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default="scalar_artist.csv")
    args = parser.parse_args()

    df = pl.read_json(args.input)
    df = df.unnest("choice")
    df = df.with_columns(pl.col("set").list.eval(pl.element().struct.field("epoch")))
    for t in CHOICE_TYPE:
        df = df.with_columns(pl.col("set").list.get(pl.col(t)).alias(t))
    long = df.select(CHOICE_TYPE).unpivot(variable_name="type", value_name="epoch")
    table = (
        long.group_by(["type", "epoch"])
        .len()
        .pivot(index="type", on="epoch", values="len")
        .fill_null(0)
    )

    table.write_csv(args.output)
