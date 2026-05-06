import argparse
import sys
from pathlib import Path

import polars as pl
from pairwise_widget import PairwiseWidget
from PySide6.QtWidgets import QApplication
from result_io import read_scalar, read_textual


def get_final_pairs(
    df: pl.DataFrame,
    step: int,
    col: str,
) -> pl.DataFrame:
    init = df.filter(pl.col("step") == 0)
    final = df.filter(pl.col("step") == step)

    pairs = init.join(final, on=["seed", "idx"], suffix=".final").select(
        "seed",
        "idx",
        pl.col(col).alias(f"{col}.init"),
        f"{col}.final",
    )
    return pairs


def rand_swap_columns(df: pl.DataFrame, col1: str, col2: str) -> pl.DataFrame:
    df = df.with_columns(swap=pl.int_range(pl.len()).shuffle() % 2 == 0)
    df = df.with_columns(
        left=pl.when(pl.col("swap")).then(pl.col(col2)).otherwise(pl.col(col1)),
        right=pl.when(pl.col("swap")).then(pl.col(col1)).otherwise(pl.col(col2)),
    )
    df = df.drop(col1, col2)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--textual", type=Path)
    parser.add_argument("--scalar", type=Path)
    parser.add_argument("--textual_step", type=int, default=2)
    parser.add_argument("--scalar_step", type=int, default=9)
    parser.add_argument("--n_scalar_pairs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default="pairwise.csv")
    args, qt_args = parser.parse_known_args()

    textual = read_textual(args.textual)
    scalar = read_scalar(args.scalar)

    pl.set_random_seed(args.seed)

    # scalar: model vs ref
    scalar_ref = scalar.sample(args.n_scalar_pairs)

    textual = get_final_pairs(df=textual, step=args.textual_step, col="artist")
    scalar = get_final_pairs(df=scalar, step=args.scalar_step, col="artist")

    textual = textual.with_columns(pl.lit("t").alias("mode"))
    scalar = scalar.with_columns(pl.lit("s").alias("mode"))

    init_final = pl.concat([textual, scalar])

    scalar_ref = rand_swap_columns(df=scalar_ref, col1="artist", col2="ref")
    init_final = rand_swap_columns(
        df=init_final, col1="artist.init", col2="artist.final"
    )

    scalar_ref = scalar_ref.with_columns(pl.lit("s").alias("mode")).drop("critic")

    pairs = pl.concat([init_final, scalar_ref], how="diagonal")
    pairs = pairs.sample(fraction=1, shuffle=True)

    texts = list(pairs.select(["left", "right"]).iter_rows())
    pairs = pairs.drop(["left", "right"])

    app = QApplication(qt_args)
    window = PairwiseWidget(pairs=texts)
    window.show()
    ret = app.exec()
    if ret != 0:
        sys.exit(ret)

    pairs = pairs.with_columns(pl.Series("choice", window.result))

    pairs.write_csv(args.output)
