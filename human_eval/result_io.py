from pathlib import Path

import polars as pl


def iter_result(root: Path):
    if not root.is_dir():
        raise FileNotFoundError(f"{root} is not a directory")

    for seed in root.iterdir():
        if not seed.is_dir():
            continue
        path = seed / "result.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"{path} is not a file")
        yield path, int(seed.name)


def read_data(root: Path) -> pl.DataFrame:
    dfs = []
    for path, seed in iter_result(root):
        df = pl.read_ndjson(path)
        if "step" not in df.columns:
            df = df.with_row_index("step")
        df = df.with_columns(pl.lit(seed).alias("seed"))
        dfs.append(df)
    df = pl.concat(dfs)
    return df


def read_textual(root: Path) -> pl.DataFrame:
    df = read_data(root)

    length = pl.col("artist").list.len()

    df = df.with_columns(
        critic=pl.col("critic").fill_null(
            pl.lit(None, dtype=pl.String).repeat_by(length)
        ),
        idx=pl.int_ranges(length, dtype=pl.UInt32),
    ).explode(["artist", "critic", "idx"])

    return df


def read_scalar(root: Path) -> pl.DataFrame:
    df = read_data(root)
    df = df.with_columns(
        idx=pl.int_ranges(pl.col("artist").list.len(), dtype=pl.UInt32)
    ).explode(["artist", "critic", "ref", "idx"])
    return df
