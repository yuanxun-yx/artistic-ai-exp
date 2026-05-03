from pathlib import Path


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
