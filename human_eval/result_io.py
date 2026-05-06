import json
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


def get_texts(
    root: Path,
    final_step: int,
    mode: str,
) -> tuple[list[dict], list[dict]]:
    init = []
    final = []
    for path, seed in iter_result(root):
        with path.open("r") as f:
            lines = f.readlines()
        for i, lst in zip((0, final_step), (init, final)):
            texts = json.loads(lines[i])["artist"]
            lst += [
                {"mode": mode, "seed": seed, "index": idx, "text": text}
                for idx, text in enumerate(texts)
            ]
    return init, final
