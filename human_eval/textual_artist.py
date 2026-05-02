import argparse
import json
import random
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from pairwise_widget import PairwiseWidget

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default="textual_artist_eval.json")
    args, qt_args = parser.parse_known_args()

    textual_result = args.input
    if not textual_result.is_dir():
        raise FileNotFoundError(f"{textual_result} is not a directory")

    runs = []

    for seed in textual_result.iterdir():
        if not seed.is_dir():
            continue
        path = seed / "result.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"{path} is not a file")

        with open(path, "r") as f:
            lines = f.readlines()

        initial = json.loads(lines[0])["artist"]
        final = json.loads(lines[-1])["artist"]

        runs.append(
            {
                "seed": int(seed.name),
                "initial": initial,
                "final": final,
                "reversed": random.choice([True, False]),
            }
        )

    random.shuffle(runs)
    pairs = []
    for r in runs:
        initial = r.pop("initial")
        final = r.pop("final")
        if r["reversed"]:
            pairs.append((final, initial))
        else:
            pairs.append((initial, final))

    app = QApplication(qt_args)
    window = PairwiseWidget(pairs=pairs)
    window.show()
    ret = app.exec()
    if ret != 0:
        sys.exit(ret)

    for choice, r in zip(window.result, runs):
        r["choice"] = choice

    with open(args.output, "w") as f:
        json.dump(runs, f, indent=2)
