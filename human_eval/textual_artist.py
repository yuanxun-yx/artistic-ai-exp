import argparse
import json
import random
import sys
from pathlib import Path

from pairwise_widget import PairwiseWidget
from PySide6.QtWidgets import QApplication
from result_io import iter_result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default="textual_artist_eval.json")
    args, qt_args = parser.parse_known_args()

    runs = []

    for path, seed in iter_result(args.input):
        with open(path, "r") as f:
            lines = f.readlines()

        initial = json.loads(lines[0])["artist"]
        final = json.loads(lines[-1])["artist"]

        runs.append(
            {
                "seed": seed,
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
