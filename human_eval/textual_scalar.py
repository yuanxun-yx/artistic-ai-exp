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
    parser.add_argument("--textual", type=Path)
    parser.add_argument("--scalar", type=Path)
    parser.add_argument("--output", type=Path, default="textual_scalar_eval.json")
    args, qt_args = parser.parse_known_args()

    textual = []
    for path, seed in iter_result(args.textual):
        with open(path, "r") as f:
            line = f.readlines()[-1]
        text = json.loads(line)["artist"]
        textual.append({"seed": seed, "text": text})

    scalar = []
    for path, seed in iter_result(args.scalar):
        with open(path, "r") as f:
            line = f.readlines()[-1]
        text = json.loads(line)["completions"]
        scalar += [{"seed": seed, "index": i, "text": t} for i, t in enumerate(text)]

    random.shuffle(textual)
    random.shuffle(scalar)

    n = min(len(textual), len(scalar))

    pairs = [{"textual": t, "scalar": s} for t, s in zip(textual[:n], scalar[:n])]

    text = []
    for p in pairs:
        if random.choice([True, False]):
            p["order"] = ["textual", "scalar"]
        else:
            p["order"] = ["scalar", "textual"]

        text.append(tuple(p[m].pop("text") for m in p["order"]))

    app = QApplication(qt_args)
    window = PairwiseWidget(pairs=text)
    window.show()
    ret = app.exec()
    if ret != 0:
        sys.exit(ret)

    for choice, r in zip(window.result, pairs):
        r["choice"] = choice

    with open(args.output, "w") as f:
        json.dump(pairs, f, indent=2)
