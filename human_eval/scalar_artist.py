import argparse
import json
import random
import sys
from itertools import batched
from pathlib import Path

from best_worst_widget import BestWorstWidget
from PySide6.QtWidgets import QApplication
from result_io import iter_result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default="scalar_artist_eval.json")
    parser.add_argument("--n_repeats", type=int, default=2)
    parser.add_argument("--set_size", type=int, default=4)
    args, qt_args = parser.parse_known_args()

    half_set_size = args.set_size // 2

    sets = []

    for path, seed in iter_result(args.input):
        with open(path, "r") as f:
            lines = f.readlines()

        initial_completions = list(enumerate(json.loads(lines[0])["completions"]))
        final_completions = list(enumerate(json.loads(lines[-1])["completions"]))

        for _ in range(args.n_repeats):
            random.shuffle(initial_completions)
            random.shuffle(final_completions)

            initial_half_set = batched(initial_completions, half_set_size)
            final_half_set = batched(final_completions, half_set_size)

            for init, fin in zip(initial_half_set, final_half_set):
                s = []
                for epoch, pair in zip([0, -1], [init, fin]):
                    s += [{"epoch": epoch, "index": i[0], "text": i[1]} for i in pair]
                random.shuffle(s)
                sets.append({"seed": seed, "set": s})

    random.shuffle(sets)
    set_text = []
    for s in sets:
        text = []
        for b in s["set"]:
            text.append(b.pop("text"))
        set_text.append(text)

    app = QApplication(qt_args)
    window = BestWorstWidget(choices=set_text)
    window.show()
    ret = app.exec()
    if ret != 0:
        sys.exit(ret)

    for choice, s in zip(window.result, sets):
        s["choice"] = {"best": choice[0], "worst": choice[1]}

    with open(args.output, "w") as f:
        json.dump(sets, f, indent=2)
