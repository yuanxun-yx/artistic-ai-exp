import argparse
import json
import random
import sys
from pathlib import Path

from pairwise_widget import PairwiseWidget
from PySide6.QtWidgets import QApplication
from result_io import get_texts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--textual", type=Path)
    parser.add_argument("--scalar", type=Path)
    parser.add_argument("--textual_step", type=int, default=2)
    parser.add_argument("--scalar_step", type=int, default=10)
    parser.add_argument("--output", type=Path, default="pairwise.json")
    args, qt_args = parser.parse_known_args()

    textual_init, textual_final = get_texts(
        root=args.textual, final_step=args.textual_step, mode="textual"
    )
    scalar_init, scalar_final = get_texts(
        root=args.scalar, final_step=args.scalar_step, mode="scalar"
    )

    pairs = []
    pairs += list(zip(textual_init, textual_final))
    pairs += list(zip(scalar_init, scalar_final))

    random.shuffle(textual_final)
    random.shuffle(scalar_final)
    pairs += list(zip(textual_final, scalar_final))

    random.shuffle(pairs)

    pairs = [pair if random.choice((False, True)) else pair[::-1] for pair in pairs]
    texts = [tuple(i["text"] for i in pair) for pair in pairs]

    app = QApplication(qt_args)
    window = PairwiseWidget(pairs=texts)
    window.show()
    ret = app.exec()
    if ret != 0:
        sys.exit(ret)

    result = []
    for choice, pair in zip(window.result, pairs):
        for i in pair:
            i.pop("text", None)
        result.append({"pair": pair, "choice": choice})

    with args.output.open("w") as f:
        json.dump(result, f, indent=2)
