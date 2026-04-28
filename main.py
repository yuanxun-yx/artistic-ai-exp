import argparse
import json
import logging
import tomllib
from datetime import datetime
from pathlib import Path

from rich.logging import RichHandler
from transformers import set_seed

logger = logging.getLogger(__name__)


def main():
    logging.captureWarnings(True)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    rh = RichHandler()
    rh.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(rh)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    args = parser.parse_args()

    if not args.config.is_file():
        raise FileNotFoundError(f'config file not found: "{args.config}"')
    with args.config.open("rb") as f:
        config = tomllib.load(f)

    output_config = config["output"]
    output_root = Path(output_config["root"])
    run_name = output_config.get("run_name", datetime.now().strftime("%Y%m%d%H%M%S"))
    run_path = output_root / run_name
    run_path.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(run_path / "app.log")
    fh.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)-8s %(name)s: %(message)s")
    )
    root.addHandler(fh)

    exp_config = config["experiment"]
    set_seed(exp_config["seed"])

    with open(run_path / "config.json", "w") as f:
        json.dump(config, f)

    mode = exp_config["mode"]
    if mode == "scalar":
        from scalar import loop
    elif mode == "textual":
        from textual import loop
    else:
        raise ValueError(f"unknown mode: {mode}")

    loop(config, run_path)


if __name__ == "__main__":
    main()
