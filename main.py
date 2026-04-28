import argparse
import asyncio
import json
import logging
import random
import tomllib
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from openai import APIConnectionError, AsyncOpenAI
from peft import LoraConfig
from rich.logging import RichHandler
from rich.progress import track
from transformers import pipeline, set_seed

logger = logging.getLogger(__name__)

# critic: ChatGPT

client = AsyncOpenAI()


async def get_response(
    model: str,
    dev_input: str,
    user_input: str,
    max_retries: int,
) -> str:
    for i in range(max_retries):
        try:
            response = await client.responses.create(
                model=model,
                input=[
                    {"role": "developer", "content": dev_input},
                    {"role": "user", "content": user_input},
                ],
            )
        # openai burst rate limit
        except APIConnectionError:
            await asyncio.sleep(2**i * (1 + random.random() * 0.1))
            continue
        return response.output_text
    raise RuntimeError("critic model failed to return feedback after retries")


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
    run_name = output_config.pop("run_name", datetime.now().strftime("%Y%m%d%H%M%S"))
    run_path = output_root / run_name
    run_path.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(run_path / "app.log")
    fh.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)-8s %(name)s: %(message)s")
    )
    root.addHandler(fh)

    exp_config = config["experiment"]
    set_seed(exp_config["seed"])
    mode = exp_config["mode"]
    if mode not in ["scalar", "textual"]:
        raise ValueError(f"unknown mode: {mode}")

    with open(run_path / "config.json", "w") as f:
        json.dump(config, f)

    result_path = run_path / "result.jsonl"

    env = Environment(
        loader=FileSystemLoader("prompts"), autoescape=False, undefined=StrictUndefined
    )
    artist_prompt_init = env.get_template("artist/init.jinja")
    artist_prompt_revise = env.get_template("artist/revise.jinja")
    critic_prompt_dev = env.get_template(f"critic/{mode}/dev.jinja")
    critic_prompt_user = env.get_template(f"critic/{mode}/user.jinja")

    artist_config = config["artist"]
    pipe = pipeline(model=artist_config["model"])

    generate_config = artist_config["generate"]
    max_new_tokens = generate_config["max_new_tokens"]
    low = int(max_new_tokens * 0.45)
    high = int(max_new_tokens * 0.65)
    artist_prompt = artist_prompt_init.render(words_low=low, words_high=high)

    critic_config = config["critic"]

    if mode == "scalar":
        # trainer = DPOTrainer(model=pipe.model)
        pairing_config = config["pairing"]
        top_k = pairing_config["top_k"]
        bottom_k = pairing_config["bottom_k"]

        for step in track(range(exp_config["num_steps"]), description="Looping..."):
            model_output = pipe(
                [{"role": "user", "content": artist_prompt}],
                return_full_text=False,
                **generate_config,
            )
            model_output = [o["generated_text"].lstrip() for o in model_output]

            critic_rank = asyncio.run(
                get_response(
                    model=critic_config["model"],
                    dev_input=critic_prompt_dev.render(
                        total=len(model_output), top_k=top_k, bottom_k=bottom_k
                    ),
                    user_input=critic_prompt_user.render(texts=model_output),
                    max_retries=critic_config["max_retries"],
                )
            )

            try:
                top, bottom = critic_rank.split("\n")
                top = [int(i) for i in top.split(",")]
                if len(top) != top_k:
                    raise ValueError(f"top index length {len(top)}, should be {top_k}")
                bottom = [int(i) for i in bottom.split(",")]
                if len(bottom) != bottom_k:
                    raise ValueError(
                        f"bottom index length {len(bottom)}, should be {bottom_k}"
                    )
            except Exception as e:
                logger.warning(f"step {step}: critic model return format invalid ({e})")
                continue

    elif mode == "textual":
        model_output = pipe(
            [{"role": "user", "content": artist_prompt}],
            return_full_text=False,
            **generate_config,
        )
        model_output = model_output[0]["generated_text"].lstrip()

        with open(result_path, "a") as f:
            json.dump({"step": 0, "artist": model_output}, f)
            f.write("\n")

        for step in track(range(exp_config["num_steps"]), description="Looping..."):
            critic_feedback = asyncio.run(
                get_response(
                    model=critic_config["model"],
                    dev_input=critic_prompt_dev,
                    user_input=critic_prompt_user.render(text=model_output),
                    max_retries=critic_config["max_retries"],
                )
            )

            artist_prompt = artist_prompt_revise.render(
                current=model_output,
                feedback=critic_feedback,
                words_low=low,
                words_high=high,
            )

            model_output = pipe(
                [{"role": "user", "content": artist_prompt}],
                return_full_text=False,
                **generate_config,
            )
            model_output = model_output[0]["generated_text"].lstrip()

            with open(result_path, "a") as f:
                json.dump(
                    {
                        "step": step + 1,
                        "critic": critic_feedback,
                        "artist": model_output,
                    },
                    f,
                )
                f.write("\n")


if __name__ == "__main__":
    main()
