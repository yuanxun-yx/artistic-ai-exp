import json
import logging
from pathlib import Path

from rich.progress import track
from transformers import pipeline

from critic import get_response
from prompt import compute_length_bounds
from utils import get_jinja_env

logger = logging.getLogger(__name__)


def loop(config: dict, run_path: Path) -> Path:
    env = get_jinja_env("prompts")
    artist_prompt = env.get_template("artist/init.jinja")
    critic_prompt_dev = env.get_template("critic/scalar/dev.jinja")
    critic_prompt_user = env.get_template("critic/scalar/user.jinja")

    artist_config = config["artist"]
    pipe = pipeline(model=artist_config["model"])

    generate_config = artist_config["generate"]
    low, high = compute_length_bounds(generate_config["max_new_tokens"])
    artist_prompt = artist_prompt.render(words_low=low, words_high=high)

    critic_config = config["critic"]

    pairing_config = config["pairing"]
    top_k = pairing_config["top_k"]
    bottom_k = pairing_config["bottom_k"]

    exp_config = config["experiment"]
    for step in track(range(exp_config["num_steps"]), description="Looping..."):
        model_output = pipe(
            [{"role": "user", "content": artist_prompt}],
            return_full_text=False,
            **generate_config,
        )
        model_output = [o["generated_text"].lstrip() for o in model_output]

        critic_rank = get_response(
            model=critic_config["model"],
            dev_input=critic_prompt_dev.render(
                total=len(model_output), top_k=top_k, bottom_k=bottom_k
            ),
            user_input=critic_prompt_user.render(texts=model_output),
            max_retries=critic_config["max_retries"],
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
