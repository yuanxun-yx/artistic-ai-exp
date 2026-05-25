import asyncio
import json
from pathlib import Path

from omegaconf import DictConfig
from rich.progress import track
from transformers import pipeline, set_seed

from critic import get_response_batch
from prompt import compute_length_bounds
from utils import get_jinja_env


def make_input(prompts: list[str]):
    return [[{"role": "user", "content": p}] for p in prompts]


def make_output(model_output: list[list[dict[str, str]]]):
    return [d["generated_text"].lstrip() for b in model_output for d in b]


def loop(config: DictConfig, run_path: Path) -> None:
    result_path = run_path / "result.jsonl"

    train_config = config.training
    set_seed(train_config.seed)

    env = get_jinja_env("prompts")
    artist_prompt_init = env.get_template("artist/init.jinja")
    artist_prompt_revise = env.get_template("artist/revise.jinja")
    critic_prompt_user = env.get_template("critic/textual/user.jinja")
    with open("prompts/critic/textual/dev.txt", "r") as f:
        critic_prompt_dev = f.read()
    with open("prompts/artist/premise.txt", "r") as f:
        premise = f.read().splitlines()

    artist_config = config.artist
    pipe = pipeline(model=artist_config.model)

    generate_config = artist_config.generate
    low, high = compute_length_bounds(generate_config.max_new_tokens)
    artist_prompt = [
        artist_prompt_init.render(premise=p, words_low=low, words_high=high)
        for p in premise
    ]

    critic_config = config.critic

    model_output = pipe(
        make_input(artist_prompt),
        return_full_text=False,
        **generate_config,
    )
    model_output = make_output(model_output)

    with result_path.open("a") as f:
        json.dump({"step": 0, "artist": model_output}, f)
        f.write("\n")

    for step in track(range(train_config.num_steps), description="Looping..."):
        critic_feedback = asyncio.run(
            get_response_batch(
                model=critic_config.model,
                dev_input=critic_prompt_dev,
                user_input=[critic_prompt_user.render(text=o) for o in model_output],
                max_retries=critic_config.max_retries,
            )
        )

        artist_prompt = [
            artist_prompt_revise.render(
                premise=p,
                current=o,
                feedback=fb,
                words_low=low,
                words_high=high,
            )
            for p, o, fb in zip(premise, model_output, critic_feedback)
        ]

        model_output = pipe(
            make_input(artist_prompt),
            return_full_text=False,
            **generate_config,
        )
        model_output = make_output(model_output)

        with result_path.open("a") as f:
            json.dump(
                {
                    "step": step + 1,
                    "critic": critic_feedback,
                    "artist": model_output,
                },
                f,
            )
            f.write("\n")
