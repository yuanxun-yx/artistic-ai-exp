import json
from pathlib import Path

from omegaconf import DictConfig
from rich.progress import track
from transformers import pipeline

from critic import get_response
from prompt import compute_length_bounds
from utils import get_jinja_env


def loop(config: DictConfig, run_path: Path) -> None:
    result_path = run_path / "result.jsonl"

    env = get_jinja_env("prompts")
    artist_prompt_init = env.get_template("artist/init.jinja")
    artist_prompt_revise = env.get_template("artist/revise.jinja")
    critic_prompt_user = env.get_template("critic/textual/user.jinja")
    with open("prompts/critic/textual/dev.txt", "r") as f:
        critic_prompt_dev = f.read()

    artist_config = config.artist
    pipe = pipeline(model=artist_config.model)

    generate_config = artist_config.generate
    low, high = compute_length_bounds(generate_config.max_new_tokens)
    artist_prompt = artist_prompt_init.render(words_low=low, words_high=high)

    critic_config = config.critic

    model_output = pipe(
        [{"role": "user", "content": artist_prompt}],
        return_full_text=False,
        **generate_config,
    )
    model_output = model_output[0]["generated_text"].lstrip()

    with open(result_path, "a") as f:
        json.dump({"epoch": 0, "artist": model_output}, f)
        f.write("\n")

    for epoch in track(
        range(config.training.num_train_epochs), description="Looping..."
    ):
        critic_feedback = get_response(
            model=critic_config.model,
            dev_input=critic_prompt_dev,
            user_input=critic_prompt_user.render(text=model_output),
            max_retries=critic_config.max_retries,
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

        with result_path.open("a") as f:
            json.dump(
                {
                    "epoch": epoch + 1,
                    "critic": critic_feedback,
                    "artist": model_output,
                },
                f,
            )
            f.write("\n")
