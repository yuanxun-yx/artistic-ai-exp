import asyncio
import json
import logging
import random
from pathlib import Path

from datasets import Dataset
from omegaconf import DictConfig
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl.experimental.xpo import XPOConfig

from critic import get_response_batch
from prompt import compute_length_bounds
from trainer import MyXPOTrainer
from utils import get_jinja_env

logger = logging.getLogger(__name__)


class JsonlLogCallback(TrainerCallback):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or logs is None:
            return
        record = {"step": state.global_step, **logs}
        with self.path.open("a") as f:
            json.dump(record, f)
            f.write("\n")


def loop(config: DictConfig, run_path: Path) -> None:
    env = get_jinja_env("prompts")
    artist_prompt = env.get_template("artist/init.jinja")
    with open("prompts/critic/scalar/dev.txt", "r") as f:
        critic_prompt_dev = f.read()
    critic_prompt_user = env.get_template("critic/scalar/user.jinja")
    with open("prompts/artist/premise.txt", "r") as f:
        premise = f.read().splitlines()

    artist_config = config.artist
    generate_config = artist_config.generate
    low, high = compute_length_bounds(generate_config.max_new_tokens)
    model = AutoModelForCausalLM.from_pretrained(artist_config.model)
    tokenizer = AutoTokenizer.from_pretrained(artist_config.model, padding="left")
    train_dataset = Dataset.from_list(
        [
            {
                "prompt": [
                    {
                        "role": "user",
                        "content": artist_prompt.render(
                            premise=p, words_low=low, words_high=high
                        ),
                    }
                ]
            }
            for p in premise
        ]
    )

    critic_config = config.critic

    def get_preference(model_output: list[str], ref_output: list[str]) -> list[bool]:
        n = len(model_output)
        reverse = [random.choice([False, True]) for _ in range(n)]
        pair = [
            (r, m) if re else (m, r)
            for re, m, r in zip(reverse, model_output, ref_output)
        ]
        response = asyncio.run(
            get_response_batch(
                model=critic_config.model,
                dev_input=critic_prompt_dev,
                user_input=[critic_prompt_user.render(A=a, B=b) for a, b in pair],
                max_retries=critic_config.max_retries,
            )
        )
        choice = []
        for res, rev in zip(response, reverse):
            if res not in ["A", "B"]:
                raise ValueError(f"invalid response: {res}")
            choice.append((res == "A") ^ rev)

        with (run_path / "result.jsonl").open("a") as f:
            json.dump({"artist": model_output, "ref": ref_output, "critic": choice}, f)
            f.write("\n")

        return choice

    train_config = config.get("training", {})
    lora_config = train_config.pop("lora", None)
    if lora_config is not None:
        lora_config = LoraConfig(**lora_config)
    train_args = train_config.pop("train_args", {})
    args = XPOConfig(
        generation_kwargs=artist_config.generate,
        output_dir=str(run_path),
        **train_config,
    )
    trainer = MyXPOTrainer(
        model=model,
        reward_funcs=get_preference,
        train_dataset=train_dataset,
        args=args,
        processing_class=tokenizer,
        peft_config=lora_config,
        callbacks=[JsonlLogCallback(run_path / "metrics.jsonl")],
    )
    trainer.train(**train_args)
