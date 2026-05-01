import json
import logging
from pathlib import Path

from datasets import Dataset
from omegaconf import DictConfig
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl.experimental.online_dpo import OnlineDPOConfig

from critic import get_response
from prompt import compute_length_bounds
from trainer import MyOnlineDPOTrainer
from utils import get_jinja_env

logger = logging.getLogger(__name__)


class JsonlLogCallback(TrainerCallback):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or logs is None:
            return
        record = {"step": state.global_step, "epoch": state.epoch, **logs}
        with self.path.open("a") as f:
            json.dump(record, f)
            f.write("\n")


def loop(config: DictConfig, run_path: Path) -> None:
    env = get_jinja_env("prompts")
    artist_prompt = env.get_template("artist/init.jinja")
    critic_prompt_dev = env.get_template("critic/scalar/dev.jinja")
    critic_prompt_user = env.get_template("critic/scalar/user.jinja")

    artist_config = config.artist
    generate_config = artist_config.generate
    low, high = compute_length_bounds(generate_config.max_new_tokens)
    artist_prompt = [
        {
            "role": "user",
            "content": artist_prompt.render(words_low=low, words_high=high),
        }
    ]
    model = AutoModelForCausalLM.from_pretrained(artist_config.model)
    tokenizer = AutoTokenizer.from_pretrained(artist_config.model, padding="left")
    train_dataset = Dataset.from_list([{"prompt": artist_prompt}])

    critic_config = config.critic

    pairing_config = config.pairing
    top_k = pairing_config.top_k
    bottom_k = pairing_config.bottom_k

    def get_preference(text: list[str]):
        critic_rank = get_response(
            model=critic_config.model,
            dev_input=critic_prompt_dev.render(
                total=len(text), top_k=top_k, bottom_k=bottom_k
            ),
            user_input=critic_prompt_user.render(texts=text),
            max_retries=critic_config.max_retries,
        )

        top, bottom = critic_rank.split("\n")
        top = [int(i) for i in top.split(",")]
        if len(top) != top_k:
            raise ValueError(f"top index length {len(top)}, should be {top_k}")
        bottom = [int(i) for i in bottom.split(",")]
        if len(bottom) != bottom_k:
            raise ValueError(f"bottom index length {len(bottom)}, should be {bottom_k}")
        return top, bottom

    train_config = config.get("training", {})
    lora_config = train_config.pop("lora", None)
    if lora_config is not None:
        lora_config = LoraConfig(**lora_config)
    dpo_config = train_config.pop("dpo", {})
    train_args = train_config.pop("train_args", {})
    args = OnlineDPOConfig(
        generation_kwargs=artist_config.generate,
        output_dir=str(run_path),
        **dpo_config,
        **train_config,
    )
    trainer = MyOnlineDPOTrainer(
        model=model,
        preference_func=get_preference,
        train_dataset=train_dataset,
        args=args,
        processing_class=tokenizer,
        peft_config=lora_config,
        callbacks=[JsonlLogCallback(run_path / "metrics.jsonl")],
    )
    trainer.train(**train_args)
