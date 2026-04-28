import json
import logging
from pathlib import Path

from peft import LoraConfig, get_peft_model
from rich.progress import track
from transformers import AutoModelForCausalLM, AutoTokenizer

from critic import get_response
from prompt import compute_length_bounds
from utils import get_jinja_env

logger = logging.getLogger(__name__)


def loop(config: dict, run_path: Path) -> None:
    env = get_jinja_env("prompts")
    artist_prompt = env.get_template("artist/init.jinja")
    critic_prompt_dev = env.get_template("critic/scalar/dev.jinja")
    critic_prompt_user = env.get_template("critic/scalar/user.jinja")

    artist_config = config["artist"]
    model_name = artist_config["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    if "lora" in artist_config:
        lora_config = LoraConfig(**artist_config["lora"])
        model = get_peft_model(model, lora_config)

    generate_config = artist_config["generate"]
    low, high = compute_length_bounds(generate_config["max_new_tokens"])
    artist_prompt = [
        {
            "role": "user",
            "content": artist_prompt.render(words_low=low, words_high=high),
        }
    ]
    artist_prompt = tokenizer.apply_chat_template(
        artist_prompt,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)
    # remove prompt by string length, consistent with transformers.Pipeline behavior in textual mode
    prompt_len = len(
        tokenizer.decode(
            artist_prompt.input_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
    )

    critic_config = config["critic"]

    pairing_config = config["pairing"]
    top_k = pairing_config["top_k"]
    bottom_k = pairing_config["bottom_k"]

    exp_config = config["experiment"]
    for step in track(range(exp_config["num_steps"]), description="Looping..."):
        model_output = model.generate(**artist_prompt, **generate_config)
        text = tokenizer.decode(
            model_output,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        text = [t[prompt_len:].lstrip() for t in text]

        critic_rank = get_response(
            model=critic_config["model"],
            dev_input=critic_prompt_dev.render(
                total=len(text), top_k=top_k, bottom_k=bottom_k
            ),
            user_input=critic_prompt_user.render(texts=text),
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
