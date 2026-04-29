import json
import logging
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from rich.progress import track
from torch.nn.functional import log_softmax, logsigmoid
from transformers import AutoModelForCausalLM, AutoTokenizer

from critic import get_response
from prompt import compute_length_bounds
from utils import get_jinja_env

logger = logging.getLogger(__name__)


def sequence_log_prob(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_len: int,
) -> torch.Tensor:
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[:, :-1, :]  # next token predicted
    labels = input_ids[:, 1:]  # next token actual
    log_probs = log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(
        dim=-1,
        index=labels.unsqueeze(-1),
    ).squeeze(-1)
    return token_log_probs[:, prompt_len - 1 :].sum(dim=-1)


def loop(config: dict, run_path: Path) -> None:
    """
    trl.OnlineDPOTrainer requires a reward model. trl.DPOTrainer doesn't support online training.
    Therefore, I have to implement preference based online DPO from scratch here.
    """

    result_path = run_path / "result.jsonl"

    env = get_jinja_env("prompts")
    artist_prompt = env.get_template("artist/init.jinja")
    critic_prompt_dev = env.get_template("critic/scalar/dev.jinja")
    critic_prompt_user = env.get_template("critic/scalar/user.jinja")

    artist_config = config["artist"]
    model_name = artist_config["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # from_pretrained() sets model.eval() by default
    ref_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    # make sure reference model is frozen
    ref_model.requires_grad_(False)
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
    prompt_len = artist_prompt["input_ids"].shape[1]
    # remove prompt by string length, consistent with transformers.Pipeline behavior in textual mode
    prompt_str_len = len(
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

    train_config = config["training"]
    optimizer = torch.optim.AdamW(
        params=(p for p in model.parameters() if p.requires_grad),
        **train_config["optimizer"],
    )

    for step in track(range(train_config["num_steps"]), description="Looping..."):
        model.eval()
        # torch.no_grad() is enabled
        model_output = model.generate(
            **artist_prompt,
            **generate_config,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        text = tokenizer.decode(
            model_output,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        text = [t[prompt_str_len:].lstrip() for t in text]

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

        model.train()
        attention_mask = model_output != tokenizer.pad_token_id
        log_pi = sequence_log_prob(
            model=model,
            input_ids=model_output,
            attention_mask=attention_mask,
            prompt_len=prompt_len,
        )
        with torch.no_grad():
            log_ref = sequence_log_prob(
                model=ref_model,
                input_ids=model_output,
                attention_mask=attention_mask,
                prompt_len=prompt_len,
            )

        pairs = torch.cartesian_prod(
            torch.as_tensor(top),
            torch.as_tensor(bottom),
        ).to(model.device)

        chosen = pairs[:, 0]
        rejected = pairs[:, 1]

        pi_margin = log_pi[chosen] - log_pi[rejected]
        ref_margin = log_ref[chosen] - log_ref[rejected]
        logits = train_config["dpo"]["beta"] * (pi_margin - ref_margin)
        loss = -logsigmoid(logits).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        stat = {
            "loss": loss.item(),
            "pi_margin": pi_margin.detach().cpu().tolist(),
            "ref_margin": ref_margin.detach().cpu().tolist(),
            "logit": logits.detach().cpu().tolist(),
        }

        logger.info(f"step {step}: {stat}")

        with open(result_path, "a") as f:
            json.dump(
                {
                    "step": step,
                    "artist": text,
                    "critic": {"top": top, "bottom": bottom},
                    "training": stat,
                },
                f,
            )
            f.write("\n")
