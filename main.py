import asyncio
import json
import logging
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from openai import APIConnectionError, AsyncOpenAI
from peft import LoraConfig
from rich.logging import RichHandler
from rich.progress import track
from transformers import AutoTokenizer, HfArgumentParser
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

with open("prompts/artist.txt") as f:
    ARTIST_PROMPT = f.read()
with open("prompts/critic.txt") as f:
    CRITIC_PROMPT = f.read()

logger = logging.getLogger(__name__)


@dataclass
class ScriptArguments:
    critic_max_retries: int = 3
    critic_budget: int = 500
    max_work_tokens: int = 192
    num_steps: int = 40
    result_dir: str = "result"
    run_name: str = None
    whitespace_ratio: float = 0.6
    weird_unicode_ratio: float = 0.01
    pair_mining_rounds: int = 10
    num_candidates: int = 8
    training_pairs: int = 2
    pair_mining: bool = False
    lora: bool = False
    artist_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    critic_model: str = "gpt-5.1"


class Budget:
    def __init__(self, max_calls: int):
        if max_calls <= 0:
            raise ValueError("max_calls should be > 0")
        self.remaining = max_calls
        self._lock = asyncio.Lock()

    async def consume(self):
        async with self._lock:
            if self.remaining <= 0:
                raise RuntimeError("Budget exhausted")
            self.remaining -= 1


def get_char_ratio(s: str, func: Callable[[str], bool]) -> float:
    total = len(s)
    non_ws = sum(1 for ch in s if func(ch))
    return non_ws / total if total > 0 else 0.0


def whitespace_ratio(s: str) -> float:
    return get_char_ratio(s, lambda ch: ch.isspace())


def weird_unicode_ratio(s: str) -> float:
    return get_char_ratio(s, lambda ch: ch in ["\ufffd"])


# critic: ChatGPT

client = AsyncOpenAI()


def build_critic_prompt(a: str, b: str) -> str:
    return CRITIC_PROMPT.replace("{{A}}", a).replace("{{B}}", b)


async def get_critic_choice(
    a: str, b: str, model: str, budget: Budget, max_retries: int
) -> bool:
    prompt = build_critic_prompt(a, b)
    for i in range(max_retries):
        await budget.consume()
        try:
            response = await client.responses.create(model=model, input=prompt)
        # openai burst rate limit
        except APIConnectionError:
            await asyncio.sleep(2**i * (1 + random.random() * 0.1))
            continue
        out = response.output_text.strip()
        if out == "A":
            return True
        if out == "B":
            return False
    raise RuntimeError("Critic failed to return valid A/B after retries")


async def batch_judge_text(
    pairs: tuple[str, str], model: str, budget: Budget, max_retries: int
):
    tasks = [get_critic_choice(a, b, model, budget, max_retries) for a, b in pairs]
    return await asyncio.gather(*tasks, return_exceptions=True)


async def batch_judge(
    pairs: list[tuple[int, int]],
    responses: list[dict],
    model: str,
    budget: Budget,
    max_retries: int,
):
    text_pairs = [(responses[i]["text"], responses[j]["text"]) for i, j in pairs]
    return await batch_judge_text(text_pairs, model, budget, max_retries)


def main():
    logging.captureWarnings(True)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    rh = RichHandler()
    rh.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(rh)

    parser = HfArgumentParser(ScriptArguments)
    (script_args,) = parser.parse_args_into_dataclasses()
    if script_args.run_name is None:
        script_args.run_name = datetime.now().strftime("%Y%m%d%H%M%S")
    if script_args.pair_mining:
        expected = 2 * script_args.training_pairs
        if script_args.num_candidates < expected:
            raise ValueError(f"num_candidates should be >= {expected}")
        expected = script_args.num_candidates * (script_args.num_candidates - 1) // 2
        if script_args.pair_mining_rounds > expected:
            raise ValueError(f"pair_mining rounds should be <= {expected}")
        batch_size = 2 * script_args.training_pairs
    else:
        batch_size = script_args.num_candidates

    run_dir = Path(script_args.result_dir) / script_args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(run_dir / "app.log")
    fh.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)-8s %(name)s: %(message)s")
    )
    root.addHandler(fh)

    with open(run_dir / "config.json", "w") as f:
        json.dump(asdict(script_args), f)

    # default params
    ppo_config = PPOConfig(
        learning_rate=5e-6,
        batch_size=batch_size,
        mini_batch_size=batch_size,
        ppo_epochs=2,
    )
    lora_config = LoraConfig(
        r=16, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )

    tokenizer = AutoTokenizer.from_pretrained(script_args.artist_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"dtype": torch.bfloat16, "device_map": "auto"}
    if script_args.lora:
        kwargs["peft_config"] = lora_config
    policy = AutoModelForCausalLMWithValueHead.from_pretrained(
        script_args.artist_model, **kwargs
    )

    trainer = PPOTrainer(config=ppo_config, model=policy, tokenizer=tokenizer)

    device = next(trainer.model.parameters()).device

    budget = Budget(max_calls=script_args.critic_budget)

    for step in track(range(script_args.num_steps), description="PPO Training..."):
        # generate pairs
        prompt_ids = trainer.tokenizer(ARTIST_PROMPT, padding=False, truncation=True)[
            "input_ids"
        ]
        prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device)

        response_tensors = trainer.generate(
            [prompt_tensor] * script_args.num_candidates,
            max_new_tokens=script_args.max_work_tokens,
            # default values
            temperature=0.9,
            do_sample=True,
            top_p=1.0,
            top_k=0,
            pad_token_id=trainer.tokenizer.pad_token_id,
            eos_token_id=trainer.tokenizer.eos_token_id,
        )
        responses = []
        for r_ids in response_tensors:
            gen_ids = r_ids[len(prompt_ids) :]
            r = tokenizer.decode(gen_ids, skip_special_tokens=True).lstrip()
            if (
                whitespace_ratio(r) > script_args.whitespace_ratio
                or weird_unicode_ratio(r) > script_args.weird_unicode_ratio
            ):
                continue
            responses.append({"tensor": r_ids, "text": r, "score": 0})
        if len(responses) < len(response_tensors):
            logger.warning(
                f"step {step}: filtered {len(response_tensors) - len(responses)} responses"
            )

        N = len(responses)
        # pair mining
        if script_args.pair_mining:
            all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
            random.shuffle(all_pairs)
            all_pairs = all_pairs[: script_args.pair_mining_rounds]
            results = asyncio.run(
                batch_judge(
                    all_pairs,
                    responses,
                    script_args.critic_model,
                    budget,
                    script_args.critic_max_retries,
                )
            )
            pair_mining_result = {
                p: is_a_winner
                for p, is_a_winner in zip(all_pairs, results)
                if not isinstance(is_a_winner, RuntimeError)
            }
            if len(pair_mining_result) < len(all_pairs):
                logger.warning(
                    f"step {step}: lost {len(all_pairs) - len(pair_mining_result)} pairs in pair mining"
                )

            for (i, j), is_a_winner in pair_mining_result.items():
                if is_a_winner:
                    responses[i]["score"] += 1
                    responses[j]["score"] -= 1
                else:
                    responses[i]["score"] -= 1
                    responses[j]["score"] += 1

            sorted_index = np.argsort([r["score"] for r in responses]).tolist()
            training_pairs = [
                (sorted_index[-(i + 1)], sorted_index[i])
                for i in range(script_args.training_pairs)
            ]
            # retrieve results for same pairs
            training_pair_results = {
                p: pair_mining_result[p]
                for p in training_pairs
                if p in pair_mining_result
            }
            pairs_to_judge = [
                p for p in training_pairs if p not in training_pair_results
            ]
        else:
            training_pair_results = {}
            pairs_to_judge = [(i, i + 1) for i in range(0, N - 1, 2)]

        results = asyncio.run(
            batch_judge(
                pairs_to_judge,
                responses,
                script_args.critic_model,
                budget,
                script_args.critic_max_retries,
            )
        )
        for p, is_a_winner in zip(pairs_to_judge, results):
            if not isinstance(is_a_winner, RuntimeError):
                training_pair_results[p] = is_a_winner
        if len(training_pair_results) == 0:
            logger.warning(f"step {step} failed: no valid critic judgements")
            continue
        if len(training_pair_results) < len(pairs_to_judge):
            logger.warning(
                f"step {step}: lost {len(pairs_to_judge) - len(training_pair_results)} training pairs"
            )

        kept_response_tensors = []
        rewards = []

        result_jsonl = []

        for (i, j), is_a_winner in training_pair_results.items():
            kept_response_tensors.append(responses[i]["tensor"])
            kept_response_tensors.append(responses[j]["tensor"])

            if is_a_winner:
                rewards.extend([1.0, -1.0])
            else:
                rewards.extend([-1.0, 1.0])

            result_jsonl.append(
                json.dumps(
                    {
                        "step": step,
                        "a": responses[i]["text"],
                        "b": responses[j]["text"],
                        "is_a_winner": is_a_winner,
                    }
                )
            )

        with open(run_dir / "result.jsonl", "a") as f:
            for l in result_jsonl:
                f.write(l + "\n")

        query_tensors = [prompt_tensor] * len(kept_response_tensors)
        reward_tensors = [
            torch.tensor([r], dtype=torch.float32, device=device) for r in rewards
        ]

        stats = trainer.step(query_tensors, kept_response_tensors, reward_tensors)
        for k, v in stats.items():
            if isinstance(v, np.ndarray):
                stats[k] = v.tolist()

        with open(run_dir / "stat.jsonl", "a") as f:
            json.dump(stats, f)
            f.write("\n")


if __name__ == "__main__":
    main()
