import asyncio
from datetime import datetime
import os
import json
import torch
import numpy as np
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from peft import LoraConfig
from openai import AsyncOpenAI
from transformers import HfArgumentParser, AutoTokenizer
from dataclasses import dataclass

ARTIST_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
CRITIC_MODEL_NAME = "gpt-5.1"
with open("prompts/artist.txt") as f:
    ARTIST_PROMPT = f.read()
with open("prompts/critic.txt") as f:
    CRITIC_PROMPT = f.read()

@dataclass
class ScriptArguments:
    critic_max_retries: int = 3
    critic_budget: int = 500
    max_work_tokens: int = 320
    num_steps: int = 200
    logging_steps: int = 10
    log_dir: str = "logs"
    run_name: str = None

class Budget:
    def __init__(self, max_calls: int):
        assert max_calls > 0
        self.remaining = max_calls
        self._lock = asyncio.Lock()

    async def consume(self):
        async with self._lock:
            if self.remaining <= 0:
                raise RuntimeError("Budget exhausted")
            self.remaining -= 1

def print_log(msg: str):
    print(f"[{datetime.now().isoformat()}] {msg}")

# critic: ChatGPT

client = AsyncOpenAI()

def build_critic_prompt(a: str, b: str) -> str:
    return CRITIC_PROMPT.replace("{{A}}", a).replace("{{B}}", b)

async def get_critic_choice(a: str, b: str, budget: Budget, max_retries: int) -> bool:
    prompt = build_critic_prompt(a, b)
    for _ in range(max_retries):
        await budget.consume()

        response = await client.responses.create(
            model=CRITIC_MODEL_NAME,
            input=prompt
        )
        out = response.output_text.strip()
        if out == 'A':
            return True
        if out == 'B':
            return False
    raise RuntimeError("Critic failed to return valid A/B after retries")

async def batch_judge(pairs: tuple[str, str], budget: Budget, max_retries: int):
    tasks = [
        get_critic_choice(a, b, budget, max_retries)
        for a, b in pairs
    ]
    return await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args, = parser.parse_args_into_dataclasses()
    if script_args.run_name is None:
        script_args.run_name = datetime.now().strftime("%Y%m%d%H%M%S")
    os.makedirs(script_args.log_dir, exist_ok=True)

    # default params
    ppo_config = PPOConfig(
        learning_rate=1e-5,
        batch_size=4,
        mini_batch_size=4,
        ppo_epochs=2
    )
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    tokenizer = AutoTokenizer.from_pretrained(ARTIST_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy = AutoModelForCausalLMWithValueHead.from_pretrained(
        ARTIST_MODEL_NAME,
        peft_config=lora_config,
        dtype=torch.bfloat16,
        device_map="auto"
    )

    trainer = PPOTrainer(
        config=ppo_config,
        model=policy,
        tokenizer=tokenizer
    )

    device = next(trainer.model.parameters()).device

    budget = Budget(max_calls=script_args.critic_budget)

    for step in range(script_args.num_steps):

        # generate pairs
        prompt_ids = trainer.tokenizer(ARTIST_PROMPT, padding=False, truncation=True)["input_ids"]
        prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device)

        response_tensors = trainer.generate(
            [prompt_tensor] * trainer.config.batch_size,
            max_new_tokens=script_args.max_work_tokens,
            # default values
            temperature=1.0,
            top_p=0.9
        )
        responses = []
        for r_ids in response_tensors:
            gen_ids = r_ids[len(prompt_ids):]
            responses.append(tokenizer.decode(gen_ids, skip_special_tokens=True).lstrip())

        pair_range = range(0, len(responses), 2)
        tensor_pairs = [(response_tensors[i], response_tensors[i + 1]) for i in pair_range]
        text_pairs = [(responses[i], responses[i + 1]) for i in pair_range]

        results = asyncio.run(batch_judge(text_pairs, budget, script_args.critic_max_retries))

        kept_response_tensors = []
        rewards = []

        result_jsonl = []

        for text_pair, tensor_pair, is_a_winner in zip(text_pairs, tensor_pairs, results):
            if isinstance(is_a_winner, RuntimeError):
                continue

            assert isinstance(is_a_winner, bool)

            kept_response_tensors.append(tensor_pair[0])
            kept_response_tensors.append(tensor_pair[1])

            if is_a_winner:
                rewards.extend([1.0, -1.0])
            else:
                rewards.extend([-1.0, 1.0])

            result_jsonl.append(json.dumps(
                dict(step=step, a=text_pair[0], b=text_pair[1], is_a_winner=is_a_winner)))

        if not rewards:
            print(f"step {step} failed: no valid critic judgements")
            continue

        with open(f"{script_args.log_dir}/result_{script_args.run_name}.jsonl", "a") as f:
            for l in result_jsonl:
                f.write(l + "\n")

        query_tensors = [prompt_tensor] * len(kept_response_tensors)
        reward_tensors = [torch.tensor([r], dtype=torch.float32, device=device) for r in rewards]

        stats = trainer.step(query_tensors, kept_response_tensors, reward_tensors)
        for k, v in stats.items():
            if isinstance(v, np.ndarray):
                stats[k] = v.tolist()

        with open(f"{script_args.log_dir}/stat_{script_args.run_name}.jsonl", "a") as f:
            json.dump(stats, f)
            f.write("\n")

        if step % script_args.logging_steps == 0:
            print_log(f"step {step}")
