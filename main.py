import asyncio
import torch
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
    output_dir: str = "checkpoints"
    num_steps: int = 200
    logging_steps: int = 100

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


# rollout & reward

def rollout_batch(
    trainer: PPOTrainer,
    budget: Budget,
    max_retries: int,
    max_work_tokens: int
):

    device = next(trainer.model.parameters()).device

    # generate pairs
    prompt_ids = trainer.tokenizer(ARTIST_PROMPT, padding=False, truncation=True)["input_ids"]
    prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device)

    response_tensors = trainer.generate(
        [prompt_tensor] * trainer.config.batch_size,
        max_new_tokens=max_work_tokens,
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

    results = asyncio.run(batch_judge(text_pairs, budget, max_retries))

    kept_response_tensors = []
    rewards = []

    for pair, is_a_winner in zip(tensor_pairs, results):
        if isinstance(is_a_winner, RuntimeError):
            continue

        assert isinstance(is_a_winner, bool)

        kept_response_tensors.append(pair[0])
        kept_response_tensors.append(pair[1])

        if is_a_winner:
            rewards.extend([1.0, -1.0])
        else:
            rewards.extend([-1.0, 1.0])

    if not rewards:
        raise RuntimeError("No valid critic judgements in this batch")

    query_tensors = [prompt_tensor] * len(kept_response_tensors)
    reward_tensors = [torch.tensor([r], dtype=torch.float32, device=device) for r in rewards]

    return query_tensors, kept_response_tensors, reward_tensors


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args, = parser.parse_args_into_dataclasses()

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
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    trainer = PPOTrainer(
        config=ppo_config,
        model=policy,
        tokenizer=tokenizer
    )

    budget = Budget(max_calls=script_args.critic_budget)

    for step in range(script_args.num_steps):
        try:
            queries, responses, rewards = rollout_batch(
                trainer=trainer,
                budget=budget,
                max_retries=script_args.critic_max_retries,
                max_work_tokens=script_args.max_work_tokens
            )
        except RuntimeError as e:
            print(f"Step {step} failed: {e}")
            break

        trainer.step(queries, responses, rewards)

        if step % script_args.logging_steps == 0:
            mean_reward = torch.tensor(rewards).mean().item()
            print(f"step {step} mean_reward: {mean_reward}")

    trainer.save_pretrained(script_args.output_dir)
