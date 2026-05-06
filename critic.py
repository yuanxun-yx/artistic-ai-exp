import asyncio
import random

from openai import APIConnectionError, AsyncOpenAI

client = AsyncOpenAI()


async def get_response(
    model: str,
    dev_input: str,
    user_input: str,
    max_retries: int,
) -> str:
    for i in range(max_retries):
        try:
            response = await client.responses.create(
                model=model,
                input=[
                    {"role": "developer", "content": dev_input},
                    {"role": "user", "content": user_input},
                ],
            )
        # openai burst rate limit
        except APIConnectionError:
            await asyncio.sleep(2**i * (1 + random.random() * 0.1))
            continue
        return response.output_text
    raise RuntimeError(
        f"critic model failed to return feedback after {max_retries} retries"
    )


async def get_response_batch(
    model: str, dev_input: str, user_input: list[str], max_retries: int
):
    coro = [
        get_response(
            model=model, dev_input=dev_input, user_input=i, max_retries=max_retries
        )
        for i in user_input
    ]
    return await asyncio.gather(*coro, return_exceptions=True)
