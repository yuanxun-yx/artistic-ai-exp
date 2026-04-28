from time import sleep

from openai import APIConnectionError, OpenAI

client = OpenAI()


def get_response(
    model: str,
    dev_input: str,
    user_input: str,
    max_retries: int,
) -> str:
    for i in range(max_retries):
        try:
            response = client.responses.create(
                model=model,
                input=[
                    {"role": "developer", "content": dev_input},
                    {"role": "user", "content": user_input},
                ],
            )
        # openai burst rate limit
        except APIConnectionError:
            sleep(2**i)
            continue
        return response.output_text
    raise RuntimeError(
        f"critic model failed to return feedback after {max_retries} retries"
    )
