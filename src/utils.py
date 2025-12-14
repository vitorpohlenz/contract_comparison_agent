import sys
sys.dont_write_bytecode = True

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

AI_API_CLIENT = OpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL")
)

def prompt_template(system_prompt: str, user_prompt: str, full_model_name: str) -> list[dict]:
    provider = full_model_name.split('/')[0]

    if provider == "openai":
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": system_prompt + "\n\n" + user_prompt
            }
        ]
    return messages