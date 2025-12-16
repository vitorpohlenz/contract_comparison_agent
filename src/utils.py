import sys
sys.dont_write_bytecode = True

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

AI_API_CLIENT = ChatOpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL")
)

def prompt_template(system_prompt: str, user_prompt: str, full_model_name: str):
    """
    Create LangChain messages from system and user prompts.
    Returns a list of LangChain message objects.
    """
    provider = full_model_name.split('/')[0] if full_model_name else "unknown"

    if provider == "openai":
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
    else:
        # Gemini and other providers that don't support system messages
        messages = [
            HumanMessage(content=system_prompt + "\n\n" + user_prompt)
        ]
    return messages