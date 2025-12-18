import sys
sys.dont_write_bytecode = True

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from typing import Union

load_dotenv()

AI_API_CLIENT = ChatOpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL")
)

def prompt_template(system_prompt: str, user_prompt: str, full_model_name: str) -> list[Union[SystemMessage, HumanMessage]]:
    """
    Create LangChain messages from system and user prompts.
    
    Args:
        system_prompt: The system prompt for the model.
        user_prompt: The user prompt for the model.
        full_model_name: The full model name.
    Returns:
        A list of Message objects.
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

def _serialize_output(output):
    """Serialize output data to be JSON-serializable for Langfuse."""
    if hasattr(output, 'model_dump'):
        return output.model_dump()
    elif hasattr(output, 'dict'):
        return output.dict()
    elif isinstance(output, (str, int, float, bool, type(None))):
        return output
    elif isinstance(output, (list, tuple)):
        return [_serialize_output(item) for item in output]
    elif isinstance(output, dict):
        return {k: _serialize_output(v) for k, v in output.items()}
    else:
        return str(output)