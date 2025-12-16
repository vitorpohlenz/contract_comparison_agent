import sys
sys.dont_write_bytecode = True

import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.tracing import start_trace
from src.utils import AI_API_CLIENT, prompt_template
from src.models import ContextualizedContract

SYSTEM_PROMPT = (
    "You are a senior legal contextualization agent. "
    "Contextualize the ORIGINAL CONTRACT and the AMENDMENT and identify structure, section alignment, "
    "and which sections correspond to each other."
    "\n Return a JSON object with the following fields, containing just the text impacted by the amendment and the amendment text:"
    "\n - original_contract_text: text of the original contract just the text impacted by the amendment"
    "\n - amendment_text: text of the amendment"
)


def contextualize_documents(
        original_text: str,
        amendment_text: str,
        contract_id: str,
        client: ChatOpenAI=AI_API_CLIENT,
        system_prompt: str=SYSTEM_PROMPT,
    ) -> ContextualizedContract:
    with start_trace(
        "contextualization_agent",
        {"agent": "contextualization", "contract_id": contract_id}
    ) as trace:
        contextualization_model = os.getenv("LLM_MODEL")

        # Create a model instance with the specific model for this call
        model = ChatOpenAI(
            model=contextualization_model,
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL"),
            temperature=0
        )

        response = model.with_structured_output(ContextualizedContract).invoke(
            prompt_template(
                system_prompt=system_prompt,
                user_prompt=(f"\n\nORIGINAL CONTRACT:\n {original_text} \n\nAMENDMENT:\n {amendment_text}"),
                full_model_name=contextualization_model
            )
        )

        output = response
    
    return output
