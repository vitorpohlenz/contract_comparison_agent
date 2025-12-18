import sys
sys.dont_write_bytecode = True

import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

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


@tool
def contextualize_documents(
        original_text: str,
        amendment_text: str,
        contract_id: str
    ) -> dict:
    """
    Contextualize the original contract and amendment documents.
    
    This tool analyzes the original contract and amendment to identify structure,
    section alignment, and which sections correspond to each other. It returns
    only the text impacted by the amendment from the original contract and the
    full amendment text.
    
    Args:
        original_text: The full text of the original contract
        amendment_text: The full text of the amendment
        contract_id: Unique identifier for the contract being processed
    
    Returns:
        A dictionary with:
        - original_contract_text: Text of the original contract (only the text impacted by the amendment)
        - amendment_text: Text of the amendment
    """
    contextualization_model = os.getenv("LLM_MODEL")

    # Create a model instance with the specific model for this call
    # Callbacks are handled by LangChain's callback system through the tool invoke config
    model = ChatOpenAI(
        model=contextualization_model,
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        temperature=0,
        name=f"contextualization_agent_{contract_id}"
    )

    response = model.with_structured_output(ContextualizedContract).invoke(
        prompt_template(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=(f"\n\nORIGINAL CONTRACT:\n {original_text} \n\nAMENDMENT:\n {amendment_text}"),
            full_model_name=contextualization_model
        )
    )

    return response.model_dump()
