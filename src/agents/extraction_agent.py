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
from src.models import ContractChangeSummary

SYSTEM_PROMPT = (
    "You are a senior contract comparison analyst. "
    "Compare the ORIGINAL CONTRACT CONTENT and the AMENDMENT CONTENT and identify the topics touched, the sections changed and the summary of the change."
    "\n Return a JSON object with the following fields:"
    "\n - topics_touched: list of topics touched in the amendment"
    "\n - sections_changed: list of sections changed in the amendment"
    "\n - summary_of_the_change: summary of the change in the amendment with format Section X: -change_1 \n -change_2, ..."
)


@tool
def extract_changes(
        original_text: str,
        amendment_text: str,
        contract_id: str
    ) -> dict:
    """
    Extract and summarize changes between the original contract and amendment.
    
    This tool compares the original contract content and amendment content to identify:
    - Topics touched in the amendment
    - Sections changed in the amendment
    - A detailed summary of the changes
    
    Args:
        original_text: The contextualized original contract text (only impacted sections)
        amendment_text: The amendment text
        contract_id: Unique identifier for the contract being processed
    
    Returns:
        A dictionary with:
        - topics_touched: List of legal or business topics affected
        - sections_changed: List of contract sections that were changed
        - summary_of_the_change: Summary of changes with format "Section X: -change_1 \n -change_2, ..."
    """
    extraction_model = os.getenv("LLM_MODEL")

    # Create a model instance with the specific model for this call
    # Callbacks are handled by LangChain's callback system through the tool invoke config
    model = ChatOpenAI(
        model=extraction_model,
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        temperature=0,
        name=f"extraction_agent_{contract_id}"
    )

    response = model.with_structured_output(ContractChangeSummary).invoke(
        prompt_template(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=(f"\n\nORIGINAL CONTRACT CONTENT:\n {original_text} \n\nAMENDMENT CONTENT:\n {amendment_text}"),
            full_model_name=extraction_model
        )
    )

    return response.model_dump()
