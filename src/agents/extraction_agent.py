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
from src.models import ContractChangeSummary

SYSTEM_PROMPT = (
    "You are a senior contract comparison analyst. "
    "Compare the ORIGINAL CONTRACT CONTENT and the AMENDMENT CONTENT and identify the topics touched, the sections changed and the summary of the change."
    "\n Return a JSON object with the following fields:"
    "\n - topics_touched: list of topics touched in the amendment"
    "\n - sections_changed: list of sections changed in the amendment"
    "\n - summary_of_the_change: summary of the change in the amendment with format Section X: -change_1 \n -change_2, ..."
)


def extract_changes(
        original_text: str,
        amendment_text: str,
        contract_id: str,
        client: ChatOpenAI=AI_API_CLIENT,
        system_prompt: str=SYSTEM_PROMPT
    ) -> ContractChangeSummary:
    with start_trace(
        "extraction_agent",
        {"agent": "extraction", "contract_id": contract_id}
    ) as trace:
        extraction_model = os.getenv("LLM_MODEL")

        # Create a model instance with the specific model for this call
        model = ChatOpenAI(
            model=extraction_model,
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL"),
            temperature=0
        )

        response = model.with_structured_output(ContractChangeSummary).invoke(
            prompt_template(
                system_prompt=system_prompt,
                user_prompt=(f"\n\nORIGINAL CONTRACT CONTENT:\n {original_text} \n\nAMENDMENT CONTENT:\n {amendment_text}"),
                full_model_name=extraction_model
            )
        )

        output = response
    
    return output
