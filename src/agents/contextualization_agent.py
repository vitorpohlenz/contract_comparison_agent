import sys
sys.dont_write_bytecode = True

import os
from dotenv import load_dotenv

from openai import OpenAI

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.tracing import start_trace
from src.utils import AI_API_CLIENT, prompt_template
from src.models import ContractChangeSummary

SYSTEM_PROMPT = (
    "You are a senior contract comparison analyst. "
    "Compare two contracts and identify structure, section alignment, "
    "and which sections correspond to each other."
    "\n Return a JSON object with the following fields:"
    "\n - added_sections: list of added sections"
    "\n - removed_sections: list of removed sections"
    "\n - modified_sections: list of modified sections"
    "\n - summary_of_the_change: summary of the change with format Section X: -change_1 \n change_2, ..."
)


def contextualize_documents(
        original_text: str,
        amendment_text: str,
        client: OpenAI=AI_API_CLIENT,
        system_prompt: str=SYSTEM_PROMPT
    ) -> str:
    with start_trace(
        "contextualization_agent",
        {"agent": "contextualization"}
    ) as trace:
        contextualization_model = os.getenv("CONTEXTUALIZATION_MODEL")

        response = client.chat.completions.parse(
            model=contextualization_model,
            messages=prompt_template(
                system_prompt=system_prompt,
                user_prompt=(f"\n\nORIGINAL CONTRACT:\n {original_text} \n\nAMENDMENT:\n {amendment_text}"),
            full_model_name=contextualization_model
            ),
            temperature=0,
            response_format=ContractChangeSummary
        )

        output = response.choices[0].message.parsed
    
    return output
