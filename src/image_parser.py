import sys
sys.dont_write_bytecode = True

import os
from dotenv import load_dotenv
import base64
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from langfuse import observe

ROOT_DIR = Path(__file__).resolve().parents[1]

sys.path.append(str(ROOT_DIR))

from src.utils import AI_API_CLIENT

load_dotenv()

SYSTEM_PROMPT = (
    "You are a legal, text from image, parser. "
    "From the following image, extract the structured contract text preserving headings, sections, "
    "clauses, numbering, and hierarchy. Only return the text from image, no other text or explanation is allowed."
    )

def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

@observe()
def parse_contract_image(
    image_path: str, 
    contract_id: str, 
    client: ChatOpenAI=AI_API_CLIENT, 
    callbacks=None,
    langfuse_trace_id: str = None,
    langfuse_parent_observation_id: str = None
) -> str:
    image_b64 = encode_image(image_path)
    vision_model = os.getenv("IMAGE_MULTIMODAL_MODEL")

    provider = vision_model.split('/')[0] if vision_model else "unknown"

    # Create a model instance with the specific vision model for this call
    model = ChatOpenAI(
        model=vision_model,
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        temperature=0,
        name=f"image_parser_{contract_id}",
        callbacks=callbacks
    )

    if provider == "openai":
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=[
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
                ]
            )
        ]
    else:
        # Gemini and others providers does not support system messages
        messages = [
            HumanMessage(
                content=[
                    {"type": "text", "text": SYSTEM_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}"
                        }
                    }
                ]
            )
        ]

    response = model.invoke(messages)
    parsed_text = response.content
    
    return parsed_text

@observe()
def parse_full_contract(
    images_folder: str, 
    contract_id: str, 
    client: ChatOpenAI=AI_API_CLIENT, 
    callbacks=None,
    langfuse_trace_id: str = None,
    langfuse_parent_observation_id: str = None
) -> str:
    from langfuse import get_client
    
    images = sorted(os.listdir(images_folder))  # Sort for consistent ordering
    
    # Get trace context if not provided (for threading support)
    if langfuse_trace_id is None:
        langfuse_client = get_client()
        langfuse_trace_id = langfuse_client.get_current_trace_id()
        langfuse_parent_observation_id = langfuse_client.get_current_observation_id()
    
    # Create a wrapper function that passes trace context to each thread
    def parse_with_context(image_filename):
        return parse_contract_image(
            image_path=os.path.join(images_folder, image_filename),
            contract_id=contract_id,
            client=client,
            callbacks=callbacks,
            langfuse_trace_id=langfuse_trace_id,
            langfuse_parent_observation_id=langfuse_parent_observation_id
        )
    
    with ThreadPoolExecutor(max_workers=len(images)) as executor:
        text_list = list(executor.map(parse_with_context, images))
    
    return ''.join(text_list)