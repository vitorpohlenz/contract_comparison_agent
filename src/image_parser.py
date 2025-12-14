import sys
sys.dont_write_bytecode = True

import os
from dotenv import load_dotenv
import base64
from openai import OpenAI

from concurrent.futures import ThreadPoolExecutor

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

sys.path.append(str(ROOT_DIR))

from src.tracing import start_trace
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

def parse_contract_image(image_path: str, contract_id: str, client: OpenAI=AI_API_CLIENT) -> str:
    with start_trace(
        "image_parsing",
        {"image_path": image_path, "contract_id": contract_id}
    ) as trace:
        image_b64 = encode_image(image_path)
        vision_model = os.getenv("IMAGE_MULTIMODAL_MODEL")

        provider = vision_model.split('/')[0]

        if provider == "openai":
            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
                    ]
                }
            ]
        else:
            # Gemini and others providers does not support system messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": SYSTEM_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}"
                            }
                        }
                    ]
                }
            ]

        response = client.chat.completions.create(
            model=vision_model,
            messages=messages,
            temperature=0
        )
        parsed_text = response.choices[0].message.content
    
    return parsed_text


def parse_full_contract(images_folder: str, contract_id: str, client: OpenAI=AI_API_CLIENT) -> str:
    with start_trace(
        "parse_full_contract",
        {"images_folder": images_folder, "contract_id": contract_id}
    ) as trace:
        images = os.listdir(images_folder)
        with ThreadPoolExecutor(max_workers=len(images)) as executor:
            text_list = list(
                executor.map(
                    lambda f: parse_contract_image(image_path=images_folder+f, contract_id=contract_id, client=client),
                    images
                )
            )
    
    return ''.join(text_list)