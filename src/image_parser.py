import sys
sys.dont_write_bytecode = True

import os
from dotenv import load_dotenv
import base64
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from langfuse import observe, Langfuse, LangfuseSpan
from opentelemetry.instrumentation.threading import ThreadingInstrumentor

ROOT_DIR = Path(__file__).resolve().parents[1]

sys.path.append(str(ROOT_DIR))

from src.utils import AI_API_CLIENT
from src.tracing import start_span

load_dotenv()

# Instrument threading for automatic context propagation
# This ensures OpenTelemetry context (including Langfuse trace context) is automatically
# propagated to threads, eliminating the need to manually pass trace_id and parent_span_id
ThreadingInstrumentor().instrument()

SYSTEM_PROMPT = (
    "You are a legal, text from image, parser. "
    "From the following image, extract the structured contract text preserving headings, sections, "
    "clauses, numbering, and hierarchy. Only return the text from image, no other text or explanation is allowed."
    )

def encode_image(path: str) -> str:
    """Encode an image file to base64 string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def parse_contract_image_with_fallback_model(
    image_path: str, 
    contract_id: str, 
    callbacks=None,
    fallback_model_name: str="google/gemma-3-4b-it:free"
) -> str:
    """
    Fallback function using google/gemma-3-4b-it:free model.
    This is used when the primary vision model fails.
    
    Returns:
        Extracted text from the image using the fallback model
    """
    image_b64 = encode_image(image_path)
    
    # Create a model instance with the fallback model
    fallback_model = ChatOpenAI(
        model=fallback_model_name,
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        temperature=0,
        name=f"fallback_model_image_parser_{contract_id}",
        callbacks=callbacks
    )
    
    # Gemini and others providers do not support system messages
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
    
    response = fallback_model.invoke(messages)
    parsed_text = response.content
    
    return parsed_text

def parse_contract_image(
    image_path: str, 
    contract_id: str,
    callbacks=None,
) -> str:
    """
    Function to parse a single contract image.
    This function runs in a separate thread, and the OpenTelemetry context
    (including Langfuse trace context) is automatically propagated via ThreadingInstrumentor.
    No need to manually pass trace_id or parent_span_id - the context is inherited automatically.
    
    Uses google/gemma-3-4b-it:free as fallback if the primary vision model fails.
    
    Returns:
        Parsed text from the image
    """
    try:
        # The callback handler automatically attaches to the current trace context
        # which is propagated from the parent thread via ThreadingInstrumentor
        image_b64 = encode_image(image_path)
        vision_model = os.getenv("IMAGE_MULTIMODAL_MODEL")

        provider = vision_model.split('/')[0] if vision_model else "unknown"

        # Create a model instance with the specific vision model for this call
        model = ChatOpenAI(
            model=vision_model,
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL"),
            temperature=0,
            name=f"model_call_image_parser_{contract_id}",
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

        # The model.invoke() call will automatically create observations in the current trace context
        # The callbacks parameter ensures LangChain integrates with Langfuse
        # Callbacks are passed directly to invoke() - they automatically attach to the current trace context
        response = model.invoke(messages)
        parsed_text = response.content
        
        if not parsed_text or not parsed_text.strip():
            # If primary model returns empty, try fallback model
            return parse_contract_image_with_fallback_model(image_path, contract_id, callbacks)
        
        return parsed_text
    
    except Exception as e:
        # If primary model fails, try fallback model (google/gemma-3-4b-it:free)
        try:
            return parse_contract_image_with_fallback_model(image_path, contract_id, callbacks)
        except Exception as fallback_error:
            # If fallback also fails, raise the original error
            raise Exception(f"Both primary model ({vision_model}) and fallback model (google/gemma-3-4b-it:free) failed. Primary error: {str(e)}, Fallback error: {str(fallback_error)}") from e

def parse_full_contract(
    images_folder: str, 
    contract_id: str,
    callbacks=None,
) -> str:
    """
    Parse all images in a folder using multithreading.
    The OpenTelemetry context (including Langfuse trace context) is automatically
    propagated to worker threads via ThreadingInstrumentor, so child observations
    will automatically be nested under the current trace context.
    """
    images = sorted(os.listdir(images_folder))  # Sort for consistent ordering
    
    # Filter out non-image files (optional, but good practice)
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'}
    images = [img for img in images if any(img.lower().endswith(ext) for ext in image_extensions)]
    
    # ThreadPoolExecutor will automatically propagate the OpenTelemetry context
    # to each worker thread thanks to ThreadingInstrumentor
    # Each parse_contract_image call will automatically create observations
    # in the current trace context without needing to pass trace_id or parent_span_id
    # If primary model fails, fallback model (google/gemma-3-4b-it:free) will be used automatically
    with ThreadPoolExecutor(max_workers=len(images)) as executor:
        text_list = list(executor.map(
            lambda image: parse_contract_image(
                image_path=os.path.join(images_folder, image), 
                contract_id=contract_id, 
                callbacks=callbacks
            ), 
            images)
            )
    text = ''.join(text_list)
    return text