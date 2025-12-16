import sys
sys.dont_write_bytecode = True
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.image_parser import parse_full_contract
from src.agents.contextualization_agent import contextualize_documents
from src.agents.extraction_agent import extract_changes
from src.tracing import start_trace, start_span, get_langfuse_callback_handler

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

def main():
    if len(sys.argv) != 4:
        print("Usage: python src/main.py data/contract_folder/original data/contract_folder/amendment contract_id")
        print(f"Received arguments:{len(sys.argv)} ,{sys.argv}")
        sys.exit(1)

    original_path = sys.argv[1]
    amendment_path = sys.argv[2]
    contract_id = sys.argv[3]

    # Create centralized trace for the entire flow
    with start_trace(
        "contract_comparison",
        {
            "original_path": original_path,
            "amendment_path": amendment_path,
            "contract_id": contract_id
        },
        as_type="trace"
    ) as main_trace:
        
        # Get Langfuse callback handler for LangChain integration
        # This will automatically capture all LLM calls within this trace
        langfuse_handler = get_langfuse_callback_handler()

        print(f"Starting contract comparison for contract: {contract_id}")
        print(f"Parsing original contract: {original_path}")
        print(f"Parsing amendment: {amendment_path}")

        # Step 1: Parse original contract images
        with start_span(
            "parse_original_contract",
            {
                "step": "image_parsing",
                "contract_type": "original",
                "path": original_path,
                "contract_id": contract_id
            }
        ) as span:
            original_text = parse_full_contract(original_path, contract_id, callbacks=[langfuse_handler])
            span.update(
                output={
                    "text_length": len(original_text),
                    "text_preview": original_text[:500] if len(original_text) > 500 else original_text
                }
            )
            span.end()

        # Step 2: Parse amendment images
        with start_span(
            "parse_amendment_contract",
            {
                "step": "image_parsing",
                "contract_type": "amendment",
                "path": amendment_path,
                "contract_id": contract_id
            }
        ) as span:
            amendment_text = parse_full_contract(amendment_path, contract_id, callbacks=[langfuse_handler])
            span.update(
                output={
                    "text_length": len(amendment_text),
                    "text_preview": amendment_text[:500] if len(amendment_text) > 500 else amendment_text
                }
            )
            span.end()

        print(f"Length of Original text: {len(original_text)}")
        print(f"Length of Amendment text: {len(amendment_text)}")

        # Step 3: Contextualize documents
        with start_span(
            "contextualize_documents",
            {
                "step": "contextualization",
                "contract_id": contract_id,
                "original_text_length": len(original_text),
                "amendment_text_length": len(amendment_text)
            }
        ) as span:
            context = contextualize_documents(
                original_text=original_text,
                amendment_text=amendment_text,
                contract_id=contract_id,
                callbacks=[langfuse_handler]
            )
            span.update(output=_serialize_output(context))
            span.end()

        # Step 4: Extract changes
        with start_span(
            "extract_changes",
            {
                "step": "extraction",
                "contract_id": contract_id,
                "contextualized_original_length": len(context.original_contract_text),
                "contextualized_amendment_length": len(context.amendment_text)
            }
        ) as span:
            result = extract_changes(
                original_text=context.original_contract_text,
                amendment_text=context.amendment_text,
                contract_id=contract_id,
                callbacks=[langfuse_handler]
            )
            span.update(output=_serialize_output(result))
            span.end()

        print(result.model_dump())
        
        # Set final output on the main trace
        main_trace.update(output=_serialize_output(result))
        main_trace.end()

if __name__ == "__main__":
    main()
