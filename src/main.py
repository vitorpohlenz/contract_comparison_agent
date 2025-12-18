import sys
sys.dont_write_bytecode = True
import json
from pathlib import Path
from langfuse import get_client
import uuid

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.image_parser import parse_full_contract
from src.agents.contextualization_agent import contextualize_documents
from src.agents.extraction_agent import extract_changes
from src.tracing import start_trace, start_span, CallbackHandler
from src.models import ContextualizedContract
from src.models import ContractChangeSummary
from src.utils import _serialize_output

def main():
    if len(sys.argv) != 4:
        print("Usage: python src/main.py data/contract_folder/original data/contract_folder/amendment contract_id")
        print(f"Received arguments:{len(sys.argv)} ,{sys.argv}")
        sys.exit(1)

    original_path = sys.argv[1]
    amendment_path = sys.argv[2]
    contract_id = sys.argv[3]

    langfuse_client = get_client()
    
    # Use contract_id as session_id for all spans
    session_id = str(uuid.uuid4())

    # Create centralized trace for the entire flow using langfuse.trace()
    with start_span(
        langfuse_client=langfuse_client,
        name="contract_comparison",
        input={
            "original_path": original_path,
            "amendment_path": amendment_path,
            "contract_id": contract_id
        },
        metadata={"session_id": session_id, "contract_id": contract_id}
    ) as main_trace:
        
        # Get Langfuse callback handler for LangChain integration
        # This will automatically capture all LLM calls within this trace
        langfuse_handler = CallbackHandler()

        print(f"Starting contract comparison for contract: {contract_id}")
        print(f"Parsing original contract: {original_path}")
        print(f"Parsing amendment: {amendment_path}")

        # Step 1: Parse original contract images
        with start_span(
            langfuse_client=langfuse_client,
            name="parse_original_contract",
            input={
                "step": "image_parsing",
                "contract_type": "original",
                "path": original_path,
                "contract_id": contract_id
            },
            metadata={"session_id": session_id, "contract_id": contract_id}
        ) as span_parse_contract:
            # With ThreadingInstrumentor, the trace context is automatically propagated
            # to worker threads, so no need to manually pass trace_id or parent_span_id
            original_text = parse_full_contract(
                original_path, 
                contract_id, 
                callbacks=[langfuse_handler],
            )
            span_parse_contract.update(
                output={
                    "text_length": len(original_text),
                    "text_preview": original_text[:500] if len(original_text) > 500 else original_text
                }
            )

        # Step 2: Parse amendment images
        with start_span(
            langfuse_client=langfuse_client,
            name="parse_amendment_contract",
            input={
            
                "step": "image_parsing",
                "contract_type": "amendment",
                "path": amendment_path,
                "contract_id": contract_id
            },
            metadata={"session_id": session_id, "contract_id": contract_id}
        ) as span_parse_amendment:
            # With ThreadingInstrumentor, the trace context is automatically propagated
            # to worker threads, so no need to manually pass trace_id or parent_span_id
            amendment_text = parse_full_contract(
                amendment_path, 
                contract_id, 
                callbacks=[langfuse_handler],
            )
            span_parse_amendment.update(
                output={
                    "text_length": len(amendment_text),
                    "text_preview": amendment_text[:500] if len(amendment_text) > 500 else amendment_text
                }
            )

        print(f"Length of Original text: {len(original_text)}")
        print(f"Length of Amendment text: {len(amendment_text)}")
        # Step 3: Contextualize documents using LangChain tool
        with start_span(
            langfuse_client=langfuse_client,
            name="contextualize_documents",
            input={
                "step": "contextualization",
                "contract_id": contract_id,
                "original_text_length": len(original_text),
                "amendment_text_length": len(amendment_text)
            },
            metadata={"session_id": session_id, "contract_id": contract_id}
        ) as span_contextualize_documents:
            # Invoke the LangChain tool with callbacks in config
            context_dict = contextualize_documents.invoke(
                {
                    "original_text": original_text,
                    "amendment_text": amendment_text,
                    "contract_id": contract_id
                },
                config={"callbacks": [langfuse_handler]}
            )
            # Convert dict back to model for compatibility
            context = ContextualizedContract(**context_dict)
            span_contextualize_documents.update(output=context.model_dump())

        # Step 4: Extract changes using LangChain tool
        with start_span(
            langfuse_client=langfuse_client,
            name="extract_changes",
            input={
                "step": "extraction",
                "contract_id": contract_id,
                "contextualized_original_length": len(context.original_contract_text),
                "contextualized_amendment_length": len(context.amendment_text)
            },
            metadata={"session_id": session_id, "contract_id": contract_id}
        ) as span_extract_changes:
            # Invoke the LangChain tool with callbacks in config
            result_dict = extract_changes.invoke(
                {
                    "original_text": context.original_contract_text,
                    "amendment_text": context.amendment_text,
                    "contract_id": contract_id
                },
                config={"callbacks": [langfuse_handler]}
            )
            # Convert dict back to model for compatibility
            result = ContractChangeSummary(**result_dict)
            span_extract_changes.update(output=result.model_dump())

        print(result.model_dump())
        
        # Set final output on the main trace
        main_trace.update(output=result.model_dump())

if __name__ == "__main__":
    main()
