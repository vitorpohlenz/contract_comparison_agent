import sys
sys.dont_write_bytecode = True
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.image_parser import parse_full_contract
from src.agents.contextualization_agent import contextualize_documents
from src.agents.extraction_agent import extract_changes
from src.tracing import start_trace

def main():
    if len(sys.argv) != 4:
        print("Usage: python src/main.py data/contract_folder/original data/contract_folder/amendment contract_id")
        print(f"Received arguments:{len(sys.argv)} ,{sys.argv}")
        sys.exit(1)

    original_path = sys.argv[1]
    amendment_path = sys.argv[2]
    contract_id = sys.argv[3]

    with start_trace(
        "main",
        {"original_path": original_path, "amendment_path": amendment_path, "contract_id": contract_id}
    ) as trace:

        print(f"Starting contract comparison for contract: {contract_id}")
        print(f"Parsing original contract: {original_path}")
        print(f"Parsing amendment: {amendment_path}")

        original_text = parse_full_contract(original_path, contract_id)
        amendment_text = parse_full_contract(amendment_path, contract_id)

        print(f"Length of Original text: {len(original_text)}")
        print(f"Length of Amendment text: {len(amendment_text)}")

        context = contextualize_documents(
            original_text=original_text,
            amendment_text=amendment_text,
            contract_id=contract_id
        )
        result = extract_changes(
            original_text=context.original_contract_text,
            amendment_text=context.amendment_text,
            contract_id=contract_id
        )

        print(result.model_dump())

if __name__ == "__main__":
    main()
