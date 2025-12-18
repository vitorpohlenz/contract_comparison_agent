# Contract Comparison Agent

## Project Description

The Contract Comparison Agent is an autonomous AI system designed to analyze and extract changes from legal contracts. The system processes scanned contract images (both original and amended versions) using vision-capable large language models, intelligently identifies modifications, and returns structured, validated outputs. This agent addresses the critical need for automated contract analysis, enabling legal teams, compliance officers, and business stakeholders to quickly understand contractual changes without manual document review. The system leverages a sophisticated two-agent architecture that first contextualizes documents to identify relevant sections, then extracts and summarizes changes in a structured format. Built with LangChain for LLM orchestration, Pydantic for data validation, and Langfuse for comprehensive observability, the agent provides production-ready contract comparison capabilities with full traceability and monitoring.

## Architecture and Agent Workflow

### Repository Structure

```text
contract_comparison_agent/
├── src/
│   ├── agents/
│   │   ├── contextualization_agent.py    # Agent 1: Document alignment and filtering
│   │   └── extraction_agent.py            # Agent 2: Change detection and summarization
│   ├── image_parser.py                     # Vision model integration for image-to-text
│   ├── main.py                             # Entry point and orchestration
│   ├── models.py                           # Pydantic data models
│   ├── tracing.py                          # Langfuse observability integration
│   └── utils.py                            # Shared utilities and prompt templates
├── data/
│   └── test_contracts/                     # Test contract images
│       ├── case_1/
│       │   ├── original/                   # Original contract images
│       │   └── amendment/                  # Amendment images
│       └── case_2/
│           ├── original/
│           └── amendment/
├── tests/
│   ├── test_agents.py                      # Agent handoff and integration tests
│   └── test_validation.py                  # Pydantic validation and E2E tests
├── README.md
└── .env                                    # Environment configuration (not in repo)
```

### Workflow Diagram

```text
┌───────────────────────────────────────────────┐
│          Contract Comparison Pipeline         │
└───────────────────────────────────────────────┘

INPUT
  │
  ├─ Original Images ───┐
  │                     ├───┐
  └─ Amendment Images ──┘   │
                            │
        ┌───────────────────┴──────┐
        ▼                          ▼
┌──────────────┐         ┌──────────────┐
│   STEP 1     │         │   STEP 2     │
│   Original   │         │  Amendment   │
│   Parsing    │         │   Parsing    │
├──────────────┤         ├──────────────┤
│ Vision Model │         │ Vision Model │
│ ThreadPool   │         │ ThreadPool   │
└──────────────┘         └──────────────┘
        │                        │
   original_text             amendment_text
        │                        │
        └───────────┬────────────┘
                    │
                    ▼
        ┌──────────────────────┐
        │        STEP 3        │
        │  Contextualization   │
        │        Agent         │
        ├──────────────────────┤
        │ • Structure analysis │
        │ • Section alignment  │
        │ • Content filtering  │
        └──────────────────────┘
                    │
                    │ ContextualizedContract
                    │
                    ▼
        ┌───────────────────────┐
        │          STEP 4       │
        │        Extraction     │
        │          Agent        │
        ├───────────────────────┤
        │ • Change comparison   │
        │ • Topic identification│
        │ • Summary generation  │
        └───────────────────────┘
                    │
                    │ ContractChangeSummary
                    │
                    ▼
            ┌──────────┐
            │  OUTPUT  │
            │   JSON   │
            └──────────┘

┌───────────────────────────────────────────────┐
│  Langfuse Tracing (Throughout)                │
│                                               │
│  Trace: contract_comparison                   │
│    ├─ Span: parse_original_contract (Step 1)  │
│    ├─ Span: parse_amendment_contract (Step 2) │
│    ├─ Span: contextualize_documents (Step 3)  │
│    └─ Span: extract_changes (Step 4)          │
│  • All LLM calls automatically captured       │
│  • Input/output data logged for each step     │
└───────────────────────────────────────────────┘
```

### Workflow Description

The system employs a sequential four-step workflow designed to handle the complexity of legal document comparison. **Step 1** processes the original contract images, where vision-capable models extract structured text from scanned contract pages. Multiple images are processed in parallel using thread pools for efficiency, with automatic fallback mechanisms to ensure reliability. **Step 2** performs the same image parsing process for the amendment contract images. Both steps run independently and can execute in parallel, extracting and concatenating text from all images in their respective folders. Once both text extractions are complete, **Step 3** (Contextualization Agent) receives the full extracted text from both original and amended contracts. Its specialized role is to identify structural alignment, determine which sections correspond to each other, and filter the content to only the portions impacted by the amendment. This contextualization step is crucial because contracts can be lengthy, and focusing the comparison on relevant sections improves accuracy and reduces token costs. The contextualized output is then passed to **Step 4** (Extraction Agent), which performs the actual change analysis. This agent compares the filtered original contract text against the amendment text, identifying topics touched, sections changed, and generating a structured summary of modifications. The sequential handoff from Steps 1 and 2 to Step 3, and then to Step 4, ensures that the extraction agent works with precisely the relevant context, enabling more accurate and focused change detection. All operations are instrumented with Langfuse tracing, creating a complete observability layer that tracks each step of the process, from image parsing through final output generation.

## Setup Instructions

### Installation

- System tested with Python **3.9.7**

Clone the repository:
```bash
git clone <repository-url>
cd contract_comparison_agent
```

Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows
```

Install Python dependencies:
```bash
pip install -r requirements.txt
```

### API Keys Configuration

Create a `.env` file in the project root folder, following the `.env.example` file

**Note**: The system supports multiple LLM providers. Ensure your API key has access to both text generation models (for agents) and vision models (for image parsing).

### Test Images

The project includes test contract images in the `data/test_contracts/` directory:

- **Case 1**: Located in `data/test_contracts/case_1/`
  - Original contract: `data/test_contracts/case_1/original/` (contains multiple PNG images)
  - Amendment: `data/test_contracts/case_1/amendment/` (contains amendment image)
  
- **Case 2**: Located in `data/test_contracts/case_2/`
  - Original contract: `data/test_contracts/case_2/original/` (contains multiple PNG images)
  - Amendment: `data/test_contracts/case_2/amendment/` (contains amendment images)

You can use these test cases to verify the installation and configuration.

## Usage

Run the contract comparison agent with the following command:

```bash
python src/main.py <original_contract_folder> <amendment_folder> <contract_id>
```

### Example Command

```bash
python src/main.py data/test_contracts/case_1/original data/test_contracts/case_1/amendment case_1_test
```

**Parameters:**
- `original_contract_folder`: Path to the folder containing original contract images (PNG, JPG, etc.)
- `amendment_folder`: Path to the folder containing amendment images
- `contract_id`: Unique identifier for this contract comparison (used for tracing and logging)

The system will:
1. Parse all images in both folders
2. Extract and concatenate text from original contract images
3. Extract and concatenate text from amendment images
4. Contextualize the documents to identify relevant sections
5. Extract and summarize changes
6. Output the structured result to console and Langfuse

## Expected Output Format

The agent returns a structured JSON object conforming to the `ContractChangeSummary` model:

```json
{
  "topics_touched": [
    "Liability allocation",
    "Risk management",
    "Damages limitation"
  ],
  "sections_changed": [
    "Section 14 – Limitation of Liability"
  ],
  "summary_of_the_change": "Section 14: -Replaced entire liability clause\n-Introduced explicit liability cap at fees paid in prior 12 months\n-Added carve-outs for confidentiality breaches and indemnification\n-Preserved exclusion of indirect and consequential damages"
}
```

### Output Fields

- **topics_touched**: List of legal or business topics affected by the amendment (e.g., "Termination", "Liability", "Payment Terms")
- **sections_changed**: List of specific contract sections that were modified (e.g., "Section 5 – Termination")
- **summary_of_the_change**: Detailed summary of changes in the format "Section X: -change_1\n-change_2, ..." where each change is prefixed with a dash

The output is validated using Pydantic models to ensure data quality and structure consistency.

## Technical Decisions

The architecture employs a two-agent design to achieve separation of concerns and improved accuracy. The Contextualization Agent handles the complex task of document alignment and relevance filtering, which requires understanding contract structure and identifying correspondences between original and amended sections. This specialized focus allows the agent to excel at its specific task rather than attempting to do both contextualization and extraction simultaneously. The Extraction Agent then operates on the filtered, contextualized content, enabling more precise change detection without the noise of irrelevant contract sections. This division of labor reduces token usage, improves processing speed, and enhances the accuracy of change detection. The system uses structured outputs with Pydantic models to ensure type safety and validation, preventing malformed results from propagating through the pipeline. Vision models are configured with automatic fallback mechanisms (defaulting to `google/gemma-3-4b-it:free` when primary models fail) to ensure reliability in production environments. Parallel image processing using thread pools significantly reduces processing time for multi-page contracts. Langfuse integration provides comprehensive observability, enabling debugging, performance monitoring, and compliance tracking. The choice of temperature=0 for all LLM calls ensures deterministic, reproducible outputs critical for legal document analysis.

## Langfuse Tracing Guide

Langfuse provides comprehensive observability for the entire contract comparison workflow. After running a comparison, navigate to your Langfuse dashboard (configured via `LANGFUSE_HOST`) to view detailed traces. Each contract comparison creates a top-level trace named "contract_comparison" with child spans for each major step: image parsing (both original and amendment), contextualization, and change extraction. The dashboard displays input/output data for each span, allowing you to inspect the text extracted from images, the contextualized content passed between agents, and the final structured output. You can filter traces by contract_id (stored as session_id in metadata) to track specific contract analyses over time. The dashboard also shows LLM call details, including tokens used, latency, and model responses, enabling cost optimization and performance monitoring. This observability layer is essential for debugging issues, understanding agent behavior, and ensuring compliance with legal document processing requirements.
