import sys
sys.dont_write_bytecode = True

from pathlib import Path
import pytest
from unittest.mock import Mock, patch, MagicMock
import os

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.models import ContractChangeSummary, ContextualizedContract
from src.image_parser import parse_full_contract
from src.agents.contextualization_agent import contextualize_documents
from src.agents.extraction_agent import extract_changes


# ============================================================================
# End-to-End Integration Test
# ============================================================================

class TestEndToEndIntegration:
    """End-to-end integration test for the complete contract comparison pipeline."""
    
    @patch('src.agents.extraction_agent.ChatOpenAI')
    @patch('src.agents.contextualization_agent.ChatOpenAI')
    @patch('src.image_parser.ChatOpenAI')
    @patch('src.image_parser.os.getenv')
    @patch('src.image_parser.encode_image')
    @patch('src.image_parser.os.listdir')
    def test_full_pipeline_integration(
        self,
        mock_listdir,
        mock_encode_image,
        mock_getenv,
        mock_image_chat_model,
        mock_contextualization_model,
        mock_extraction_model
    ):
        """
        Test the complete end-to-end flow:
        1. Parse original contract images
        2. Parse amendment images
        3. Contextualize documents (Agent 1)
        4. Extract changes (Agent 2)
        
        Verifies data flows correctly through each step and final output is valid.
        """
        contract_id = "test_contract_e2e_001"
        
        # ====================================================================
        # Setup: Mock environment variables
        # ====================================================================
        mock_getenv.side_effect = lambda key: {
            "IMAGE_MULTIMODAL_MODEL": "openai/gpt-4-vision",
            "LLM_MODEL": "openai/gpt-4",
            "LLM_API_KEY": "test_key",
            "LLM_BASE_URL": "test_url"
        }.get(key)
        
        # ====================================================================
        # Step 1: Mock Image Parsing - Original Contract
        # ====================================================================
        original_folder = "data/test_contracts/case_1/original"
        mock_listdir.return_value = [
            "example_software_development_agreement-1.png",
            "example_software_development_agreement-2.png"
        ]
        
        mock_encode_image.return_value = "base64_encoded_image"
        
        # Mock image parsing responses for original contract
        original_page_1_text = "SOFTWARE DEVELOPMENT AGREEMENT\n\nSection 5 - Termination\nThis agreement may be terminated by either party with 30 days written notice."
        original_page_2_text = "Section 6 - Liability\nEach party shall be liable for damages arising from breach of this agreement."
        
        mock_image_response_1 = Mock()
        mock_image_response_1.content = original_page_1_text
        mock_image_response_2 = Mock()
        mock_image_response_2.content = original_page_2_text
        
        # Mock ChatOpenAI for image parsing (will be called multiple times)
        mock_image_model_instance = Mock()
        mock_image_model_instance.invoke.side_effect = [
            mock_image_response_1,
            mock_image_response_2
        ]
        mock_image_chat_model.return_value = mock_image_model_instance
        
        # Execute: Parse original contract
        original_text = parse_full_contract(
            images_folder=original_folder,
            contract_id=contract_id,
            callbacks=None
        )
        
        # Verify: Original text is concatenated correctly
        assert isinstance(original_text, str)
        assert len(original_text) > 0
        assert "Section 5 - Termination" in original_text
        assert "30 days written notice" in original_text
        assert "Section 6 - Liability" in original_text
        
        # ====================================================================
        # Step 2: Mock Image Parsing - Amendment
        # ====================================================================
        amendment_folder = "data/test_contracts/case_1/amendment"
        mock_listdir.return_value = ["amendment_liability_page_1.png"]
        
        # Mock amendment parsing response
        amendment_text_content = "AMENDMENT TO SOFTWARE DEVELOPMENT AGREEMENT\n\nSection 5 - Termination\nThis agreement may be terminated by either party with 60 days written notice."
        
        mock_amendment_response = Mock()
        mock_amendment_response.content = amendment_text_content
        
        # Reset mock for amendment parsing
        mock_image_model_instance.invoke.side_effect = [mock_amendment_response]
        
        # Execute: Parse amendment
        amendment_text = parse_full_contract(
            images_folder=amendment_folder,
            contract_id=contract_id,
            callbacks=None
        )
        
        # Verify: Amendment text is parsed correctly
        assert isinstance(amendment_text, str)
        assert len(amendment_text) > 0
        assert "Section 5 - Termination" in amendment_text
        assert "60 days written notice" in amendment_text
        
        # ====================================================================
        # Step 3: Mock Contextualization Agent (Agent 1)
        # ====================================================================
        # Expected contextualized output - should focus on relevant sections
        contextualized_original = "Section 5 - Termination\nThis agreement may be terminated by either party with 30 days written notice."
        contextualized_amendment = "Section 5 - Termination\nThis agreement may be terminated by either party with 60 days written notice."
        
        mock_contextualized_output = ContextualizedContract(
            original_contract_text=contextualized_original,
            amendment_text=contextualized_amendment
        )
        
        # Mock contextualization model
        mock_contextualization_instance = Mock()
        mock_contextualization_instance.with_structured_output.return_value.invoke.return_value = mock_contextualized_output
        mock_contextualization_model.return_value = mock_contextualization_instance
        
        # Execute: Contextualize documents using LangChain tool
        context_dict = contextualize_documents.invoke(
            {
                "original_text": original_text,
                "amendment_text": amendment_text,
                "contract_id": contract_id
            }
        )
        context = ContextualizedContract(**context_dict)
        
        # Verify: Contextualization output is valid
        assert isinstance(context, ContextualizedContract)
        assert len(context.original_contract_text) >= 5
        assert len(context.amendment_text) >= 5
        assert "Section 5" in context.original_contract_text
        assert "Section 5" in context.amendment_text
        
        # Verify: Contextualization agent was called with full texts
        mock_contextualization_instance.with_structured_output.return_value.invoke.assert_called_once()
        contextualization_call_args = mock_contextualization_instance.with_structured_output.return_value.invoke.call_args
        messages = contextualization_call_args[0][0]  # Get the messages list
        
        # Extract content from messages (could be SystemMessage + HumanMessage or just HumanMessage)
        contextualization_prompt_text = ""
        for msg in messages:
            if hasattr(msg, 'content'):
                contextualization_prompt_text += str(msg.content) + " "
        
        # Verify original and amendment texts were passed to contextualization
        assert original_text in contextualization_prompt_text or "30 days" in contextualization_prompt_text
        assert amendment_text in contextualization_prompt_text or "60 days" in contextualization_prompt_text
        
        # ====================================================================
        # Step 4: Mock Extraction Agent (Agent 2)
        # ====================================================================
        # Expected extraction output
        mock_extraction_output = ContractChangeSummary(
            topics_touched=["Termination", "Notice Period"],
            sections_changed=["Section 5 – Termination"],
            summary_of_the_change="Section 5: -Changed termination notice period from 30 days to 60 days"
        )
        
        # Mock extraction model
        mock_extraction_instance = Mock()
        mock_extraction_instance.with_structured_output.return_value.invoke.return_value = mock_extraction_output
        mock_extraction_model.return_value = mock_extraction_instance
        
        # Execute: Extract changes using contextualized text with LangChain tool
        result_dict = extract_changes.invoke(
            {
                "original_text": context.original_contract_text,
                "amendment_text": context.amendment_text,
                "contract_id": contract_id
            }
        )
        result = ContractChangeSummary(**result_dict)
        
        # Verify: Final output is valid ContractChangeSummary
        assert isinstance(result, ContractChangeSummary)
        assert len(result.topics_touched) > 0
        assert len(result.sections_changed) > 0
        assert len(result.summary_of_the_change) >= 5
        
        # Verify: Extraction agent received contextualized text (not original full text)
        mock_extraction_instance.with_structured_output.return_value.invoke.assert_called_once()
        extraction_call_args = mock_extraction_instance.with_structured_output.return_value.invoke.call_args
        messages = extraction_call_args[0][0]  # Get the messages list
        
        # Extract content from messages (could be SystemMessage + HumanMessage or just HumanMessage)
        extraction_prompt_text = ""
        for msg in messages:
            if hasattr(msg, 'content'):
                extraction_prompt_text += str(msg.content) + " "
        
        # Verify extraction agent received contextualized versions
        assert context.original_contract_text in extraction_prompt_text
        assert context.amendment_text in extraction_prompt_text
        
        # Verify extraction agent did NOT receive the full original texts
        # (contextualized text should be shorter/more focused)
        assert len(context.original_contract_text) <= len(original_text)
        assert len(context.amendment_text) <= len(amendment_text)
        
        # ====================================================================
        # Final Verification: Data Flow Integrity
        # ====================================================================
        # Verify the complete data flow:
        # 1. Image parsing produces full contract texts
        assert "30 days" in original_text
        assert "60 days" in amendment_text
        
        # 2. Contextualization extracts relevant sections
        assert "30 days" in context.original_contract_text
        assert "60 days" in context.amendment_text
        
        # 3. Extraction produces structured summary
        assert "30" in result.summary_of_the_change or "60" in result.summary_of_the_change
        assert "Section 5" in result.sections_changed[0]
        assert "Termination" in result.topics_touched[0]
        
        # Verify final output can be serialized (for API/JSON responses)
        result_dict = result.model_dump()
        assert isinstance(result_dict, dict)
        assert "topics_touched" in result_dict
        assert "sections_changed" in result_dict
        assert "summary_of_the_change" in result_dict
    
    @patch('src.agents.extraction_agent.ChatOpenAI')
    @patch('src.agents.contextualization_agent.ChatOpenAI')
    @patch('src.image_parser.ChatOpenAI')
    @patch('src.image_parser.os.getenv')
    @patch('src.image_parser.encode_image')
    @patch('src.image_parser.os.listdir')
    def test_pipeline_with_multiple_sections(
        self,
        mock_listdir,
        mock_encode_image,
        mock_getenv,
        mock_image_chat_model,
        mock_contextualization_model,
        mock_extraction_model
    ):
        """
        Test the pipeline with a more complex contract containing multiple sections.
        """
        contract_id = "test_contract_e2e_002"
        
        # Setup environment
        mock_getenv.side_effect = lambda key: {
            "IMAGE_MULTIMODAL_MODEL": "openai/gpt-4-vision",
            "LLM_MODEL": "openai/gpt-4",
            "LLM_API_KEY": "test_key",
            "LLM_BASE_URL": "test_url"
        }.get(key)
        
        # Mock original contract with multiple sections
        mock_listdir.return_value = ["contract_page_1.png", "contract_page_2.png"]
        mock_encode_image.return_value = "base64_encoded"
        
        original_full = (
            "CONTRACT AGREEMENT\n\n"
            "Section 1 - Payment Terms\nPayment shall be made within 30 days of invoice.\n\n"
            "Section 2 - Delivery\nDelivery must occur within 60 days of order.\n\n"
            "Section 3 - Warranty\nWarranty period is 12 months from delivery date."
        )
        
        mock_image_response = Mock()
        mock_image_response.content = original_full
        mock_image_model_instance = Mock()
        mock_image_model_instance.invoke.return_value = mock_image_response
        mock_image_chat_model.return_value = mock_image_model_instance
        
        original_text = parse_full_contract("original_folder", contract_id, None)
        
        # Mock amendment
        amendment_full = (
            "AMENDMENT\n\n"
            "Section 1 - Payment Terms\nPayment shall be made within 45 days of invoice.\n\n"
            "Section 3 - Warranty\nWarranty period is 18 months from delivery date."
        )
        
        mock_amendment_response = Mock()
        mock_amendment_response.content = amendment_full
        mock_image_model_instance.invoke.return_value = mock_amendment_response
        
        amendment_text = parse_full_contract("amendment_folder", contract_id, None)
        
        # Mock contextualization
        contextualized = ContextualizedContract(
            original_contract_text=(
                "Section 1 - Payment Terms\nPayment shall be made within 30 days of invoice.\n\n"
                "Section 3 - Warranty\nWarranty period is 12 months from delivery date."
            ),
            amendment_text=(
                "Section 1 - Payment Terms\nPayment shall be made within 45 days of invoice.\n\n"
                "Section 3 - Warranty\nWarranty period is 18 months from delivery date."
            )
        )
        
        mock_contextualization_instance = Mock()
        mock_contextualization_instance.with_structured_output.return_value.invoke.return_value = contextualized
        mock_contextualization_model.return_value = mock_contextualization_instance
        
        context_dict = contextualize_documents.invoke(
            {
                "original_text": original_text,
                "amendment_text": amendment_text,
                "contract_id": contract_id
            }
        )
        context = ContextualizedContract(**context_dict)
        
        # Mock extraction
        extraction_result = ContractChangeSummary(
            topics_touched=["Payment Terms", "Warranty"],
            sections_changed=["Section 1 – Payment Terms", "Section 3 – Warranty"],
            summary_of_the_change=(
                "Section 1: -Changed payment terms from 30 days to 45 days\n"
                "Section 3: -Extended warranty period from 12 months to 18 months"
            )
        )
        
        mock_extraction_instance = Mock()
        mock_extraction_instance.with_structured_output.return_value.invoke.return_value = extraction_result
        mock_extraction_model.return_value = mock_extraction_instance
        
        result_dict = extract_changes.invoke(
            {
                "original_text": context.original_contract_text,
                "amendment_text": context.amendment_text,
                "contract_id": contract_id
            }
        )
        result = ContractChangeSummary(**result_dict)
        
        # Verify multiple sections are handled
        assert len(result.topics_touched) == 2
        assert len(result.sections_changed) == 2
        assert "Payment Terms" in result.topics_touched
        assert "Warranty" in result.topics_touched
        assert "Section 1" in result.sections_changed[0]
        assert "Section 3" in result.sections_changed[1]
    
    @patch('src.agents.extraction_agent.ChatOpenAI')
    @patch('src.agents.contextualization_agent.ChatOpenAI')
    @patch('src.image_parser.parse_contract_image_with_fallback_model')
    @patch('src.image_parser.ChatOpenAI')
    @patch('src.image_parser.os.getenv')
    @patch('src.image_parser.encode_image')
    @patch('src.image_parser.os.listdir')
    def test_pipeline_with_fallback_model(
        self,
        mock_listdir,
        mock_encode_image,
        mock_getenv,
        mock_image_chat_model,
        mock_fallback,
        mock_contextualization_model,
        mock_extraction_model
    ):
        """
        Test the pipeline when image parsing requires fallback model.
        """
        contract_id = "test_contract_e2e_003"
        
        # Setup environment
        mock_getenv.side_effect = lambda key: {
            "IMAGE_MULTIMODAL_MODEL": "openai/gpt-4-vision",
            "LLM_MODEL": "openai/gpt-4",
            "LLM_API_KEY": "test_key",
            "LLM_BASE_URL": "test_url"
        }.get(key)
        
        # Mock primary model failure, fallback success
        mock_listdir.return_value = ["contract_page_1.png"]
        mock_encode_image.return_value = "base64_encoded"
        
        mock_image_model_instance = Mock()
        mock_image_model_instance.invoke.side_effect = Exception("Primary model failed")
        mock_image_chat_model.return_value = mock_image_model_instance
        
        # Fallback model succeeds
        mock_fallback.return_value = "Fallback extracted contract text\nSection 1: Terms"
        
        # Parse should use fallback
        original_text = parse_full_contract("original_folder", contract_id, None)
        
        assert "Fallback extracted" in original_text
        mock_fallback.assert_called()
        
        # Continue with rest of pipeline
        amendment_text = "Amendment text"
        
        # Mock contextualization
        contextualized = ContextualizedContract(
            original_contract_text="Section 1: Terms",
            amendment_text=amendment_text
        )
        
        mock_contextualization_instance = Mock()
        mock_contextualization_instance.with_structured_output.return_value.invoke.return_value = contextualized
        mock_contextualization_model.return_value = mock_contextualization_instance
        
        context_dict = contextualize_documents.invoke(
            {
                "original_text": original_text,
                "amendment_text": amendment_text,
                "contract_id": contract_id
            }
        )
        context = ContextualizedContract(**context_dict)
        
        # Mock extraction
        extraction_result = ContractChangeSummary(
            topics_touched=["Terms"],
            sections_changed=["Section 1"],
            summary_of_the_change="Section 1: -Updated terms"
        )
        
        mock_extraction_instance = Mock()
        mock_extraction_instance.with_structured_output.return_value.invoke.return_value = extraction_result
        mock_extraction_model.return_value = mock_extraction_instance
        
        result_dict = extract_changes.invoke(
            {
                "original_text": context.original_contract_text,
                "amendment_text": context.amendment_text,
                "contract_id": contract_id
            }
        )
        result = ContractChangeSummary(**result_dict)
        
        # Verify pipeline completes successfully even with fallback
        assert isinstance(result, ContractChangeSummary)
        assert len(result.topics_touched) > 0
