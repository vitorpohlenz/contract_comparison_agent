import sys
sys.dont_write_bytecode = True

from pathlib import Path
import pytest
from unittest.mock import Mock, patch, MagicMock
import os
from pydantic import ValidationError

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.models import ContractChangeSummary, ContextualizedContract
from src.agents.contextualization_agent import contextualize_documents
from src.agents.extraction_agent import extract_changes
from src.image_parser import parse_contract_image, parse_full_contract


# ============================================================================
# (1) Pydantic Validation Tests
# ============================================================================

class TestPydanticValidation:
    """Test Pydantic model validation with valid and invalid inputs."""
    
    def test_contract_change_summary_valid(self):
        """Test ContractChangeSummary with valid data."""
        data = {
            "topics_touched": ["Termination", "Liability"],
            "sections_changed": ["Section 5 – Termination"],
            "summary_of_the_change": "Section 5: -change_1\n-change_2"
        }
        model = ContractChangeSummary(**data)
        assert model.topics_touched == ["Termination", "Liability"]
        assert model.sections_changed == ["Section 5 – Termination"]
        assert len(model.summary_of_the_change) >= 5
        assert model.summary_of_the_change == "Section 5: -change_1\n-change_2"
    
    def test_contract_change_summary_invalid_empty_topics(self):
        """Test ContractChangeSummary fails with empty topics_touched."""
        with pytest.raises(ValidationError):
            ContractChangeSummary(
                topics_touched=[],
                sections_changed=["Section 5"],
                summary_of_the_change="Valid summary text here"
            )
    
    def test_contract_change_summary_invalid_empty_sections(self):
        """Test ContractChangeSummary fails with empty sections_changed."""
        with pytest.raises(ValidationError):
            ContractChangeSummary(
                topics_touched=["Termination"],
                sections_changed=[],
                summary_of_the_change="Valid summary text here"
            )
    
    def test_contract_change_summary_invalid_short_summary(self):
        """Test ContractChangeSummary fails with empty summary."""
        with pytest.raises(ValidationError):
            ContractChangeSummary(
                topics_touched=["Termination"],
                sections_changed=["Section 5"],
                summary_of_the_change=""
            )
    
    def test_contract_change_summary_invalid_missing_fields(self):
        """Test ContractChangeSummary fails with missing required fields."""
        with pytest.raises(ValidationError):
            ContractChangeSummary(
                topics_touched=["Termination"]
                # Missing sections_changed and summary_of_the_change
            )
    
    def test_contextualized_contract_valid(self):
        """Test ContextualizedContract with valid data."""
        data = {
            "original_contract_text": "This is the original contract text that is impacted by the amendment.",
            "amendment_text": "This is the amendment text that modifies the contract."
        }
        model = ContextualizedContract(**data)
        assert len(model.original_contract_text) >= 5
        assert len(model.amendment_text) >= 5
        assert model.original_contract_text == data["original_contract_text"]
        assert model.amendment_text == data["amendment_text"]
    
    def test_contextualized_contract_invalid_short_original(self):
        """Test ContextualizedContract fails with empty original text."""
        with pytest.raises(ValidationError):
            ContextualizedContract(
                original_contract_text="",
                amendment_text="This is a valid amendment text that is long enough."
            )
    
    def test_contextualized_contract_invalid_short_amendment(self):
        """Test ContextualizedContract fails with empty amendment text."""
        with pytest.raises(ValidationError):
            ContextualizedContract(
                original_contract_text="This is a valid original contract text that is long enough.",
                amendment_text=""
            )
    
    def test_contextualized_contract_invalid_missing_fields(self):
        """Test ContextualizedContract fails with missing required fields."""
        with pytest.raises(ValidationError):
            ContextualizedContract(
                original_contract_text="Valid original text here"
                # Missing amendment_text
            )


# ============================================================================
# (2) Agent Handoff Test
# ============================================================================

class TestAgentHandoff:
    """Test that Agent 2 (extraction_agent) receives Agent 1's (contextualization_agent) output."""
    
    @patch('src.agents.contextualization_agent.ChatOpenAI')
    @patch('src.agents.extraction_agent.ChatOpenAI')
    def test_agent_handoff_contextualization_to_extraction(self, mock_extraction_model, mock_contextualization_model):
        """Verify that extraction agent receives contextualization agent's output."""
        # Setup: Mock Agent 1 (contextualization_agent) output
        mock_contextualized_output = ContextualizedContract(
            original_contract_text="Original contract section about termination with 30 days notice.",
            amendment_text="Amendment changes termination notice to 60 days."
        )
        
        # Mock the contextualization model's response
        mock_contextualization_instance = Mock()
        mock_contextualization_instance.with_structured_output.return_value.invoke.return_value = mock_contextualized_output
        mock_contextualization_model.return_value = mock_contextualization_instance
        
        # Setup: Mock Agent 2 (extraction_agent) output
        mock_extraction_output = ContractChangeSummary(
            topics_touched=["Termination"],
            sections_changed=["Section 5 – Termination"],
            summary_of_the_change="Section 5: -Changed notice period from 30 to 60 days"
        )
        
        # Mock the extraction model's response
        mock_extraction_instance = Mock()
        mock_extraction_instance.with_structured_output.return_value.invoke.return_value = mock_extraction_output
        mock_extraction_model.return_value = mock_extraction_instance
        
        # Execute: Run Agent 1 (contextualization)
        original_text = "Full original contract text..."
        amendment_text = "Full amendment text..."
        contract_id = "test_contract_123"
        
        context_result = contextualize_documents(
            original_text=original_text,
            amendment_text=amendment_text,
            contract_id=contract_id,
            callbacks=None
        )
        
        # Verify Agent 1 output
        assert isinstance(context_result, ContextualizedContract)
        assert context_result.original_contract_text == mock_contextualized_output.original_contract_text
        assert context_result.amendment_text == mock_contextualized_output.amendment_text
        
        # Execute: Run Agent 2 (extraction) with Agent 1's output
        extraction_result = extract_changes(
            original_text=context_result.original_contract_text,
            amendment_text=context_result.amendment_text,
            contract_id=contract_id,
            callbacks=None
        )
        
        # Verify Agent 2 received Agent 1's output
        assert isinstance(extraction_result, ContractChangeSummary)
        
        # Verify that extraction agent was called with contextualized text
        mock_extraction_instance.with_structured_output.return_value.invoke.assert_called_once()
        call_args = mock_extraction_instance.with_structured_output.return_value.invoke.call_args
        
        # Extract the prompt from the call to verify it contains Agent 1's output
        prompt_messages = call_args[0][0]
        prompt_text = str(prompt_messages)
        
        # Verify Agent 2 received Agent 1's contextualized original text
        assert context_result.original_contract_text in prompt_text or \
               mock_contextualized_output.original_contract_text in prompt_text
        
        # Verify Agent 2 received Agent 1's contextualized amendment text
        assert context_result.amendment_text in prompt_text or \
               mock_contextualized_output.amendment_text in prompt_text
    
    @patch('src.agents.contextualization_agent.ChatOpenAI')
    @patch('src.agents.extraction_agent.ChatOpenAI')
    def test_agent_handoff_data_integrity(self, mock_extraction_model, mock_contextualization_model):
        """Test that data integrity is maintained during agent handoff."""
        # Setup mock outputs
        original_contextualized = "Contextualized original: Section 5 about termination"
        amendment_contextualized = "Contextualized amendment: Change to 60 days"
        
        mock_contextualized_output = ContextualizedContract(
            original_contract_text=original_contextualized,
            amendment_text=amendment_contextualized
        )
        
        mock_contextualization_instance = Mock()
        mock_contextualization_instance.with_structured_output.return_value.invoke.return_value = mock_contextualized_output
        mock_contextualization_model.return_value = mock_contextualization_instance
        
        mock_extraction_output = ContractChangeSummary(
            topics_touched=["Termination"],
            sections_changed=["Section 5"],
            summary_of_the_change="Section 5: -Changed notice period"
        )
        
        mock_extraction_instance = Mock()
        mock_extraction_instance.with_structured_output.return_value.invoke.return_value = mock_extraction_output
        mock_extraction_model.return_value = mock_extraction_instance
        
        # Execute handoff
        context = contextualize_documents(
            original_text="Full original",
            amendment_text="Full amendment",
            contract_id="test",
            callbacks=None
        )
        
        result = extract_changes(
            original_text=context.original_contract_text,
            amendment_text=context.amendment_text,
            contract_id="test",
            callbacks=None
        )
        
        # Verify data integrity: extraction should have used contextualized text
        call_args = mock_extraction_instance.with_structured_output.return_value.invoke.call_args
        prompt_text = str(call_args[0][0])
        
        # The extraction agent should have received the contextualized versions
        assert original_contextualized in prompt_text
        assert amendment_contextualized in prompt_text


# ============================================================================
# (3) Image Parsing Test
# ============================================================================

class TestImageParsing:
    """Test image parsing functionality."""
    
    @patch('src.image_parser.os.getenv')
    @patch('src.image_parser.ChatOpenAI')
    @patch('src.image_parser.encode_image')
    def test_parse_contract_image_success(self, mock_encode_image, mock_chat_model, mock_getenv):
        """Test successful parsing of a single contract image."""
        # Setup: Mock environment variables
        mock_getenv.side_effect = lambda key: {
            "IMAGE_MULTIMODAL_MODEL": "openai/gpt-4-vision",
            "LLM_API_KEY": "test_key",
            "LLM_BASE_URL": "test_url"
        }.get(key)
        
        # Setup: Mock image encoding
        mock_encoded_image = "base64_encoded_image_string"
        mock_encode_image.return_value = mock_encoded_image
        
        # Setup: Mock model response
        mock_response = Mock()
        mock_response.content = "Extracted contract text from image\nSection 1: Terms and Conditions\nSection 2: Payment Terms"
        
        mock_model_instance = Mock()
        mock_model_instance.invoke.return_value = mock_response
        mock_chat_model.return_value = mock_model_instance
        
        # Execute
        result = parse_contract_image(
            image_path="test_image.png",
            contract_id="test_123",
            callbacks=None
        )
        
        # Verify
        assert isinstance(result, str)
        assert len(result) > 0
        assert "Extracted contract text" in result
        mock_encode_image.assert_called_once_with("test_image.png")
        mock_model_instance.invoke.assert_called_once()
    
    @patch('src.image_parser.os.getenv')
    @patch('src.image_parser.ChatOpenAI')
    @patch('src.image_parser.encode_image')
    @patch('src.image_parser.parse_contract_image_with_fallback_model')
    def test_parse_contract_image_fallback_on_empty(self, mock_fallback, mock_encode_image, mock_chat_model, mock_getenv):
        """Test that fallback model is used when primary model returns empty text."""
        # Setup: Mock environment variables
        mock_getenv.side_effect = lambda key: {
            "IMAGE_MULTIMODAL_MODEL": "openai/gpt-4-vision",
            "LLM_API_KEY": "test_key",
            "LLM_BASE_URL": "test_url"
        }.get(key)
        
        # Setup: Primary model returns empty
        mock_encode_image.return_value = "base64_encoded"
        mock_response = Mock()
        mock_response.content = ""  # Empty response
        
        mock_model_instance = Mock()
        mock_model_instance.invoke.return_value = mock_response
        mock_chat_model.return_value = mock_model_instance
        
        # Setup: Fallback returns valid text
        mock_fallback.return_value = "Fallback extracted text"
        
        # Execute
        result = parse_contract_image(
            image_path="test_image.png",
            contract_id="test_123",
            callbacks=None
        )
        
        # Verify fallback was called
        mock_fallback.assert_called_once_with("test_image.png", "test_123", None)
        assert result == "Fallback extracted text"
    
    @patch('src.image_parser.os.getenv')
    @patch('src.image_parser.ChatOpenAI')
    @patch('src.image_parser.encode_image')
    @patch('src.image_parser.parse_contract_image_with_fallback_model')
    def test_parse_contract_image_fallback_on_exception(self, mock_fallback, mock_encode_image, mock_chat_model, mock_getenv):
        """Test that fallback model is used when primary model raises exception."""
        # Setup: Mock environment variables
        mock_getenv.side_effect = lambda key: {
            "IMAGE_MULTIMODAL_MODEL": "openai/gpt-4-vision",
            "LLM_API_KEY": "test_key",
            "LLM_BASE_URL": "test_url"
        }.get(key)
        
        # Setup: Primary model raises exception
        mock_encode_image.return_value = "base64_encoded"
        mock_model_instance = Mock()
        mock_model_instance.invoke.side_effect = Exception("Primary model failed")
        mock_chat_model.return_value = mock_model_instance
        
        # Setup: Fallback returns valid text
        mock_fallback.return_value = "Fallback extracted text"
        
        # Execute
        result = parse_contract_image(
            image_path="test_image.png",
            contract_id="test_123",
            callbacks=None
        )
        
        # Verify fallback was called
        mock_fallback.assert_called_once_with("test_image.png", "test_123", None)
        assert result == "Fallback extracted text"
    
    @patch('src.image_parser.parse_contract_image')
    @patch('src.image_parser.os.listdir')
    def test_parse_full_contract_multiple_images(self, mock_listdir, mock_parse_image):
        """Test parsing a folder with multiple images."""
        # Setup: Mock folder with multiple images
        mock_listdir.return_value = [
            "contract_page_1.png",
            "contract_page_2.png",
            "contract_page_3.png"
        ]
        
        # Setup: Mock individual image parsing results
        mock_parse_image.side_effect = [
            "Page 1 text\n",
            "Page 2 text\n",
            "Page 3 text\n"
        ]
        
        # Execute
        result = parse_full_contract(
            images_folder="test_folder",
            contract_id="test_123",
            callbacks=None
        )
        
        # Verify
        assert isinstance(result, str)
        assert "Page 1 text" in result
        assert "Page 2 text" in result
        assert "Page 3 text" in result
        assert mock_parse_image.call_count == 3
    
    @patch('src.image_parser.parse_contract_image')
    @patch('src.image_parser.os.listdir')
    def test_parse_full_contract_filters_non_images(self, mock_listdir, mock_parse_image):
        """Test that parse_full_contract filters out non-image files."""
        # Setup: Folder with images and non-image files
        mock_listdir.return_value = [
            "contract_page_1.png",
            "readme.txt",
            "contract_page_2.jpg",
            "data.json",
            "contract_page_3.png"
        ]
        
        mock_parse_image.side_effect = [
            "Page 1 text\n",
            "Page 2 text\n",
            "Page 3 text\n"
        ]
        
        # Execute
        result = parse_full_contract(
            images_folder="test_folder",
            contract_id="test_123",
            callbacks=None
        )
        
        # Verify: Only image files should be parsed (3 calls, not 5)
        assert mock_parse_image.call_count == 3
        assert "Page 1 text" in result
        assert "Page 2 text" in result
        assert "Page 3 text" in result
