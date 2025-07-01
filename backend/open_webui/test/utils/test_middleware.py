import json
import pytest
from unittest.mock import Mock, AsyncMock, patch
from uuid import uuid4

from open_webui.utils.middleware import chat_completion_tools_handler
from open_webui.models.users import UserModel
from open_webui.config import ENABLE_TOOL_RESULT_PERSISTENCE


class TestMiddleware:
    """Test suite for middleware functionality, specifically tool persistence."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.mock_request = Mock()
        self.mock_request.app.state.config.TASK_MODEL = "test-model"
        self.mock_request.app.state.config.TASK_MODEL_EXTERNAL = "test-model-external"
        self.mock_request.app.state.config.TOOLS_FUNCTION_CALLING_PROMPT_TEMPLATE = ""
        
        self.mock_user = UserModel(
            id="test-user-id",
            name="Test User",
            email="test@example.com",
            role="user",
            profile_image_url="/user.png",
            last_active_at=1627351200,
            updated_at=1627351200,
            created_at=1627351200
        )
        
        self.test_body = {
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Please use the test tool"}
            ]
        }
        
        self.test_extra_params = {
            "__event_call__": AsyncMock(),
            "__metadata__": {"session_id": "test-session"}
        }
        
        self.test_models = {
            "test-model": {"connection_type": "local"}
        }
        
        self.test_tools = {
            "test_tool": {
                "spec": {
                    "name": "test_tool",
                    "parameters": {
                        "properties": {
                            "param1": {"type": "string"}
                        }
                    }
                },
                "callable": AsyncMock(return_value="Test tool result"),
                "tool_id": "test-tool-id",
                "metadata": {"citation": False}
            }
        }

    @pytest.mark.asyncio
    async def test_tool_persistence_enabled(self):
        """Test that tool calls and results are added to messages when persistence is enabled."""
        with patch('open_webui.utils.middleware.ENABLE_TOOL_RESULT_PERSISTENCE', True):
            
            with patch('open_webui.utils.middleware.generate_chat_completion') as mock_generate:
                # Mock the response from the function calling model
                mock_response = {
                    "choices": [{
                        "message": {
                            "content": '{"name": "test_tool", "parameters": {"param1": "value1"}}'
                        }
                    }]
                }
                mock_generate.return_value = mock_response
                
                # Call the handler
                result_body, result_flags = await chat_completion_tools_handler(
                    self.mock_request,
                    self.test_body.copy(),
                    self.test_extra_params,
                    self.mock_user,
                    self.test_models,
                    self.test_tools
                )
                
                # Verify that tool call and result messages were added
                messages = result_body["messages"]
                
                # Should have original user message + tool call message + tool result message
                assert len(messages) >= 3
                
                # Find the assistant message with tool_calls
                tool_call_message = None
                tool_result_message = None
                
                for msg in messages:
                    if msg.get("role") == "assistant" and msg.get("tool_calls"):
                        tool_call_message = msg
                    elif msg.get("role") == "tool":
                        tool_result_message = msg
                
                # Verify tool call message structure
                assert tool_call_message is not None
                assert tool_call_message["role"] == "assistant"
                assert tool_call_message["content"] is None
                assert len(tool_call_message["tool_calls"]) == 1
                
                tool_call = tool_call_message["tool_calls"][0]
                assert tool_call["type"] == "function"
                assert tool_call["function"]["name"] == "test_tool"
                assert "id" in tool_call
                
                # Verify tool result message structure
                assert tool_result_message is not None
                assert tool_result_message["role"] == "tool"
                assert tool_result_message["tool_call_id"] == tool_call["id"]
                assert tool_result_message["content"] == "Test tool result"

    @pytest.mark.asyncio
    async def test_tool_persistence_disabled(self):
        """Test that tool calls and results are NOT added to messages when persistence is disabled."""
        with patch('open_webui.utils.middleware.ENABLE_TOOL_RESULT_PERSISTENCE', False):
            
            with patch('open_webui.utils.middleware.generate_chat_completion') as mock_generate:
                # Mock the response from the function calling model
                mock_response = {
                    "choices": [{
                        "message": {
                            "content": '{"name": "test_tool", "parameters": {"param1": "value1"}}'
                        }
                    }]
                }
                mock_generate.return_value = mock_response
                
                original_message_count = len(self.test_body["messages"])
                
                # Call the handler
                result_body, result_flags = await chat_completion_tools_handler(
                    self.mock_request,
                    self.test_body.copy(),
                    self.test_extra_params,
                    self.mock_user,
                    self.test_models,
                    self.test_tools
                )
                
                # Verify that no tool call/result messages were added
                messages = result_body["messages"]
                
                # Should only have the original user message + any user message updates
                # but no assistant tool_calls or tool role messages
                tool_call_messages = [msg for msg in messages if msg.get("role") == "assistant" and msg.get("tool_calls")]
                tool_result_messages = [msg for msg in messages if msg.get("role") == "tool"]
                
                assert len(tool_call_messages) == 0
                assert len(tool_result_messages) == 0

    @pytest.mark.asyncio
    async def test_tool_call_id_consistency(self):
        """Test that tool call IDs are consistent between tool call and result messages."""
        with patch('open_webui.utils.middleware.ENABLE_TOOL_RESULT_PERSISTENCE', True):
            
            with patch('open_webui.utils.middleware.generate_chat_completion') as mock_generate:
                mock_response = {
                    "choices": [{
                        "message": {
                            "content": '{"name": "test_tool", "parameters": {"param1": "value1"}}'
                        }
                    }]
                }
                mock_generate.return_value = mock_response
                
                result_body, _ = await chat_completion_tools_handler(
                    self.mock_request,
                    self.test_body.copy(),
                    self.test_extra_params,
                    self.mock_user,
                    self.test_models,
                    self.test_tools
                )
                
                messages = result_body["messages"]
                
                # Find tool call and result messages
                tool_call_message = next(
                    (msg for msg in messages if msg.get("role") == "assistant" and msg.get("tool_calls")),
                    None
                )
                tool_result_message = next(
                    (msg for msg in messages if msg.get("role") == "tool"),
                    None
                )
                
                assert tool_call_message is not None
                assert tool_result_message is not None
                
                # Verify ID consistency
                tool_call_id = tool_call_message["tool_calls"][0]["id"]
                assert tool_result_message["tool_call_id"] == tool_call_id

    def test_tool_persistence_configuration_default(self):
        """Test that ENABLE_TOOL_RESULT_PERSISTENCE has the correct default value."""
        # The default should be True to enable persistence by default
        assert ENABLE_TOOL_RESULT_PERSISTENCE.value is True

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_persistence(self):
        """Test persistence behavior with multiple tool calls."""
        # Create a second tool for testing
        multi_tools = {
            **self.test_tools,
            "second_tool": {
                "spec": {
                    "name": "second_tool",
                    "parameters": {
                        "properties": {
                            "param2": {"type": "string"}
                        }
                    }
                },
                "callable": AsyncMock(return_value="Second tool result"),
                "tool_id": "second-tool-id",
                "metadata": {"citation": False}
            }
        }
        
        with patch('open_webui.utils.middleware.ENABLE_TOOL_RESULT_PERSISTENCE', True):
            
            with patch('open_webui.utils.middleware.generate_chat_completion') as mock_generate:
                # Mock response with multiple tool calls
                mock_response = {
                    "choices": [{
                        "message": {
                            "content": '{"tool_calls": [{"name": "test_tool", "parameters": {"param1": "value1"}}, {"name": "second_tool", "parameters": {"param2": "value2"}}]}'
                        }
                    }]
                }
                mock_generate.return_value = mock_response
                
                result_body, _ = await chat_completion_tools_handler(
                    self.mock_request,
                    self.test_body.copy(),
                    self.test_extra_params,
                    self.mock_user,
                    self.test_models,
                    multi_tools
                )
                
                messages = result_body["messages"]
                
                # Count tool-related messages
                tool_call_messages = [msg for msg in messages if msg.get("role") == "assistant" and msg.get("tool_calls")]
                tool_result_messages = [msg for msg in messages if msg.get("role") == "tool"]
                
                # Should have messages for each tool call when multiple tools are used
                # The exact number depends on how the handler processes multiple tools
                assert len(tool_call_messages) >= 1
                assert len(tool_result_messages) >= 1
