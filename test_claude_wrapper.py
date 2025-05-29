import pytest
from unittest.mock import patch
from claude_wrapper import send_to_claude

@patch("claude_wrapper.requests.post")
def test_claude_wrapper_basic(mock_post):
    # Arrange: mock a successful Claude response
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {
        "assistant_response": {"role": "assistant", "content": "Hi there!"}
    }

    # Act
    result = send_to_claude([{"role": "user", "content": "Hello, Claude!"}])

    # Assert
    assert isinstance(result, dict)
    assert result["role"] == "assistant"
    assert "Hi there!" in result["content"]

