import pytest
from unittest.mock import patch
from chatgpt_wrapper import send_to_chatgpt

@patch("chatgpt_wrapper.requests.post")
def test_chatgpt_wrapper_basic(mock_post):
    # Arrange: mock a successful ChatGPT response
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {
        "choices": [
            {"message": {"role": "assistant", "content": "Hello from ChatGPT!"}}
        ]
    }

    # Act
    result = send_to_chatgpt([{"role": "user", "content": "Hi ChatGPT!"}])

    # Assert
    assert isinstance(result, dict)
    assert result["role"] == "assistant"
    assert "Hello from ChatGPT!" in result["content"]

