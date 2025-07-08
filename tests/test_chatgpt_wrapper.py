import os
os.environ["OPENAI_API_KEY"] = "test"
os.environ["CLAUDE_API_KEY"] = "test"

from chatgpt_wrapper import send_to_chatgpt
from unittest.mock import patch

@patch("chatgpt_wrapper.requests.post")
def test_chatgpt_wrapper_basic(mock_post):
    # Mock successful response
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {
        "choices": [
            {"message": {"role": "assistant", "content": "Hello from ChatGPT!"}}
        ]
    }

    result = send_to_chatgpt([{"role": "user", "content": "Hi"}])
    assert result == "Hello from ChatGPT!"

