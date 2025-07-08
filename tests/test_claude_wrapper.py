import os
os.environ["OPENAI_API_KEY"] = "test"
os.environ["CLAUDE_API_KEY"] = "test"

from claude_wrapper import send_to_claude
from unittest.mock import patch

@patch("claude_wrapper.requests.post")
def test_claude_wrapper_basic(mock_post):
    # Mock successful response
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {
        "assistant_response": {"role": "assistant", "content": "Hi from Claude!"}
    }

    result = send_to_claude([{"role": "user", "content": "Hello"}])
    assert result == "Hi from Claude!"

