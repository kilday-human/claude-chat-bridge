import pytest
from unittest.mock import patch, MagicMock
from requests.exceptions import HTTPError

from claude_wrapper import send_to_claude
from chatgpt_wrapper import send_to_chatgpt

# 1) Test retry on HTTP 500 and eventual failure (Claude)
@patch("claude_wrapper.requests.post")
def test_claude_retries_http_error(mock_post):
    fake = MagicMock()
    fake.raise_for_status.side_effect = HTTPError("500 Server Error")
    mock_post.return_value = fake

    with pytest.raises(HTTPError):
        send_to_claude([{"role": "user", "content": "hi"}])

# 2) Test malformed JSON path (Claude)
@patch("claude_wrapper.requests.post")
def test_claude_malformed_json(mock_post):
    fake = MagicMock(status_code=200)
    fake.json.return_value = {"unexpected": {}}
    mock_post.return_value = fake

    with pytest.raises(ValueError) as ei:
        send_to_claude([{"role": "user", "content": "hi"}])
    assert "Unexpected response format" in str(ei.value)

# 3) Test retry on HTTP 503 (ChatGPT)
@patch("chatgpt_wrapper.requests.post")
def test_chatgpt_retries_http_error(mock_post):
    fake = MagicMock()
    fake.raise_for_status.side_effect = HTTPError("503 Service Unavailable")
    mock_post.return_value = fake

    with pytest.raises(HTTPError):
        send_to_chatgpt([{"role": "user", "content": "hello"}])

# 4) Test malformed JSON path (ChatGPT)
@patch("chatgpt_wrapper.requests.post")
def test_chatgpt_malformed_json(mock_post):
    fake = MagicMock(status_code=200)
    fake.json.return_value = {"choices": []}  # no message
    mock_post.return_value = fake

    with pytest.raises(ValueError) as ei:
        send_to_chatgpt([{"role": "user", "content": "hello"}])
    assert "Unexpected response format" in str(ei.value)

