import pytest
from unittest.mock import patch, MagicMock
from claude_wrapper import send_to_claude

@patch("claude_wrapper.requests.post")
def test_claude_latency(mock_post, caplog):
    # Arrange: fake successful response
    fake = MagicMock()
    fake.status_code = 200
    fake.json.return_value = {"assistant_response": {"role": "assistant", "content": "X"}}
    mock_post.return_value = fake

    # Capture INFO logs from our claude logger
    caplog.set_level("INFO", logger="bridge.claude")

    # Act
    send_to_claude([{"role": "user", "content": "hi"}])

    # Assert: we logged a latency line
    assert any("Claude call latency" in rec.message for rec in caplog.records)

