import pytest
from unittest.mock import patch

from bridge import bridge_conversation

@patch("bridge.send_to_claude")
@patch("bridge.send_to_chatgpt")
def test_bridge_conversation_turns(mock_chatgpt, mock_claude):
    # Arrange: set up predictable responses
    mock_claude.return_value = {"role": "assistant", "content": "Response from Claude"}
    mock_chatgpt.return_value = {"role": "assistant", "content": "Reply from ChatGPT"}

    # Act: run a 2-turn bridge (Claude → ChatGPT → Claude)
    transcript = bridge_conversation(
        initial_prompt=[{"role": "user", "content": "Start"}],
        turns=2
    )

    # Assert: verify call sequence and transcript format
    # Should alternate: user → Claude → ChatGPT → Claude
    expected = [
        {"role": "user",    "content": "Start"},
        {"role": "assistant","from": "claude",    "content": "Response from Claude"},
        {"role": "assistant","from": "chatgpt",   "content": "Reply from ChatGPT"},
        {"role": "assistant","from": "claude",    "content": "Response from Claude"},
    ]
    assert transcript == expected

    # Also ensure our mocks were called the right number of times
    assert mock_claude.call_count == 2
    assert mock_chatgpt.call_count == 1

