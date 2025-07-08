from unittest.mock import patch
from cli_bridge import bridge_conversation

@patch("cli_bridge.send_to_claude")
@patch("cli_bridge.send_to_chatgpt")
def test_bridge_conversation_turns(mock_chatgpt, mock_claude):
    # Arrange: set up predictable mock responses
    mock_claude.return_value = "Response from Claude"
    mock_chatgpt.return_value = "Reply from ChatGPT"
    
    initial_prompt = "Start"  # String, not list
    
    # Act
    transcript = bridge_conversation(
        prompt=initial_prompt,  # Fixed parameter name
        turns=2,
        mock=True
    )
    
    # Assert
    assert len(transcript) > 0
    assert transcript[0]["role"] == "user"
    assert transcript[0]["content"] == "Start"
