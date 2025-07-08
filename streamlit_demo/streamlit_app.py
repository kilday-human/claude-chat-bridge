import sys
import os

# Add the project root (one level up) to Python’s module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from src.bridge import bridge_conversation

st.title("Claude ↔ ChatGPT Bridge Demo")

# 1) User inputs
user_prompt = st.text_input("Enter your prompt:", value="Hello, how are you?")
turns = st.slider(
    "Number of Claude ↔ ChatGPT turns:", 
    min_value=1, 
    max_value=4, 
    value=2
)

# 2) Run the bridge (mock mode for now)
if st.button("Run Bridge"):
    mock_claude = lambda msgs: {"role": "assistant", "content": "[Mock Claude reply]"}
    mock_chatgpt = lambda msgs: {"role": "assistant", "content": "[Mock ChatGPT reply]"}

    transcript = bridge_conversation(
        initial_prompt=[{"role": "user", "content": user_prompt}],
        turns=turns,
        claude_fn=mock_claude,
        chatgpt_fn=mock_chatgpt,
    )

    # 3) Display the back-and-forth transcript
    for msg in transcript:
        speaker = msg["role"]
        source  = msg.get("from", "user")
        content = msg["content"]
        st.markdown(f"**{speaker} ({source}):** {content}")

