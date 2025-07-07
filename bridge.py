import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Client as ClaudeClient
from metrics import compare_responses

# Load environment variables from .env file
load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# 1) Initialize OpenAI and Anthropic clients using secrets.toml
# ──────────────────────────────────────────────────────────────────────────────
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
claude = ClaudeClient(api_key=os.getenv("CLAUDE_API_KEY"))
st.title("ChatGPT ↔ Claude Bridge")
# ──────────────────────────────────────────────────────────────────────────────
# 2) Inputs from the user: prompt + (optional) keywords
# ──────────────────────────────────────────────────────────────────────────────
prompt = st.text_area("Enter Prompt", height=150)
keywords_input = st.text_input("Keywords (comma-sep)", "")
keywords = {k.strip() for k in keywords_input.split(",") if k.strip()}
if st.button("Run Comparison"):
    with st.spinner("Calling APIs…"):
        # ──────────────────────────────────────────────────────────────────
        # 3a) Call OpenAI's chat completion (GPT-4o-mini)
        # ──────────────────────────────────────────────────────────────────
        res_a = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        resp_a = res_a.choices[0].message.content
        # ──────────────────────────────────────────────────────────────────
        # 3b) Call Anthropic's Messages API (Claude Opus 4) with lowercase roles
        # ──────────────────────────────────────────────────────────────────
        anthropic_messages = [
            {"role": "user", "content": prompt}
        ]
        res_b = claude.messages.create(
            model="claude-opus-4-20250514",
            messages=anthropic_messages,
            max_tokens=1000
        )
        resp_b = res_b.content[0].text  # Fixed: Use Anthropic's response format
    # ──────────────────────────────────────────────────────────────────────────
    # 4) Display both LLM responses side by side
    # ──────────────────────────────────────────────────────────────────────────
    st.subheader("Responses")
    cols = st.columns(2)
    cols[0].markdown(f"**ChatGPT (gpt-4o-mini)**  \n{resp_a}")
    cols[1].markdown(f"**Claude (claude-opus-4-20250514)**  \n{resp_b}")
    # ──────────────────────────────────────────────────────────────────────────
    # 5) Compute & display keyword‐based comparison metrics
    # ──────────────────────────────────────────────────────────────────────────
    results = compare_responses(resp_a, resp_b, keywords)
    st.subheader("Metrics")
    st.json(results)
