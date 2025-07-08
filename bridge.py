import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Client as ClaudeClient
from metrics import compare_responses
import time
import traceback

# Load environment variables from .env file
load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# 1) Initialize OpenAI and Anthropic clients
# ──────────────────────────────────────────────────────────────────────────────
try:
    openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    claude = ClaudeClient(api_key=os.getenv("CLAUDE_API_KEY"))
except Exception as e:
    st.error(f"❌ Failed to initialize API clients: {e}")
    st.stop()

st.title("ChatGPT ↔ Claude Bridge")
st.markdown("*Compare responses from ChatGPT and Claude side-by-side*")

# ──────────────────────────────────────────────────────────────────────────────
# 2) Inputs from the user: prompt + (optional) keywords
# ──────────────────────────────────────────────────────────────────────────────
prompt = st.text_area("Enter Prompt", height=150, placeholder="e.g., Explain quantum computing in simple terms")
keywords_input = st.text_input("Keywords (comma-sep)", "", placeholder="Optional: quantum, physics, computing")
keywords = {k.strip() for k in keywords_input.split(",") if k.strip()}

def call_openai_with_retry(prompt, max_retries=3):
    """Call OpenAI with retry logic and error handling"""
    for attempt in range(max_retries):
        try:
            res = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                timeout=30
            )
            return res.choices[0].message.content, None
        except Exception as e:
            if attempt == max_retries - 1:
                return None, f"OpenAI Error (after {max_retries} attempts): {str(e)}"
            time.sleep(2 ** attempt)  # Exponential backoff
    
def call_claude_with_retry(prompt, max_retries=3):
    """Call Claude with retry logic and error handling"""
    for attempt in range(max_retries):
        try:
            res = claude.messages.create(
                model="claude-3-haiku-20240307",  # Use working model
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000
            )
            return res.content[0].text, None
        except Exception as e:
            error_msg = str(e)
            if "overloaded" in error_msg.lower():
                if attempt == max_retries - 1:
                    return None, "🔄 Claude is currently overloaded. Please try again in a few minutes."
                time.sleep(5)  # Wait longer for overload errors
            elif "rate_limit" in error_msg.lower():
                return None, "⏱️ Rate limit exceeded. Please wait before trying again."
            else:
                if attempt == max_retries - 1:
                    return None, f"Claude Error (after {max_retries} attempts): {error_msg}"
                time.sleep(2 ** attempt)

if st.button("🚀 Run Comparison", type="primary"):
    if not prompt.strip():
        st.warning("⚠️ Please enter a prompt before running the comparison.")
    else:
        # Create columns for side-by-side display
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🤖 ChatGPT (GPT-4o-mini)")
            with st.spinner("Calling OpenAI..."):
                gpt_response, gpt_error = call_openai_with_retry(prompt)
            
            if gpt_error:
                st.error(gpt_error)
                st.code("Error details available in logs", language="text")
            else:
                st.success("✅ Response received")
                st.markdown(gpt_response)
        
        with col2:
            st.subheader("🧠 Claude (Haiku)")
            with st.spinner("Calling Anthropic..."):
                claude_response, claude_error = call_claude_with_retry(prompt)
            
            if claude_error:
                st.error(claude_error)
                if "overloaded" in claude_error.lower():
                    st.info("💡 Tip: Claude Haiku is usually less busy than Opus models")
            else:
                st.success("✅ Response received")
                st.markdown(claude_response)
        
        # ──────────────────────────────────────────────────────────────────
        # 4) Analysis section (only if both responses succeeded)
        # ──────────────────────────────────────────────────────────────────
        if not gpt_error and not claude_error and gpt_response and claude_response:
            st.divider()
            st.subheader("📊 Response Analysis")
            
            try:
                # Use your existing metrics comparison
                analysis = compare_responses(gpt_response, claude_response, keywords)
                st.json(analysis)
            except Exception as e:
                st.warning(f"⚠️ Could not generate analysis: {e}")
        
        elif gpt_error and claude_error:
            st.error("❌ Both APIs failed. Please check your API keys and try again.")
        elif gpt_error or claude_error:
            st.warning("⚠️ One API failed, but you can still compare the successful response.")

# ──────────────────────────────────────────────────────────────────────────────
# 5) Footer with usage info
# ──────────────────────────────────────────────────────────────────────────────
st.divider()
with st.expander("ℹ️ About this tool"):
    st.markdown("""
    **Claude-ChatGPT Bridge** allows you to compare responses from two leading AI models:
    - **ChatGPT**: OpenAI's GPT-4o-mini model
    - **Claude**: Anthropic's Claude-3 Haiku model
    
    **Features:**
    - 🔄 Automatic retry logic with exponential backoff
    - 🛡️ Robust error handling for API failures
    - 📊 Response analysis and comparison
    - 🎯 Optional keyword-based analysis
    
    **Tips:**
    - Leave keywords blank for general comparison
    - Add keywords (comma-separated) for focused analysis
    - If one API is overloaded, try again in a few minutes
    """)

# Display API status
if st.checkbox("🔧 Show API Status"):
    col1, col2 = st.columns(2)
    with col1:
        if os.getenv("OPENAI_API_KEY") and not os.getenv("OPENAI_API_KEY").startswith("your-"):
            st.success("✅ OpenAI API key configured")
        else:
            st.error("❌ OpenAI API key not configured")
    
    with col2:
        if os.getenv("CLAUDE_API_KEY") and not os.getenv("CLAUDE_API_KEY").startswith("your-"):
            st.success("✅ Claude API key configured")
        else:
            st.error("❌ Claude API key not configured")
