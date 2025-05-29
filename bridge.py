import os
import sys
import argparse
from requests.exceptions import HTTPError

from claude_wrapper import send_to_claude
from chatgpt_wrapper import send_to_chatgpt

def bridge_conversation(initial_prompt, turns=1, claude_fn=send_to_claude, chatgpt_fn=send_to_chatgpt):
    """
    Orchestrates a conversation: sends initial_prompt to Claude,
    then alternates between ChatGPT and Claude for `turns` iterations.
    Returns a transcript list.
    """
    transcript = [{"role": "user", "content": initial_prompt[0]["content"]}]
    history = list(initial_prompt)
    for turn in range(turns):
        claude_resp = claude_fn(history)
        transcript.append({"role": "assistant", "from": "claude", "content": claude_resp["content"]})
        history.append(claude_resp)

        if turn < turns - 1:
            chatgpt_resp = chatgpt_fn(history)
            transcript.append({"role": "assistant", "from": "chatgpt", "content": chatgpt_resp["content"]})
            history.append(chatgpt_resp)

    return transcript

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", help="Initial prompt for the bridge")
    parser.add_argument("turns", nargs="?", type=int, default=2, help="Number of Claude↔ChatGPT turns")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode with canned responses")
    args = parser.parse_args()

    if args.mock:
        print("🚀 Running in MOCK mode—no real API calls.")
        claude_fn   = lambda msgs: {"role":"assistant","content":"[Mock Claude reply]"}
        chatgpt_fn  = lambda msgs: {"role":"assistant","content":"[Mock ChatGPT reply]"}
    else:
        if not os.getenv("CLAUDE_API_KEY") or not os.getenv("OPENAI_API_KEY"):
            print("ERROR: set CLAUDE_API_KEY and OPENAI_API_KEY (or use --mock).")
            sys.exit(1)
        claude_fn  = send_to_claude
        chatgpt_fn = send_to_chatgpt

    try:
        transcript = bridge_conversation(
            initial_prompt=[{"role":"user","content":args.prompt}],
            turns=args.turns,
            claude_fn=claude_fn,
            chatgpt_fn=chatgpt_fn
        )
    except HTTPError as e:
        print(f"API error: {e}\nCheck your keys or network.")
        sys.exit(1)

    for msg in transcript:
        speaker = msg["role"]
        src     = msg.get("from","user")
        print(f"[{speaker}/{src}]: {msg['content']}\n")
