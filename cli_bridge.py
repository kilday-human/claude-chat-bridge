import os
import argparse
from dotenv import load_dotenv
from chatgpt_wrapper import send_to_chatgpt
from claude_wrapper import send_to_claude

# Load your .env file from CWD
load_dotenv(dotenv_path="./.env")

def bridge_conversation(prompt: str, turns: int = 1, mock: bool = False):
    transcript = [{"role": "user", "content": prompt}]
    for i in range(turns):
        print(f"\n--- Iteration {i+1} ---\n")
        if mock:
            gpt_response = "[MOCK] ChatGPT would reply here"
            claude_response = "[MOCK] Claude would reply here"
        else:
            # call real wrappers
            gpt_response = send_to_chatgpt(transcript)
            claude_response = send_to_claude(transcript)

        # print them in order
        print(f"GPT: {gpt_response}\n")
        print(f"Claude: {claude_response}\n")

        # append to our conversation history
        transcript.append({"role": "assistant", "from": "chatgpt", "content": gpt_response})
        transcript.append({"role": "assistant", "from": "claude",   "content": claude_response})

    return transcript

def main():
    p = argparse.ArgumentParser()
    p.add_argument("prompt",   help="Your starting prompt")
    p.add_argument("turns",    nargs="?", type=int, default=1, help="How many back-and-forths")
    p.add_argument("--mock",   action="store_true",         help="Use canned replies (no API calls)")
    args = p.parse_args()

    bridge_conversation(args.prompt, turns=args.turns, mock=args.mock)

if __name__ == "__main__":
    main()

