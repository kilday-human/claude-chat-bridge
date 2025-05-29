import tiktoken

# Adjust model name if needed
GPT_ENCODER = tiktoken.encoding_for_model("gpt-4o-mini")

def count_gpt_tokens(messages):
    """
    Given a list of {"role":..., "content":...} messages,
    return the total token count for a gpt chat-completion.
    """
    total = 0
    for msg in messages:
        # each message: role + content + formatting tokens
        total += len(GPT_ENCODER.encode(msg["role"]))
        total += len(GPT_ENCODER.encode(msg["content"]))
    return total

