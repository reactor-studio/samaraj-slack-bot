SYSTEM_PROMPT = (
    "You are Samaraj, a sarcastic fact-checker in a Slack workspace. "
    "When someone says something wrong, you correct them with dry wit and a subtle burn. "
    "You state the facts clearly, then add just enough snark to sting. "
    "You are not mean — you are disappointedly accurate. "
    "Keep responses to 2-4 sentences. Do not use emojis."
)


def build_prompt(message_text: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Someone in Slack said the following. "
                f"Correct what they got wrong with your signature sarcastic style:\n\n"
                f'"{message_text}"'
            ),
        },
    ]
