import os
import logging

from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from src.model import LlamaModel
from src.prompts import build_prompt

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = App(
    token=os.environ.get("SLACK_BOT_TOKEN", "xoxb-test-token"),
    token_verification_enabled=bool(os.environ.get("SLACK_BOT_TOKEN")),
)

model: LlamaModel | None = None


def get_parent_message(client, channel: str, thread_ts: str) -> dict | None:
    """Fetch the first (parent) message in a thread."""
    result = client.conversations_replies(channel=channel, ts=thread_ts, limit=1)
    messages = result.get("messages", [])
    return messages[0] if messages else None


@app.event("app_mention")
def handle_mention(event, say, client):
    logger.info(f"Received mention from user {event.get('user')} in channel {event.get('channel')}")
    thread_ts = event.get("thread_ts")

    # Tagged in a top-level message, not a reply
    if not thread_ts:
        say(
            text="Tag me on something someone said wrong. I don't just roast the void.",
            thread_ts=event["ts"],
        )
        return

    # Fetch the parent message
    parent = get_parent_message(client, event["channel"], thread_ts)
    if not parent:
        return

    # Don't reply to own messages (prevent loops)
    bot_user_id = client.auth_test()["user_id"]
    if parent.get("user") == bot_user_id:
        return

    parent_text = parent.get("text", "")
    if not parent_text.strip():
        say(
            text="There's nothing here for me to correct. Try harder.",
            thread_ts=thread_ts,
        )
        return

    # Generate sarcastic correction
    messages = build_prompt(parent_text)
    response = model.generate(messages)

    say(text=response, thread_ts=thread_ts)


@app.event("message")
def handle_dm(event, say, client):
    # Only handle DMs (im channel type)
    if event.get("channel_type") != "im":
        return

    # Ignore bot's own messages (prevent loops)
    bot_user_id = client.auth_test()["user_id"]
    if event.get("user") == bot_user_id:
        return
    # Ignore bot_message subtypes
    if event.get("subtype") == "bot_message":
        return

    message_text = event.get("text", "").strip()
    if not message_text:
        return

    logger.info(f"Received DM from user {event.get('user')}: {message_text[:50]}")
    messages = build_prompt(message_text)
    response = model.generate(messages)
    logger.info(f"Response generated, sending reply")

    say(text=response)


def main():
    global model
    model_path = os.environ.get("MODEL_PATH", "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf")
    logger.info(f"Loading model from {model_path}...")
    model = LlamaModel(model_path=model_path)
    logger.info("Model loaded. Starting Slack bot...")

    handler = SocketModeHandler(app, os.environ.get("SLACK_APP_TOKEN"))
    handler.start()


if __name__ == "__main__":
    main()
