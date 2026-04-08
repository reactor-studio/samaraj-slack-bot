# Samaraj Bot — Design Spec

## Overview

Samaraj is a Slack bot that delivers sarcastic, factual corrections when tagged. When someone posts something wrong in a channel, another user replies to that message with `@Samaraj`, and the bot generates a dry, witty correction using a self-hosted Llama 3.2 3B model.

The project is primarily a learning exercise in self-hosting open-source LLMs on Railway.

## Architecture

Single Docker container deployed on Railway (CPU instance) containing:

- A Python Slack bot (`slack_bolt`)
- An embedded Llama 3.2 3B model via `llama-cpp-python`
- The GGUF-quantized model file pulled from HuggingFace at Docker build time

```
Slack Workspace
    |
    | (app_mention event via Socket Mode)
    v
+----------------------------------+
|  Railway CPU Instance            |
|  +-----------+    +------------+ |
|  | Slack Bot |<-->| Llama 3.2  | |
|  | (app.py)  |    | 3B (GGUF)  | |
|  +-----------+    +------------+ |
+----------------------------------+
```

There is no separate inference server. The bot loads the model directly in-process using `llama-cpp-python`.

## How It Works

1. A user posts a message in a Slack channel (e.g., "Python was created in 1995")
2. Another user replies to that message with `@Samaraj`
3. Slack sends an `app_mention` event to the bot via Socket Mode (persistent WebSocket connection — no public URL needed)
4. The bot extracts the parent message from the thread
5. The bot constructs a prompt combining Samaraj's personality system prompt + the parent message content
6. `llama-cpp-python` runs inference on the Llama 3.2 3B model
7. The bot posts the generated correction as a reply in the same thread
8. Expected response time: 10-30 seconds on CPU

## Components

### `app.py` — Slack Event Handler
- Initializes `slack_bolt` app in Socket Mode
- Listens for `app_mention` events
- When triggered:
  - Extracts the channel and thread timestamp from the event
  - Fetches the parent message using `conversations.history` or `conversations.replies`
  - Passes the parent message text to the model
  - Posts the response in the thread
- Handles edge cases:
  - Tagged in a top-level message (not a reply) — respond with a quip like "Tag me on something someone said wrong. I don't just roast the void."
  - Parent message is from Samaraj itself — ignore to prevent loops

### `model.py` — Model Loading and Inference
- Loads the GGUF model file at startup using `llama-cpp-python`
- Provides a `generate(system_prompt, user_message) -> str` function
- Configuration: max tokens, temperature (slightly elevated ~0.7 for personality), context window
- Model path configured via environment variable or default path

### `prompts.py` — Personality Definition
- Contains the system prompt that defines Samaraj's personality:
  - Tone: Sarcastic but grounded. Dry wit. States facts then delivers a subtle burn.
  - Behavior: Correct factual errors, point out logical flaws, call out contradictions
  - Constraints: Keep responses concise (2-4 sentences). Don't be cruel, just pointedly accurate.
- Example system prompt direction:
  > You are Samaraj, a sarcastic fact-checker in a Slack workspace. When someone says something wrong, you correct them with dry wit and a subtle burn. You state the facts clearly, then add just enough snark to sting. You are not mean — you are disappointedly accurate. Keep responses to 2-4 sentences.

### `Dockerfile`
- Base image: `python:3.11-slim`
- Installs system dependencies for `llama-cpp-python` (CMake, build tools)
- Installs Python dependencies from `requirements.txt`
- Downloads the GGUF model from HuggingFace using `huggingface-cli` at build time
- Sets the entrypoint to run `app.py`

### `requirements.txt`
- `slack_bolt` — Slack SDK with Socket Mode support
- `llama-cpp-python` — Python bindings for llama.cpp (CPU inference)
- `huggingface_hub` — For downloading the model

## Model Details

- **Model**: Llama 3.2 3B Instruct
- **Format**: GGUF (quantized Q4_K_M — good balance of quality and size for CPU)
- **Source**: HuggingFace (e.g., `bartowski/Llama-3.2-3B-Instruct-GGUF` or similar community quantization)
- **Size**: ~2 GB (Q4_K_M quantization)
- **Auth**: Requires HuggingFace token (`HF_TOKEN`) since Llama models require accepting Meta's license

## Slack App Setup (Step-by-Step)

1. Go to https://api.slack.com/apps
2. Click **"Create New App"** > **"From scratch"**
3. Name it **"Samaraj"**, select your workspace
4. In the left sidebar, go to **"Socket Mode"** > Enable it > Create an app-level token (name it "samaraj-socket", scope: `connections:write`) > Save the token as `SLACK_APP_TOKEN`
5. Go to **"OAuth & Permissions"** > Add these Bot Token Scopes:
   - `app_mentions:read` — to receive @Samaraj mentions
   - `chat:write` — to post responses
   - `channels:history` — to read the parent message being replied to
   - `groups:history` — same but for private channels (optional)
6. Go to **"Event Subscriptions"** > Enable Events > Subscribe to bot event: `app_mention`
7. Go to **"App Home"** > Set the bot display name to "Samaraj"
8. Click **"Install to Workspace"** at the top of OAuth & Permissions
9. Copy the **Bot User OAuth Token** — this is your `SLACK_BOT_TOKEN`
10. Invite `@Samaraj` to the channels where you want it active

## Railway Deployment

1. Push the code to a GitHub repository
2. Go to https://railway.app and create a new project
3. Select **"Deploy from GitHub Repo"** and connect the repo
4. Railway auto-detects the Dockerfile
5. Add environment variables in Railway dashboard:
   - `SLACK_BOT_TOKEN` — from Slack app setup step 9
   - `SLACK_APP_TOKEN` — from Slack app setup step 4
   - `HF_TOKEN` — from https://huggingface.co/settings/tokens
6. Deploy — Railway builds the Docker image (including model download) and starts the service
7. Uses a CPU instance (no GPU needed)

## Environment Variables

| Variable | Source | Purpose |
|----------|--------|---------|
| `SLACK_BOT_TOKEN` | Slack App > OAuth & Permissions | Authenticate bot API calls |
| `SLACK_APP_TOKEN` | Slack App > Socket Mode | Establish WebSocket connection |
| `HF_TOKEN` | HuggingFace > Settings > Tokens | Download gated Llama model |

## Edge Cases

- **Tagged on a top-level message (not a reply)**: Respond with a quip asking to be tagged on someone's message
- **Tagged on Samaraj's own message**: Ignore (prevent infinite loops)
- **Parent message is empty or just an image/file**: Respond that there's nothing to correct
- **Model takes too long**: Set a timeout; if exceeded, reply with "Even I need a moment to process that level of wrongness"

## Cost Estimate

- **Railway CPU instance**: ~$5-20/month depending on usage (billed per execution second on starter plans)
- **HuggingFace**: Free (just need an account and accept Meta's Llama license)
- **Slack**: Free (bot features are included in free Slack tier)

## Out of Scope

- No web dashboard or admin panel
- No message history/logging
- No multi-workspace support
- No conversation memory (each correction is independent)
- No image/file analysis — text messages only
