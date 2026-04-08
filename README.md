# Samaraj

A sarcastic Slack bot that corrects people when they're wrong. Tag `@Samaraj` on a message and it will deliver a factual correction with dry wit and a subtle burn.

Powered by a self-hosted Llama 3.2 3B model — no third-party AI APIs involved.

## How it works

1. Someone posts something wrong in a channel
2. You reply to that message with `@Samaraj`
3. Samaraj reads the message, generates a snarky correction, and replies in the thread

You can also DM Samaraj directly — just send a message and it'll correct you.

## Example

> **User:** Python was created in 1995
>
> **Samaraj:** Python was actually created in 1991 by Guido van Rossum. I'm not sure what decade they were counting on, but it's not 1995.

## Tech stack

- **Python** with [slack_bolt](https://slack.dev/bolt-python/) (Socket Mode)
- **Llama 3.2 3B** via [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) (CPU inference)
- **GGUF** quantized model (Q4_K_M, ~2 GB) from HuggingFace
- **Docker** container deployed on [Railway](https://railway.app)

## Setup

### Prerequisites

- Python 3.11+
- A [HuggingFace](https://huggingface.co) account (free)
- A Slack workspace where you can install apps
- A [Railway](https://railway.app) account (for deployment)

### 1. Clone and install

```bash
git clone <your-repo-url>
cd samaraj
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Accept the Llama license

Go to [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) on HuggingFace and accept Meta's license agreement.

### 3. Create a HuggingFace token

Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and create a **Read** token.

### 4. Create the Slack App

1. Go to [api.slack.com/apps](https://api.slack.com/apps) > **Create New App** > **From scratch**
2. Name it **"Samaraj"**, select your workspace
3. **Socket Mode** > Enable > Create app-level token (name: `samaraj-socket`, scope: `connections:write`) > Save the token as `SLACK_APP_TOKEN`
4. **OAuth & Permissions** > Add Bot Token Scopes:
   - `app_mentions:read`
   - `chat:write`
   - `channels:history`
   - `im:history`
5. **Event Subscriptions** > Enable > Subscribe to bot events: `app_mention`, `message.im`
6. **App Home** > Enable **Messages Tab** > Check **"Allow users to send Slash commands and messages from the messages tab"**
7. **Install to Workspace** > Copy the **Bot User OAuth Token** as `SLACK_BOT_TOKEN`
8. Invite `@Samaraj` to channels: `/invite @Samaraj`

### 5. Configure environment

```bash
cp .env.example .env
```

Fill in your `.env`:

```
SLACK_BOT_TOKEN=xoxb-your-bot-token
SLACK_APP_TOKEN=xapp-your-app-token
HF_TOKEN=hf_your_huggingface_token
MODEL_PATH=models/Llama-3.2-3B-Instruct-Q4_K_M.gguf
```

### 6. Download the model

```bash
python scripts/download_model.py
```

This downloads ~2 GB.

### 7. Run locally

```bash
python -m src.app
```

## Deploy to Railway

1. Push the repo to GitHub
2. Go to [railway.app](https://railway.app) > **New Project** > **Deploy from GitHub Repo**
3. Add environment variables in the Railway dashboard:
   - `SLACK_BOT_TOKEN`
   - `SLACK_APP_TOKEN`
   - `HF_TOKEN`
4. Deploy — Railway builds the Docker image and downloads the model during build

## Local testing without Slack

Test Samaraj's responses without the Slack connection:

```bash
python scripts/test_locally.py "The Great Wall of China is visible from space"
```

## Project structure

```
├── src/
│   ├── app.py           # Slack event handler (main entrypoint)
│   ├── model.py         # Llama model wrapper
│   └── prompts.py       # Samaraj's personality prompt
├── tests/               # Unit tests
├── scripts/
│   ├── download_model.py  # Download model from HuggingFace
│   └── test_locally.py    # Test without Slack
├── Dockerfile           # Container with model baked in
└── railway.toml         # Railway deployment config
```

## Running tests

```bash
python -m pytest tests/ -v
```
