# Samaraj Bot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Slack bot that uses a self-hosted Llama 3.2 3B model to deliver sarcastic corrections when tagged on a message.

**Architecture:** Single Python application using `slack_bolt` (Socket Mode) for Slack events and `llama-cpp-python` for in-process LLM inference. Deployed as a Docker container on Railway CPU instance. Model downloaded from HuggingFace at build time.

**Tech Stack:** Python 3.11, slack_bolt, llama-cpp-python, huggingface_hub, Docker

---

## File Structure

```
samaraj/
├── src/
│   ├── prompts.py      # System prompt defining Samaraj's personality
│   ├── model.py         # Llama model loading and inference
│   └── app.py           # Slack event handler and main entrypoint
├── tests/
│   ├── test_prompts.py  # Tests for prompt construction
│   ├── test_model.py    # Tests for model wrapper (mocked inference)
│   └── test_app.py      # Tests for Slack event handling (mocked Slack + model)
├── Dockerfile           # Container definition with model download
├── requirements.txt     # Python dependencies
├── .env.example         # Template for environment variables
└── .gitignore           # Ignore .env, model files, __pycache__
```

---

### Task 1: Project Scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `src/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create `requirements.txt`**

```
slack_bolt==1.21.3
llama-cpp-python==0.3.8
huggingface_hub>=0.27.0
python-dotenv==1.1.0
pytest==8.3.5
```

- [ ] **Step 2: Create `.env.example`**

```
SLACK_BOT_TOKEN=xoxb-your-bot-token
SLACK_APP_TOKEN=xapp-your-app-token
HF_TOKEN=hf_your_huggingface_token
MODEL_PATH=models/llama-3.2-3b-instruct-q4_k_m.gguf
```

- [ ] **Step 3: Create `.gitignore`**

```
.env
__pycache__/
*.pyc
models/
*.gguf
.pytest_cache/
```

- [ ] **Step 4: Create empty `src/__init__.py` and `tests/__init__.py`**

Both files are empty — they just make the directories importable.

- [ ] **Step 5: Install dependencies and verify**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -c "import llama_cpp; import slack_bolt; print('All imports OK')"
```

Expected: `All imports OK`

- [ ] **Step 6: Commit**

```bash
git add requirements.txt .env.example .gitignore src/__init__.py tests/__init__.py
git commit -m "feat: scaffold project with dependencies and config"
```

---

### Task 2: Personality System Prompt

**Files:**
- Create: `src/prompts.py`
- Create: `tests/test_prompts.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_prompts.py`:

```python
from src.prompts import build_prompt


def test_build_prompt_returns_messages_list():
    messages = build_prompt("Python was created in 1995")
    assert isinstance(messages, list)
    assert len(messages) == 2


def test_build_prompt_has_system_message():
    messages = build_prompt("Python was created in 1995")
    system_msg = messages[0]
    assert system_msg["role"] == "system"
    assert "Samaraj" in system_msg["content"]
    assert "sarcastic" in system_msg["content"].lower()


def test_build_prompt_has_user_message():
    messages = build_prompt("Python was created in 1995")
    user_msg = messages[1]
    assert user_msg["role"] == "user"
    assert "Python was created in 1995" in user_msg["content"]


def test_build_prompt_user_message_includes_instruction():
    messages = build_prompt("The earth is flat")
    user_msg = messages[1]
    assert "correct" in user_msg["content"].lower() or "wrong" in user_msg["content"].lower()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/ivanhorvat/Documents/Development/Reactor/SlackLocalTextToImage
python -m pytest tests/test_prompts.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.prompts'`

- [ ] **Step 3: Write `src/prompts.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_prompts.py -v
```

Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/prompts.py tests/test_prompts.py
git commit -m "feat: add Samaraj personality system prompt"
```

---

### Task 3: Model Wrapper

**Files:**
- Create: `src/model.py`
- Create: `tests/test_model.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_model.py`:

```python
from unittest.mock import MagicMock, patch

from src.model import LlamaModel


@patch("src.model.Llama")
def test_load_model_creates_llama_instance(mock_llama_cls):
    model = LlamaModel(model_path="/fake/path.gguf")
    mock_llama_cls.assert_called_once_with(
        model_path="/fake/path.gguf",
        n_ctx=2048,
        verbose=False,
    )


@patch("src.model.Llama")
def test_generate_calls_create_chat_completion(mock_llama_cls):
    mock_llama = MagicMock()
    mock_llama.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "Actually, that's wrong."}}]
    }
    mock_llama_cls.return_value = mock_llama

    model = LlamaModel(model_path="/fake/path.gguf")
    messages = [
        {"role": "system", "content": "You are Samaraj"},
        {"role": "user", "content": "Correct this"},
    ]
    result = model.generate(messages)

    mock_llama.create_chat_completion.assert_called_once_with(
        messages=messages,
        max_tokens=256,
        temperature=0.7,
    )
    assert result == "Actually, that's wrong."


@patch("src.model.Llama")
def test_generate_returns_empty_string_on_no_choices(mock_llama_cls):
    mock_llama = MagicMock()
    mock_llama.create_chat_completion.return_value = {"choices": []}
    mock_llama_cls.return_value = mock_llama

    model = LlamaModel(model_path="/fake/path.gguf")
    result = model.generate([{"role": "user", "content": "test"}])
    assert result == ""
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_model.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.model'`

- [ ] **Step 3: Write `src/model.py`**

```python
from llama_cpp import Llama


class LlamaModel:
    def __init__(self, model_path: str):
        self._llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            verbose=False,
        )

    def generate(self, messages: list[dict]) -> str:
        response = self._llm.create_chat_completion(
            messages=messages,
            max_tokens=256,
            temperature=0.7,
        )
        choices = response.get("choices", [])
        if not choices:
            return ""
        return choices[0]["message"]["content"]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_model.py -v
```

Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/model.py tests/test_model.py
git commit -m "feat: add Llama model wrapper with inference"
```

---

### Task 4: Slack Bot Event Handler

**Files:**
- Create: `src/app.py`
- Create: `tests/test_app.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_app.py`:

```python
from unittest.mock import MagicMock, patch


def _make_event(text="<@U123BOT> ", channel="C123", thread_ts="1234.5678", ts="1234.9999"):
    """Helper to create a Slack app_mention event dict."""
    return {
        "text": text,
        "channel": channel,
        "thread_ts": thread_ts,
        "ts": ts,
        "user": "U999USER",
    }


def _make_say():
    return MagicMock()


def _make_client(parent_text="Python was created in 1995", bot_user_id="U123BOT"):
    client = MagicMock()
    client.conversations_replies.return_value = {
        "messages": [
            {"text": parent_text, "ts": "1234.5678", "user": "U999USER"},
            {"text": "<@U123BOT>", "ts": "1234.9999", "user": "U888TAGGER"},
        ]
    }
    client.auth_test.return_value = {"user_id": bot_user_id}
    return client


@patch("src.app.model")
def test_handle_mention_replies_in_thread(mock_model):
    from src.app import handle_mention

    mock_model.generate.return_value = "Actually, Python was created in 1991."

    event = _make_event()
    say = _make_say()
    client = _make_client()

    handle_mention(event=event, say=say, client=client)

    say.assert_called_once()
    call_kwargs = say.call_args
    assert "1991" in call_kwargs[1]["text"] or "1991" in call_kwargs[0][0]
    assert call_kwargs[1].get("thread_ts") == "1234.5678"


@patch("src.app.model")
def test_handle_mention_top_level_message(mock_model):
    """When tagged in a top-level message (not a thread reply), respond with a quip."""
    from src.app import handle_mention

    event = _make_event(thread_ts=None)
    # Remove thread_ts key entirely to simulate top-level mention
    del event["thread_ts"]
    say = _make_say()
    client = _make_client()

    handle_mention(event=event, say=say, client=client)

    say.assert_called_once()
    call_kwargs = say.call_args
    # Should NOT call model.generate for top-level messages
    mock_model.generate.assert_not_called()


@patch("src.app.model")
def test_handle_mention_ignores_own_message(mock_model):
    """When the parent message is from Samaraj itself, do nothing."""
    from src.app import handle_mention

    client = MagicMock()
    client.conversations_replies.return_value = {
        "messages": [
            {"text": "I already roasted this", "ts": "1234.5678", "user": "U123BOT"},
        ]
    }
    client.auth_test.return_value = {"user_id": "U123BOT"}

    event = _make_event()
    say = _make_say()

    handle_mention(event=event, say=say, client=client)

    mock_model.generate.assert_not_called()
    say.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_app.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.app'`

- [ ] **Step 3: Write `src/app.py`**

```python
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

app = App(token=os.environ.get("SLACK_BOT_TOKEN"))

model: LlamaModel | None = None


def get_parent_message(client, channel: str, thread_ts: str) -> dict | None:
    """Fetch the first (parent) message in a thread."""
    result = client.conversations_replies(channel=channel, ts=thread_ts, limit=1)
    messages = result.get("messages", [])
    return messages[0] if messages else None


@app.event("app_mention")
def handle_mention(event, say, client):
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


def main():
    global model
    model_path = os.environ.get("MODEL_PATH", "models/llama-3.2-3b-instruct-q4_k_m.gguf")
    logger.info(f"Loading model from {model_path}...")
    model = LlamaModel(model_path=model_path)
    logger.info("Model loaded. Starting Slack bot...")

    handler = SocketModeHandler(app, os.environ.get("SLACK_APP_TOKEN"))
    handler.start()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_app.py -v
```

Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/app.py tests/test_app.py
git commit -m "feat: add Slack event handler with mention routing"
```

---

### Task 5: Dockerfile and Deployment Config

**Files:**
- Create: `Dockerfile`

- [ ] **Step 1: Write the `Dockerfile`**

```dockerfile
FROM python:3.11-slim

# Install build dependencies for llama-cpp-python
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the model from HuggingFace at build time
ARG HF_TOKEN
RUN mkdir -p models && \
    huggingface-cli download \
        bartowski/Llama-3.2-3B-Instruct-GGUF \
        Llama-3.2-3B-Instruct-Q4_K_M.gguf \
        --local-dir models \
        --token "$HF_TOKEN"

# Copy application code
COPY src/ src/

ENV MODEL_PATH=models/Llama-3.2-3B-Instruct-Q4_K_M.gguf

CMD ["python", "-m", "src.app"]
```

- [ ] **Step 2: Verify Dockerfile builds locally (optional — requires Docker)**

```bash
docker build --build-arg HF_TOKEN=$HF_TOKEN -t samaraj .
```

Expected: Image builds successfully, model downloads during build.

- [ ] **Step 3: Commit**

```bash
git add Dockerfile
git commit -m "feat: add Dockerfile with model download at build time"
```

---

### Task 6: Railway Deployment Configuration

**Files:**
- Create: `railway.toml`

- [ ] **Step 1: Create `railway.toml`**

```toml
[build]
builder = "dockerfile"
dockerfilePath = "Dockerfile"

[build.args]
HF_TOKEN = "${{HF_TOKEN}}"

[deploy]
startCommand = "python -m src.app"
restartPolicyType = "on_failure"
restartPolicyMaxRetries = 3
```

- [ ] **Step 2: Commit**

```bash
git add railway.toml
git commit -m "feat: add Railway deployment config"
```

---

### Task 7: Local Testing with Downloaded Model

**Files:**
- Create: `scripts/download_model.py`

This script lets you download the model locally for testing without Docker.

- [ ] **Step 1: Create `scripts/download_model.py`**

```python
"""Download the Llama 3.2 3B GGUF model from HuggingFace for local testing."""
import os

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv()


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: Set HF_TOKEN in your .env file")
        print("Get your token at: https://huggingface.co/settings/tokens")
        return

    print("Downloading Llama 3.2 3B Instruct (Q4_K_M)...")
    print("This is ~2 GB and may take a few minutes.\n")

    path = hf_hub_download(
        repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF",
        filename="Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        local_dir="models",
        token=token,
    )

    print(f"\nModel downloaded to: {path}")
    print("You can now run the bot with: python -m src.app")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the download script**

```bash
cp .env.example .env
# Edit .env and fill in your HF_TOKEN
python scripts/download_model.py
```

Expected: Model downloads to `models/` directory (~2 GB).

- [ ] **Step 3: Test the full bot locally**

```bash
# Edit .env and fill in SLACK_BOT_TOKEN and SLACK_APP_TOKEN
python -m src.app
```

Expected: Logs show "Loading model..." then "Model loaded. Starting Slack bot..."
Tag `@Samaraj` on a message in Slack and verify it responds in the thread.

- [ ] **Step 4: Commit**

```bash
git add scripts/download_model.py
git commit -m "feat: add model download script for local testing"
```

---

### Task 8: Run All Tests and Final Commit

- [ ] **Step 1: Run the full test suite**

```bash
python -m pytest tests/ -v
```

Expected: All tests pass (10 total across 3 test files).

- [ ] **Step 2: Final commit with all files verified**

```bash
git status
# Ensure working tree is clean — all files committed
```

---

## Deployment Checklist (Manual Steps)

After all code tasks are complete, follow these steps to deploy:

1. **HuggingFace**: Go to https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct and accept Meta's license agreement
2. **HuggingFace**: Create a token at https://huggingface.co/settings/tokens
3. **Slack**: Follow the step-by-step guide in the design spec to create the Slack app and get tokens
4. **GitHub**: Push the repo to GitHub
5. **Railway**: Create a new project, connect the GitHub repo, add env vars (`SLACK_BOT_TOKEN`, `SLACK_APP_TOKEN`, `HF_TOKEN`), deploy
6. **Test**: Tag `@Samaraj` on a message in Slack and verify it responds
