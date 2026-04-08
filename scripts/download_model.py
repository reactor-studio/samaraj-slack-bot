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
