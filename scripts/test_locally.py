"""Test Samaraj's response locally without Slack."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

load_dotenv()

from src.model import LlamaModel
from src.prompts import build_prompt


def main():
    if len(sys.argv) < 2:
        print('Usage: python scripts/test_locally.py "Python was created in 1995"')
        return

    message = " ".join(sys.argv[1:])
    model_path = os.environ.get("MODEL_PATH", "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf")

    print(f"Loading model from {model_path}...")
    model = LlamaModel(model_path=model_path)
    print("Model loaded.\n")

    print(f'Message: "{message}"')
    print("Samaraj is thinking...\n")

    messages = build_prompt(message)
    response = model.generate(messages)

    print(f"Samaraj: {response}")


if __name__ == "__main__":
    main()
