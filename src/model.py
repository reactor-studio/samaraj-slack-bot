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
