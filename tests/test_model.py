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
