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
