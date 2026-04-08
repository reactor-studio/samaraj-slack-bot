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
