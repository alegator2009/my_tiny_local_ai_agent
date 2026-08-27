from app.services.orchestrator import _provider_stream_events


def test_provider_stream_events_keeps_reasoning_separate_from_content():
    payload = {
        "choices": [
            {
                "delta": {
                    "reasoning_content": "I should check this first.",
                    "content": "Here is the answer.",
                }
            }
        ]
    }

    assert _provider_stream_events(payload) == [
        ("reasoning", "I should check this first."),
        ("content", "Here is the answer."),
    ]


def test_provider_stream_events_supports_newer_reasoning_field():
    payload = {"choices": [{"delta": {"reasoning": "Thinking."}}]}

    assert _provider_stream_events(payload) == [("reasoning", "Thinking.")]


def test_provider_stream_events_ignores_empty_non_text_fields():
    payload = {"choices": [{"delta": {"reasoning_content": "", "content": None}}]}

    assert _provider_stream_events(payload) == []
