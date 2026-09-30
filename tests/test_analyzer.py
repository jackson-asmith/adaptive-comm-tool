from types import SimpleNamespace

import pytest
from conftest import FakeClient, fallback_block, full_payload, reaction, text_block

from adaptive_comm import analyzer as analyzer_module
from adaptive_comm.analyzer import (
    AnalysisError,
    Analyzer,
    MissingCredentialsError,
    build_schema,
    has_credentials,
    response_text,
)


def test_schema_pins_persona_names(personas):
    schema = build_schema(personas)
    enum = schema["properties"]["reactions"]["items"]["properties"]["persona"]["enum"]
    assert enum == [p.name for p in personas]


def test_analyze_orders_and_clamps(personas):
    names = [p.name for p in personas]
    reactions = [reaction(names[2]), reaction(names[0], 14), reaction(names[1], -3)]
    payload = {"tone_tags": ["terse"], "reactions": reactions}
    client = FakeClient(payload)

    result = Analyzer(personas, client=client).analyze("Re-run the tests.")

    assert [r.persona for r in result.reactions] == names
    assert [r.rapport_score for r in result.reactions] == [10, 0, 7]
    call = client.calls[0]
    assert "Re-run the tests." in call["messages"][0]["content"]
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert call["fallbacks"] == "default"


def test_missing_persona_is_an_error(personas):
    client = FakeClient({"tone_tags": [], "reactions": [reaction(personas[0].name)]})
    with pytest.raises(AnalysisError, match="missing personas"):
        Analyzer(personas, client=client).analyze("hi")


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_bad_stop_reasons(personas, stop_reason):
    with pytest.raises(AnalysisError):
        Analyzer(personas, client=FakeClient({}, stop_reason=stop_reason)).analyze("hi")


def test_no_text_in_response(personas):
    with pytest.raises(AnalysisError, match="no text"):
        Analyzer(personas, client=FakeClient(content=[])).analyze("hi")


@pytest.mark.parametrize("text", ["not json", '{"tone_tags": []}', '["a list"]'])
def test_unparseable_response(personas, text):
    with pytest.raises(AnalysisError, match="could not parse"):
        Analyzer(personas, client=FakeClient(content=[text_block(text)])).analyze("hi")


def test_fallback_response_uses_fallback_models_answer(personas):
    import json

    content = [fallback_block(), text_block(json.dumps(full_payload(tone_tags=["from fallback"])))]
    result = Analyzer(personas, client=FakeClient(content=content)).analyze("hi")
    assert result.tone_tags == ["from fallback"]


def test_response_text_ignores_anything_before_the_last_fallback():
    content = [text_block("partial"), fallback_block(), text_block("final"), SimpleNamespace(type="thinking")]
    assert response_text(content) == "final"
    assert response_text([text_block("a"), text_block("b")]) == "ab"
    assert response_text([fallback_block()]) is None


def test_requires_personas():
    with pytest.raises(ValueError):
        Analyzer([], client=FakeClient())


@pytest.mark.parametrize(
    "attrs, expected",
    [
        ({"api_key": "k", "auth_token": None, "credentials": None}, True),
        ({"api_key": None, "auth_token": "t", "credentials": None}, True),
        ({"api_key": None, "auth_token": None, "credentials": object()}, True),
        ({"api_key": None, "auth_token": None, "credentials": None}, False),
    ],
)
def test_has_credentials(attrs, expected):
    assert has_credentials(SimpleNamespace(**attrs)) is expected


def test_missing_credentials_detected_when_creating_client(personas, monkeypatch):
    no_creds = SimpleNamespace(api_key=None, auth_token=None, credentials=None)
    monkeypatch.setattr(analyzer_module.anthropic, "Anthropic", lambda: no_creds)
    with pytest.raises(MissingCredentialsError):
        Analyzer(personas)


@pytest.fixture
def no_anthropic_config(monkeypatch, tmp_path):
    """An environment with no API key, token, profile, or federation settings."""
    for var in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_PROFILE",
        "ANTHROPIC_CONFIG_DIR",
        "ANTHROPIC_FEDERATION_RULE_ID",
        "ANTHROPIC_IDENTITY_TOKEN",
        "ANTHROPIC_IDENTITY_TOKEN_FILE",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    return tmp_path


def test_real_sdk_client_with_nothing_configured(personas, no_anthropic_config):
    """Guards against SDK changes: with nothing configured, the real client reports no credentials."""
    with pytest.raises(MissingCredentialsError, match="no Anthropic credentials"):
        Analyzer(personas)


def test_real_sdk_client_with_missing_profile(personas, no_anthropic_config, monkeypatch):
    """The SDK raises CredentialsError itself when a configured profile doesn't exist."""
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(no_anthropic_config))
    with pytest.raises(MissingCredentialsError, match="Config file not found"):
        Analyzer(personas)


def test_real_sdk_client_with_api_key(personas, no_anthropic_config, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-real")
    assert Analyzer(personas).client.api_key == "sk-ant-test-not-real"
