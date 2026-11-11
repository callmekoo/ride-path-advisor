import json

import httpx
import pytest
from pydantic import BaseModel, Field

from rpa.nlu.client import LlmClient, LlmError, LlmSettings, client_options, strict_schema

SETTINGS = LlmSettings("https://llm.example/v1", "some-model", "secret")


class Wish(BaseModel):
    place: str | None = None
    km: float | None = Field(default=None, ge=0)
    tags: list[str] = []


def answer(content: str | None, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json={"choices": [{"message": {"role": "assistant", "content": content}}]})


def client_with(*responses: httpx.Response) -> tuple[LlmClient, list[httpx.Request]]:
    """A client whose server gives `responses` in turn; also the requests it received."""
    queue, seen = list(responses), []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return queue.pop(0)

    return LlmClient(SETTINGS, transport=httpx.MockTransport(handler)), seen


def test_request_carries_key_model_messages_and_schema():
    client, seen = client_with(answer('{"place": "Ревда", "km": 45, "tags": []}'))

    assert client.parse("system text", "user text", Wish) == Wish(place="Ревда", km=45)
    (request,) = seen
    assert str(request.url) == "https://llm.example/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer secret"
    body = json.loads(request.content)
    assert body["model"] == "some-model"
    assert body["messages"] == [
        {"role": "system", "content": "system text"},
        {"role": "user", "content": "user text"},
    ]
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["response_format"]["json_schema"]["name"] == "Wish"


def test_strict_schema_requires_every_field_and_drops_what_providers_reject():
    schema = strict_schema(Wish)

    assert schema["required"] == ["place", "km", "tags"] and schema["additionalProperties"] is False
    assert set(schema["properties"]) == {"place", "km", "tags"}
    text = json.dumps(schema)
    assert '"default"' not in text and '"minimum"' not in text and '"title"' not in text


def test_provider_without_structured_output_gets_the_format_in_words():
    client, seen = client_with(answer(None, status=400), answer('Вот: ```json\n{"place": "Ревда"}\n```'))

    assert client.parse("system text", "user text", Wish) == Wish(place="Ревда")
    second = json.loads(seen[1].content)
    assert "response_format" not in second
    assert second["messages"][0]["content"].startswith("system text") and '"place"' in second["messages"][0]["content"]


def test_unusable_answer_is_asked_for_once_more():
    client, seen = client_with(answer("не знаю"), answer('{"km": 30}'))
    assert client.parse("s", "u", Wish) == Wish(km=30)
    assert len(seen) == 2


def test_two_unusable_answers_are_an_error():
    client, _ = client_with(answer('{"km": -5}'), answer('{"km": "много"}'))
    with pytest.raises(LlmError, match="не по формату"):
        client.parse("s", "u", Wish)


def test_refused_key_is_an_error_with_the_status():
    client, _ = client_with(httpx.Response(401, text="bad key"))
    with pytest.raises(LlmError, match="401"):
        client.parse("s", "u", Wish)


def test_garbage_and_network_failures_are_errors():
    client, _ = client_with(httpx.Response(200, text="<html>"))
    with pytest.raises(LlmError, match="Непонятный"):
        client.parse("s", "u", Wish)

    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    with pytest.raises(LlmError, match="недоступна"):
        LlmClient(SETTINGS, transport=httpx.MockTransport(unreachable)).parse("s", "u", Wish)


def test_only_the_configured_proxy_is_used(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://ambient-proxy:3128")

    options = client_options(LlmSettings("https://llm.example/v1/", "m", "k", proxy_url="socks5://127.0.0.1:1080"))
    assert options["proxy"] == "socks5://127.0.0.1:1080" and options["trust_env"] is False
    assert options["base_url"] == "https://llm.example/v1/"
    assert client_options(SETTINGS)["proxy"] is None
    # Each kind of proxy URL is accepted by the HTTP library.
    for proxy in ("socks5://127.0.0.1:1080", "http://user:password@proxy.example:3128"):
        LlmClient(LlmSettings("https://llm.example/v1", "m", "k", proxy_url=proxy))


def test_settings_come_from_the_file_and_the_environment_wins(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        '# provider\nLLM_BASE_URL=https://file.example/v1\nLLM_MODEL="file-model"\nLLM_API_KEY=file-key\n'
        "LLM_PROXY_URL=socks5://127.0.0.1:1080\n\nnot a setting\n"
    )

    assert LlmSettings.from_env(env_file, {}) == LlmSettings(
        "https://file.example/v1", "file-model", "file-key", "socks5://127.0.0.1:1080"
    )
    mixed = LlmSettings.from_env(env_file, {"LLM_MODEL": "env-model", "LLM_PROXY_URL": "", "LLM_TIMEOUT_S": "20"})
    assert (mixed.model, mixed.proxy_url, mixed.timeout_s) == ("env-model", None, 20.0)


def test_incomplete_settings_mean_no_llm(tmp_path):
    assert LlmSettings.from_env(tmp_path / "missing.env", {}) is None
    assert LlmSettings.from_env(None, {"LLM_BASE_URL": "https://x/v1", "LLM_MODEL": "m"}) is None
