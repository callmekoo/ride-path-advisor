"""Talk to an OpenAI-compatible chat API and get a validated structure back.

The provider is whatever the settings point at. The proxy given in the settings
is used for these calls only; proxy variables of the environment are ignored,
so nothing else the service does is routed through it and vice versa.
"""

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import httpx
from pydantic import BaseModel, ValidationError

# Schema keywords that strict structured output rejects on some providers; the
# values are checked on our side after parsing anyway.
_UNSUPPORTED_KEYWORDS = {"default", "title", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"}


class LlmError(Exception):
    """The model could not be reached or did not answer usably; the text is for the user."""


@dataclass(frozen=True)
class LlmSettings:
    base_url: str  # e.g. https://api.example.com/v1
    model: str
    api_key: str
    proxy_url: str | None = None  # http://, https:// or socks5://
    timeout_s: float = 60.0

    @classmethod
    def from_env(cls, env_file: Path | None = Path(".env"), environ: Mapping[str, str] = os.environ) -> Self | None:
        """Settings from LLM_* variables; the environment wins over the file. None if incomplete."""
        values = {**_read_env_file(env_file), **environ}
        base_url, model, api_key = (values.get(name, "").strip() for name in ("LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY"))
        if not (base_url and model and api_key):
            return None
        timeout = float(values.get("LLM_TIMEOUT_S") or cls.timeout_s)
        return cls(base_url, model, api_key, values.get("LLM_PROXY_URL", "").strip() or None, timeout)


def _read_env_file(path: Path | None) -> dict[str, str]:
    if path is None or not Path(path).exists():
        return {}
    values = {}
    for line in Path(path).read_text().splitlines():
        name, separator, value = line.strip().partition("=")
        if separator and name and not name.startswith("#"):
            values[name.strip()] = value.strip().strip("\"'")
    return values


def strict_schema(model: type[BaseModel]) -> dict:
    """JSON schema of `model` in the form strict structured output wants: every field required, nothing extra."""

    def tidy(node: object) -> object:
        if isinstance(node, list):
            return [tidy(item) for item in node]
        if not isinstance(node, dict):
            return node
        cleaned = {key: tidy(value) for key, value in node.items() if key not in _UNSUPPORTED_KEYWORDS}
        if "properties" in node:
            # Property names are data, not keywords: restore any that the filter above dropped.
            cleaned["properties"] = {name: tidy(value) for name, value in node["properties"].items()}
            cleaned["required"] = list(node["properties"])
            cleaned["additionalProperties"] = False
        return cleaned

    return tidy(model.model_json_schema())


def client_options(settings: LlmSettings) -> dict:
    """How the HTTP client for LLM calls is set up."""
    return {
        "base_url": settings.base_url.rstrip("/") + "/",
        "headers": {"Authorization": f"Bearer {settings.api_key}"},
        "proxy": settings.proxy_url,
        "trust_env": False,
        "timeout": settings.timeout_s,
    }


class LlmClient:
    def __init__(self, settings: LlmSettings, transport: httpx.BaseTransport | None = None) -> None:
        self.settings = settings
        options = client_options(settings)
        if transport is not None:  # tests: no network, hence no proxy either
            options = {**options, "proxy": None, "transport": transport}
        self._http = httpx.Client(**options)

    def parse[T: BaseModel](self, system: str, user: str, model: type[T]) -> T:
        """The model's answer to (system, user) as a `model` instance; asks once more if the first answer is unusable."""
        schema = strict_schema(model)
        problem: Exception | None = None
        for _ in range(2):
            try:
                return model.model_validate(self._complete(system, user, model.__name__, schema))
            except (ValidationError, json.JSONDecodeError) as error:
                problem = error
        raise LlmError(f"Модель дважды ответила не по формату: {str(problem)[:200]}")

    def _complete(self, system: str, user: str, name: str, schema: dict) -> dict:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        structured = {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}
        response = self._post({"model": self.settings.model, "messages": messages, "response_format": structured})
        if response.status_code in (400, 422):
            # The provider does not know structured output: describe the format in words instead.
            wording = f"\n\nОтветь одним JSON-объектом без пояснений по схеме:\n{json.dumps(schema, ensure_ascii=False)}"
            messages[0] = {"role": "system", "content": system + wording}
            response = self._post({"model": self.settings.model, "messages": messages})
        if response.status_code >= 400:
            raise LlmError(f"Модель ответила ошибкой {response.status_code}: {response.text[:200]}")
        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (ValueError, LookupError, TypeError):
            raise LlmError(f"Непонятный ответ модели: {response.text[:200]}") from None
        return _extract_json(content or "")

    def _post(self, body: dict) -> httpx.Response:
        try:
            return self._http.post("chat/completions", json=body)
        except httpx.HTTPError as error:
            raise LlmError(f"Модель недоступна: {type(error).__name__}: {error}") from None


def _extract_json(text: str) -> dict:
    """The JSON object in `text`, which may be wrapped in prose or a code fence."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise json.JSONDecodeError("no JSON object", text, 0)
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise json.JSONDecodeError("not an object", text, 0)
    return value
