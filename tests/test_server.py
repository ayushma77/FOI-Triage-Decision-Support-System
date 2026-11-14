"""Server endpoints, with Claude and ElevenLabs mocked (no network, no keys)."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import server  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "AUDIT", tmp_path / "audit.jsonl")
    return TestClient(server.app)


def test_health_reports_missing_keys(client, monkeypatch):
    monkeypatch.setattr(server, "ANTHROPIC_KEY", "")
    monkeypatch.setattr(server, "ELEVEN_KEY", "")
    h = client.get("/api/health").json()
    assert h["claude"] is False and h["elevenlabs"] is False and h["model"] is None


def test_claude_returns_only_api_safe_blocks(client, monkeypatch):
    monkeypatch.setattr(server, "ANTHROPIC_KEY", "k")
    blocks = [NS(type="text", text="hi", citations=None), NS(type="tool_use", id="t1", name="get_status", input={}, caller=None)]
    monkeypatch.setattr(server, "_claude", NS(messages=NS(create=lambda **kw: NS(stop_reason="tool_use", content=blocks))))
    r = client.post("/api/claude", json={"system": "s", "messages": [{"role": "user", "content": "q"}], "tools": []}).json()
    assert r["content"] == [{"type": "text", "text": "hi"}, {"type": "tool_use", "id": "t1", "name": "get_status", "input": {}}]


def test_claude_without_key_is_503(client, monkeypatch):
    monkeypatch.setattr(server, "ANTHROPIC_KEY", "")
    assert client.post("/api/claude", json={"messages": []}).status_code == 503


class FakeAsyncClient:
    calls = []

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, **kw):
        FakeAsyncClient.calls.append((url, kw))
        if "text-to-speech" in url:
            return NS(status_code=200, content=b"ID3fake-mp3", text="")
        return NS(status_code=200, json=lambda: {"text": "what's left"}, text="")


def test_tts_proxies_to_elevenlabs_with_key_header(client, monkeypatch):
    monkeypatch.setattr(server, "ELEVEN_KEY", "xi")
    monkeypatch.setattr(server.httpx, "AsyncClient", FakeAsyncClient)
    r = client.post("/api/tts", json={"text": "Hello officer"})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/mpeg" and r.content.startswith(b"ID3")
    url, kw = FakeAsyncClient.calls[-1]
    assert "/v1/text-to-speech/" in url and kw["headers"]["xi-api-key"] == "xi" and kw["json"]["text"] == "Hello officer"


def test_stt_returns_transcript(client, monkeypatch):
    monkeypatch.setattr(server, "ELEVEN_KEY", "xi")
    monkeypatch.setattr(server.httpx, "AsyncClient", FakeAsyncClient)
    r = client.post("/api/stt", content=b"\x1aE\xdf\xa3webm", headers={"content-type": "audio/webm"})
    assert r.json() == {"text": "what's left"}
    url, kw = FakeAsyncClient.calls[-1]
    assert url.endswith("/v1/speech-to-text") and kw["data"]["model_id"]


def test_audit_is_appended_and_exported(client):
    client.post("/api/audit", json={"time": "10:00:00", "who": "Officer", "req": "REQ-0001", "what": "Redact", "rec": "REC-1", "detail": "x"})
    client.post("/api/audit", json={"time": "10:01:00", "who": "Jarvis (for officer)", "req": "REQ-0001", "what": "Jarvis opened record", "rec": "REC-2", "detail": ""})
    csv = client.get("/api/audit.csv").text.strip().splitlines()
    assert len(csv) == 3 and "Jarvis opened record" in csv[2]
