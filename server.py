"""FOI Triage Desk: local server with the Jarvis voice assistant.

    cp .env.example .env        # add your keys (both optional)
    python run_all.py           # builds outputs/app.html
    python server.py            # http://localhost:8000

Keys stay on this server; the browser never sees them.
  ANTHROPIC_API_KEY   Jarvis's reasoning (Claude, with tool use)
  ELEVENLABS_API_KEY  Jarvis's voice (text-to-speech) and listening (speech-to-text)
Without keys the app still works: Jarvis falls back to built-in command
handling, the browser's own voice, and the browser's speech recognition.

Jarvis can navigate, search, explain and draft. It has no tool that redacts,
releases or signs off: those stay with the officer (see ui/template.html).
"""

import csv
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from foi_triage.workflow import Workflow, WorkflowError
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"


def load_env(path=ROOT / ".env"):
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env()
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")
ELEVEN_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
# Default: "George", a stock ElevenLabs voice (warm, British). Any voice ID from your
# ElevenLabs voice library works. Do not clone a real person's voice without consent.
ELEVEN_VOICE = os.environ.get("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")
ELEVEN_TTS_MODEL = os.environ.get("ELEVENLABS_TTS_MODEL", "eleven_flash_v2_5")
ELEVEN_STT_MODEL = os.environ.get("ELEVENLABS_STT_MODEL", "scribe_v1")

app = FastAPI(title="FOI Triage Desk")
_claude = None
_wf = None


def workflow():
    """Loads the pipeline output once and the shared case store (outputs/workflow.json)."""
    global _wf
    if _wf is None:
        from foi_triage.pipeline import DATA, REAL, load_jsonl
        from foi_triage.retrieval import Retriever
        data = json.loads((OUT / "app_data.json").read_text())
        sources = {"nt": DATA / "records.jsonl", "enron": REAL / "enron_records.jsonl"}
        cols = {}
        for col, c in data["collections"].items():
            raw = load_jsonl(sources[col])
            cols[col] = {
                "records": {r["record_id"]: {"subject": r["subject"], "date": r["date"], "body": r["body"]} for r in raw},
                "flags": {rid: [[f["i"], f["c"], f["p"], f["s"], f["e"]] for f in rec["f"]] for rid, rec in c["records"].items()},
                "packets": {p["id"]: p for p in c["packets"]},
                "retriever": Retriever(raw),
            }
        _wf = Workflow(OUT / "workflow.json", cols, data["categories"], data["defaults"])
    return _wf


def _user(req: Request):
    token = req.headers.get("authorization", "").removeprefix("Bearer ").strip()
    try:
        return workflow().user_for(token)
    except WorkflowError as e:
        raise HTTPException(e.code, e.message)


def _wf_call(fn, *a):
    try:
        return fn(*a)
    except WorkflowError as e:
        raise HTTPException(e.code, e.message)


def claude():
    global _claude
    if _claude is None:
        import anthropic
        _claude = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    return _claude


@app.get("/")
def index():
    page = OUT / "app.html"
    if not page.exists():
        raise HTTPException(500, "Run `python run_all.py` first to build outputs/app.html")
    return FileResponse(page, media_type="text/html")


@app.get("/api/health")
def health():
    return {"workflow": True, "claude": bool(ANTHROPIC_KEY), "model": CLAUDE_MODEL if ANTHROPIC_KEY else None,
            "elevenlabs": bool(ELEVEN_KEY), "voice_id": ELEVEN_VOICE if ELEVEN_KEY else None}


@app.post("/api/login")
async def login(req: Request):
    b = await req.json()
    token, user = _wf_call(workflow().login, b.get("username"), b.get("password"))
    return {"token": token, "user": user}


@app.get("/api/cases")
def list_cases(req: Request, since: int = -1):
    u = _user(req)
    wf = workflow()
    if since == wf.state["version"]:
        return {"version": since, "unchanged": True}
    return wf.list_for(u)


@app.post("/api/cases")
async def create_case(req: Request):
    u = _user(req)
    b = await req.json()
    c = _wf_call(workflow().create_case, u, b.get("text"), b.get("col", "nt"))
    return {"version": workflow().state["version"], "case": workflow().view_for(u, c)}


@app.post("/api/cases/{cid}/act")
async def act(cid: str, req: Request):
    u = _user(req)
    b = await req.json()
    c = _wf_call(workflow().act, cid, u, b.get("type"), b.get("payload") or {})
    return {"version": workflow().state["version"], "case": workflow().view_for(u, c)}


@app.post("/api/claude")
async def claude_turn(req: Request):
    """One round of Jarvis's tool loop. The browser runs the tools (they act on
    the page) and calls again with the results, so the server stays stateless."""
    if not ANTHROPIC_KEY:
        raise HTTPException(503, "ANTHROPIC_API_KEY not set")
    body = await req.json()
    try:
        msg = claude().messages.create(
            model=CLAUDE_MODEL, max_tokens=int(body.get("max_tokens", 900)),
            system=body.get("system", ""), messages=body["messages"], tools=body.get("tools", []),
        )
    except Exception as e:  # surface API errors to the UI as text
        raise HTTPException(502, f"Claude API error: {e}")
    content = []
    for b in msg.content:  # send back only the fields the API accepts on the next turn
        if b.type == "text":
            content.append({"type": "text", "text": b.text})
        elif b.type == "tool_use":
            content.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
    return {"stop_reason": msg.stop_reason, "content": content}


@app.post("/api/tts")
async def tts(req: Request):
    if not ELEVEN_KEY:
        raise HTTPException(503, "ELEVENLABS_API_KEY not set")
    text = (await req.json()).get("text", "")[:2500]
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE}?output_format=mp3_44100_128"
    payload = {"text": text, "model_id": ELEVEN_TTS_MODEL,
               "voice_settings": {"stability": 0.5, "similarity_boost": 0.75, "style": 0.15}}
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(url, json=payload, headers={"xi-api-key": ELEVEN_KEY, "accept": "audio/mpeg"})
    if r.status_code != 200:
        raise HTTPException(502, f"ElevenLabs TTS error {r.status_code}: {r.text[:300]}")
    return Response(r.content, media_type="audio/mpeg")


@app.post("/api/stt")
async def stt(req: Request):
    if not ELEVEN_KEY:
        raise HTTPException(503, "ELEVENLABS_API_KEY not set")
    audio = await req.body()
    ctype = req.headers.get("content-type", "audio/webm")
    files = {"file": ("speech.webm", audio, ctype)}
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post("https://api.elevenlabs.io/v1/speech-to-text", headers={"xi-api-key": ELEVEN_KEY},
                              data={"model_id": ELEVEN_STT_MODEL, "language_code": "en"}, files=files)
    if r.status_code != 200:
        raise HTTPException(502, f"ElevenLabs STT error {r.status_code}: {r.text[:300]}")
    return {"text": r.json().get("text", "")}


AUDIT = OUT / "audit_log.jsonl"


@app.post("/api/audit")
async def audit(req: Request):
    """Server-side, append-only copy of every officer action."""
    row = await req.json()
    row["server_time"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(AUDIT, "a") as f:
        f.write(json.dumps(row) + "\n")
    return {"ok": True}


@app.get("/api/audit.csv")
def audit_csv():
    rows = [json.loads(l) for l in AUDIT.read_text().splitlines()] if AUDIT.exists() else []
    buf = io.StringIO()
    keys = ["server_time", "time", "who", "req", "what", "rec", "detail"]
    w = csv.DictWriter(buf, fieldnames=keys, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=foi_audit_log.csv"})


if __name__ == "__main__":
    import uvicorn
    print(f"Claude: {'on (' + CLAUDE_MODEL + ')' if ANTHROPIC_KEY else 'off (offline Jarvis)'} | "
          f"ElevenLabs: {'on' if ELEVEN_KEY else 'off (browser voice)'}")
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))
