"""
server/app.py
=============
FastAPI tabanli web sohbet sunucusu. Modeli yukler ve token token (streaming)
cevap veren bir /api/chat ucu sunar. Tarayicidan kullanilan arayuz: static/index.html

Kullanim:
    python -m server.app --ckpt checkpoints/ckpt.pt --device cuda
    # sonra tarayicidan: http://127.0.0.1:8000 veya http://SUNUCU_IP:8000
"""
from __future__ import annotations
import json
import os
import re
import hashlib
import secrets
import threading
import time
import urllib.parse
import uuid

from fastapi import FastAPI, Body, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import torch

from generate import load_model, chat_stream
from tokenizer import load_tokenizer
from chat_template import DEFAULT_SYSTEM
from online_learn import OnlineLearner
import web_learn
import web_lookup
import quick_intents
from server.node_registry import NodeRegistry

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(HERE, "static")
ROOT_DIR = os.path.dirname(HERE)
API_TOKEN = os.environ.get("YERELLM_API_TOKEN", "").strip()
CHAT_BACKEND = os.environ.get("YERELLM_CHAT_BACKEND", "local").strip().lower()
CHAT_QUICK_MODE = os.environ.get("YERELLM_CHAT_QUICK_MODE", "model_first").strip().lower()
CHAT_FALLBACK_MODE = os.environ.get("YERELLM_CHAT_FALLBACK_MODE", "quality_message").strip().lower()
CHAT_WORKER_TIMEOUT_SEC = int(os.environ.get("YERELLM_CHAT_WORKER_TIMEOUT_SEC", "180"))
CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "YERELLM_CORS_ORIGINS",
        "http://127.0.0.1:8000,http://localhost:8000",
    ).split(",")
    if origin.strip()
]

app = FastAPI(title="yerelLLM")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-YerelLM-Token"],
)
STATE: dict = {}
REGISTRY = NodeRegistry(os.environ.get("YERELLM_REGISTRY_DB", os.path.join(ROOT_DIR, "state", "local_nodes.db")))

# ---- Web'den kendi kendine ogrenme (otomatik/zamanli) --------------------
WEB: dict = {"running": False, "last_run": None, "last_error": None,
             "config_path": os.environ.get("YERELLM_WEB_CONFIG", "web_sources.json")}
DEFAULT_WEB_CONFIG = {
    "enabled": False, "interval_minutes": 60, "max_pages_per_run": 4,
    "max_depth": 0, "request_delay_sec": 2.0, "respect_robots": True,
    "qa_steps": 6, "doc_passes": 1, "study_docs": False, "auto_save": True,
    "sources": [],
}

PATCH_FILE_ALLOWLIST = {
    "chat_template.py",
    "config.py",
    "generate.py",
    "quick_intents.py",
    "web_search.py",
    "train.py",
    "data/__init__.py",
    "data/make_chat_tokens.py",
    "data/prepare_data.py",
    "model/__init__.py",
    "model/gpt.py",
    "tokenizer/__init__.py",
    "tokenizer/tokenizer.json",
    "worker/__init__.py",
    "worker/local_node.py",
    "server/__init__.py",
    "server/app.py",
    "server/node_registry.py",
    "server/static/index.html",
    "server/static/cluster.html",
    "scripts/lan_train.sh",
    "scripts/lan_train.ps1",
    "scripts/system_check.py",
    "scripts/vram_probe.py",
}


def _safe_patch_file(rel_path: str) -> str:
    rel = os.path.normpath(rel_path).replace("\\", "/").lstrip("/")
    if rel.startswith("../") or rel not in PATCH_FILE_ALLOWLIST:
        raise HTTPException(status_code=404, detail="patch dosyasi izinli degil")
    path = os.path.realpath(os.path.join(ROOT_DIR, rel))
    root = os.path.realpath(ROOT_DIR)
    if path != root and not path.startswith(root + os.sep):
        raise HTTPException(status_code=404, detail="gecersiz patch yolu")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="patch dosyasi yok")
    return path


def _request_token(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.headers.get("x-yerellm-token", "").strip()


def _require_api_token(request: Request):
    if not API_TOKEN:
        return
    if not secrets.compare_digest(_request_token(request), API_TOKEN):
        raise HTTPException(status_code=401, detail="API token gerekli")


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_bool(value, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() not in {"0", "false", "no", "off", "hayir", "hayır"}


def _payload_int(value, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _patch_manifest() -> list[dict]:
    files = []
    for rel in sorted(PATCH_FILE_ALLOWLIST):
        path = os.path.join(ROOT_DIR, rel)
        if not os.path.isfile(path):
            continue
        stat = os.stat(path)
        files.append({
            "path": rel,
            "url": f"/patches/localllm/{rel}",
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "sha256": _sha256(path),
        })
    return files


def _eligible_sync_nodes(node_ids: list[str] | None = None) -> tuple[list[dict], list[dict]]:
    nodes = REGISTRY.list_nodes()
    selected_ids = None
    if node_ids is not None:
        selected_ids = {str(node_id).strip() for node_id in node_ids if str(node_id).strip()}
        if not selected_ids:
            return [], []
    selected = [
        node for node in nodes
        if node["status"] == "online" and (selected_ids is None or node["node_id"] in selected_ids)
    ]
    eligible = []
    waiting = []
    for node in selected:
        caps = node.get("capabilities", {})
        if caps.get("allow_training_jobs") or caps.get("allow_remote_commands"):
            eligible.append(node)
        else:
            waiting.append(node)
    return eligible, waiting


def _sync_summary() -> dict:
    nodes = REGISTRY.list_nodes()
    commands = REGISTRY.list_commands(limit=50)
    online = [node for node in nodes if node["status"] == "online"]
    sync_ready = [
        node for node in online
        if node.get("capabilities", {}).get("allow_training_jobs")
        or node.get("capabilities", {}).get("allow_remote_commands")
    ]
    queued = [cmd for cmd in commands if cmd.get("status") in {"queued", "running"}]
    last_done = next((cmd for cmd in commands if cmd.get("type") == "sync_patch" and cmd.get("status") == "done"), None)
    return {
        "target_devices": 3,
        "registered": len(nodes),
        "online": len(online),
        "sync_ready": len(sync_ready),
        "pending_commands": len(queued),
        "last_sync": last_done,
    }


def load_web_config() -> dict:
    p = WEB["config_path"]
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return {**DEFAULT_WEB_CONFIG, **json.load(f)}
        except Exception as e:
            WEB["last_error"] = f"config okunamadi: {e}"
    return dict(DEFAULT_WEB_CONFIG)


def save_web_config(cfg: dict):
    with open(WEB["config_path"], "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def run_web_study(cfg: dict | None = None) -> dict | None:
    """Whitelist'teki kaynaklari gezip CANLI modeli egitir (dokuman + S/C)."""
    if WEB["running"] or not STATE.get("ready"):
        return None
    cfg = cfg or load_web_config()
    WEB["running"] = True
    total = {"pages": 0, "docs_studied": 0, "qa_taught": 0}
    try:
        learner = STATE["learner"]
        fetcher = web_learn.Fetcher(delay=float(cfg["request_delay_sec"]),
                                    respect_robots=bool(cfg["respect_robots"]))
        for src in cfg.get("sources", []):
            seeds = ([src["seed"]] if isinstance(src.get("seed"), str)
                     else list(src.get("seeds", [])))
            if not seeds:
                continue
            allow = src.get("allow") or [urllib.parse.urlsplit(s).netloc for s in seeds]
            pages = web_learn.crawl(seeds, allow,
                                    max_pages=int(cfg["max_pages_per_run"]),
                                    max_depth=int(cfg.get("max_depth", 0)),
                                    fetcher=fetcher)
            if not pages:
                continue
            web_learn.save_harvest(pages)
            # ham-metin okuma (do_docs) kucuk modeli bozabilir -> varsayilan kapali.
            st = web_learn.study_pages(learner, pages,
                                       do_docs=bool(cfg.get("study_docs", False)),
                                       qa_steps=int(cfg["qa_steps"]),
                                       doc_passes=int(cfg["doc_passes"]))
            for k in ("pages", "docs_studied", "qa_taught"):
                total[k] += st[k]
        if cfg.get("auto_save") and total["pages"]:
            learner.save(os.environ.get("YERELLM_CKPT", "checkpoints/ckpt.pt"))
        WEB["last_run"] = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **total}
        WEB["last_error"] = None
        print(f"[web] ogrenme bitti: {WEB['last_run']}")
        return total
    except Exception as e:
        WEB["last_error"] = str(e)
        print(f"[web] HATA: {e}")
        return None
    finally:
        WEB["running"] = False


def _web_scheduler_loop():
    """Arka plan: interval dolunca ve enabled ise web ogrenmesini tetikler."""
    last = 0.0
    while True:
        try:
            cfg = load_web_config()
            if cfg.get("enabled") and STATE.get("ready") and not WEB["running"]:
                interval = max(60, int(cfg["interval_minutes"]) * 60)
                now = time.monotonic()
                if last == 0.0 or now - last >= interval:
                    last = now
                    run_web_study(cfg)
        except Exception as e:
            WEB["last_error"] = str(e)
        time.sleep(15)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    system: str | None = DEFAULT_SYSTEM
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    backend: str | None = None
    quick_mode: str | None = None


class TeachRequest(BaseModel):
    # ya tam bir konusma (messages) ya da tek soru-cevap (user+assistant) ver
    messages: list[Message] | None = None
    user: str | None = None
    assistant: str | None = None
    steps: int = 6
    save: bool = False     # True -> ogrenilen modeli checkpoint'e de yaz


def _teach_norm(text: str) -> str:
    try:
        return quick_intents._ascii_lower(text)
    except Exception:
        return re.sub(r"\s+", " ", text.lower().replace("ı", "i")).strip()


def _tool_reply_for_text(text: str) -> str | None:
    """Only deterministic/tool-like answers that should bypass the model."""
    for name in ("arithmetic_reply", "weather_reply"):
        fn = getattr(quick_intents, name, None)
        if not fn:
            continue
        reply = fn(text)
        if reply:
            return reply
    return None


def _quick_reply_for_messages(messages: list[dict], mode: str | None = None) -> str | None:
    mode = (mode or CHAT_QUICK_MODE or "model_first").strip().lower()
    if mode in {"off", "none", "false", "0", "model", "model_only"}:
        return None
    last_user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
    if not last_user.strip():
        return None
    if mode in {"tools", "tool", "model_first", "safe"}:
        return _tool_reply_for_text(last_user)
    if mode not in {"legacy", "quick", "quick_first"}:
        return None
    if hasattr(quick_intents, "contextual_reply"):
        reply = quick_intents.contextual_reply(messages)
        if reply:
            return reply
    return quick_intents.quick_reply(last_user)


def _generate_local_answer(messages: list[dict], req: ChatRequest) -> str:
    pieces = []
    for piece in chat_stream(
        STATE["model"], STATE["tok"], messages, STATE["device"],
        system=req.system,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
    ):
        pieces.append(piece)
    return "".join(pieces).strip()


def _is_bad_answer(last_user: str, answer: str) -> bool:
    checker = getattr(quick_intents, "is_bad_model_reply", None)
    return bool(checker and checker(answer, last_user))


def _quality_message(last_user: str) -> str:
    return (
        "Bu soruya çalışan model sağlam bir cevap üretemedi. Hazır hafıza/fallback "
        "cevabı göstermiyorum; worker adapter eğitimi tamamlandıktan sonra aynı "
        "soruyu tekrar dene."
    )


def _finalize_model_answer(last_user: str, answer: str, req: ChatRequest) -> tuple[str, bool, str | None]:
    if not _is_bad_answer(last_user, answer):
        return answer, False, None

    retry_update = {
        "temperature": min(float(req.temperature), 0.25),
        "top_k": min(int(req.top_k or 40), 20),
        "top_p": min(float(req.top_p), 0.85),
    }
    retry_req = req.copy(update=retry_update)
    retry = _generate_local_answer(
        [{"role": m.role, "content": m.content} for m in retry_req.messages],
        retry_req,
    )
    if retry and not _is_bad_answer(last_user, retry):
        return retry, True, None

    greet = getattr(quick_intents, "greeting_reply", lambda _text: None)(last_user)
    if greet:
        return greet, True, None

    if hasattr(quick_intents, "should_lookup") and quick_intents.should_lookup(last_user):
        try:
            fresh = web_lookup.lookup(last_user)
        except Exception:
            fresh = None
        if fresh:
            return (
                f"Araştırdım:\n\n{fresh['summary']}\n\nKaynak: {fresh['title']} (Wikipedia)",
                True,
                fresh["url"],
            )

    if CHAT_FALLBACK_MODE in {"legacy", "quick", "fallback"} and hasattr(quick_intents, "fallback_reply"):
        return quick_intents.fallback_reply(last_user), True, None
    return _quality_message(last_user), True, None


def _chat_backend(value: str | None) -> str:
    backend = (value or CHAT_BACKEND or "local").strip().lower()
    if backend not in {"local", "worker", "auto"}:
        return "local"
    return backend


def _chat_worker_candidates(node_ids: list[str] | None = None) -> list[dict]:
    nodes = REGISTRY.list_nodes()
    selected = None
    if node_ids is not None:
        selected = {str(node_id).strip() for node_id in node_ids if str(node_id).strip()}
        if not selected:
            return []
    candidates = []
    for node in nodes:
        caps = node.get("capabilities", {})
        if node.get("status") != "online":
            continue
        if selected is not None and node.get("node_id") not in selected:
            continue
        if caps.get("can_infer") is False:
            continue
        if not caps.get("allow_training_jobs"):
            continue
        candidates.append(node)

    def score(node: dict) -> tuple[int, float]:
        caps = node.get("capabilities", {})
        device = str(caps.get("preferred_device") or "").lower()
        priority = 2 if device.startswith("cuda") else 1 if device.startswith("mps") else 0
        return (priority, float(node.get("last_seen") or 0))

    return sorted(candidates, key=score, reverse=True)


def _wait_for_command(command_id: str, timeout_sec: int) -> dict | None:
    deadline = time.monotonic() + max(1, timeout_sec)
    while time.monotonic() < deadline:
        command = REGISTRY.get_command(command_id)
        if command and command.get("status") in {"done", "failed", "refused", "cancelled"}:
            return command
        time.sleep(1.0)
    return REGISTRY.get_command(command_id)


def _worker_chat_answer(req: ChatRequest, messages: list[dict]) -> tuple[str | None, str | None]:
    candidates = _chat_worker_candidates()
    if not candidates:
        return None, "worker yok"
    node = candidates[0]
    command = REGISTRY.enqueue_command(node["node_id"], {
        "id": str(uuid.uuid4()),
        "type": "chat_infer",
        "title": "Worker sohbet cevabı",
        "messages": messages,
        "system": req.system,
        "max_new_tokens": req.max_new_tokens,
        "temperature": req.temperature,
        "top_k": req.top_k,
        "top_p": req.top_p,
        "quick_mode": req.quick_mode or CHAT_QUICK_MODE,
        "sync_patch": True,
        "patch_base_url": "/patches/localllm",
    })
    done = _wait_for_command(command["id"], CHAT_WORKER_TIMEOUT_SEC)
    if not done:
        return None, "worker zaman aşımı"
    result = done.get("result") or {}
    if done.get("status") == "done" and result.get("answer"):
        return str(result["answer"]).strip(), f"worker:{node.get('name') or node.get('node_id')}"
    return None, done.get("error") or result.get("error") or done.get("status") or "worker hata"


def _last_teach_pair(messages: list[dict]) -> tuple[str, str] | None:
    assistant = ""
    for message in reversed(messages):
        role = message.get("role")
        content = str(message.get("content") or "")
        if role == "assistant" and content.strip() and not assistant:
            assistant = content.strip()
            continue
        if role == "user" and assistant:
            return content.strip(), assistant
    return None


def _is_arithmetic_pair(user: str, assistant: str) -> bool:
    user_norm = _teach_norm(user)
    assistant_norm = _teach_norm(assistant).strip(" .")
    has_math = bool(re.search(r"\d+\s*[\+\-\*/x]\s*\d+", user_norm))
    return has_math and bool(re.fullmatch(r"(sonuc:\s*)?-?\d+(?:[.,]\d+)?(\s*eder)?", assistant_norm))


def _is_code_request(user: str) -> bool:
    t = _teach_norm(user)
    tech = ("kod", "code", "html", "css", "php", "javascript", "python", "fastapi", "docker", "api", "fonksiyon")
    action = ("yaz", "ornek", "dilinde", "istiyorum", "olsun", "yapi", "form", "endpoint", "dockerfile")
    return any(word in t for word in tech) and any(word in t for word in action)


def _looks_like_code_answer(assistant: str) -> bool:
    t = _teach_norm(assistant)
    markers = (
        "```", "<!doctype", "<html", "</html>", "<body", "<div", "<?php", "$",
        "def ", "function ", "const ", "let ", "class ", "@app.", "from fastapi",
        "dockerfile", "from python", "cmd [", "echo ", "return ",
    )
    return any(marker in t for marker in markers)


def _teach_reject_reason(user: str, assistant: str) -> str | None:
    user = (user or "").strip()
    assistant = (assistant or "").strip()
    if not user or not assistant:
        return "Soru ve dogru cevap bos olamaz."

    u = _teach_norm(user).strip(" .!?")
    a = _teach_norm(assistant).strip(" .!?")
    if a == u:
        return "Soru ile ayni metni dogru cevap olarak hafizaya kaydetmiyorum."

    bad_fragments = (
        "bunu tam bilmiyorum",
        "tam bilmiyorum",
        "bende bilmiyorum",
        "ben de bilmiyorum",
        "dogru cevabi yazarsan",
        "hemen ogrenirim",
        "ogrenme tamamlanamadi",
        "ogrendim bundan sonra",
    )
    if any(fragment in a for fragment in bad_fragments):
        return "Eksik veya 'bilmiyorum' cevabini hafizaya kaydetmiyorum."

    if len(a) < 12 and not _is_arithmetic_pair(user, assistant):
        return "Dogru cevap cok kisa; kalici ogrenme icin acik ve tam cevap yaz."

    if _is_code_request(user) and not _looks_like_code_answer(assistant):
        return "Kod istegi icin gercek kod ornegi olmadan hafizaya kaydetmiyorum."

    return None


@app.on_event("startup")
def _startup():
    ckpt = os.environ.get("YERELLM_CKPT", "checkpoints/ckpt.pt")
    adapter = os.environ.get("YERELLM_ADAPTER", "auto")
    tok_path = os.environ.get("YERELLM_TOKENIZER", "tokenizer/tokenizer.json")
    device = os.environ.get("YERELLM_DEVICE",
                            "cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(ckpt):
        print(f"[server] UYARI: checkpoint yok ({ckpt}). Once train.py calistir.")
        STATE["ready"] = False
        return
    tok = load_tokenizer(tok_path)
    model, cfg = load_model(ckpt, device, adapter_path=adapter)
    learner = OnlineLearner(model, tok, device)
    # Tsetlin niyet/kapsam hakemi (varsa) -- aciklanabilir, gradyansiz, CPU'da
    gate = None
    try:
        from tsetlin.intent_gate import IntentGate
        gate = IntentGate()
        print("[server] niyet hakemi (Tsetlin) yuklendi: aciklanabilir niyet rozeti aktif")
    except Exception as e:
        print(f"[server] niyet hakemi yok ({e}) -- once: python train_intent.py")
    STATE.update(model=model, tok=tok, cfg=cfg, device=device,
                 learner=learner, gate=gate, ready=True)
    adapter_path = getattr(model, "adapter_path", "")
    print(f"[server] model hazir: {model.num_params()/1e6:.0f}M, device={device}"
          f"{', adapter=' + adapter_path if adapter_path else ''}")
    print(f"[server] online ogrenme aktif (ogretilenler: data/chat_live/learned.jsonl)")
    # web'den otomatik ogrenme zamanlayicisi (web_sources.json ile kontrol edilir)
    t = threading.Thread(target=_web_scheduler_loop, daemon=True)
    t.start()
    cfg = load_web_config()
    print(f"[server] web ogrenme zamanlayicisi basladi "
          f"(enabled={cfg.get('enabled')}, her {cfg.get('interval_minutes')} dk)")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/cluster")
def cluster():
    return FileResponse(os.path.join(STATIC_DIR, "cluster.html"))


@app.get("/api/health")
def health():
    return {"ready": STATE.get("ready", False),
            "device": STATE.get("device"),
            "adapter": getattr(STATE.get("model"), "adapter_path", None) if STATE.get("ready") else None,
            "params_m": (STATE["model"].num_params() / 1e6) if STATE.get("ready") else None}


@app.get("/api/nodes")
def nodes(request: Request):
    """Merkezi kayit defterindeki yerel compute node'larini listeler."""
    _require_api_token(request)
    return {"ok": True, "nodes": REGISTRY.list_nodes(), "events": REGISTRY.recent_events(20)}


@app.post("/api/nodes/register")
def register_node(request: Request, payload: dict = Body(default={})):
    """Yerel worker bu uca kendi donanim/kabiliyet bilgisini kaydeder."""
    _require_api_token(request)
    node = REGISTRY.upsert_node(payload or {})
    return {"ok": True, "node_id": node["node_id"], "node": node}


@app.post("/api/nodes/{node_id}/heartbeat")
def node_heartbeat(node_id: str, request: Request, payload: dict = Body(default={})):
    _require_api_token(request)
    node = REGISTRY.heartbeat(node_id, payload or {})
    if not node:
        return {"ok": False, "error": "node kayitli degil"}
    return {"ok": True, "node": node}


@app.post("/api/nodes/{node_id}/events")
def node_event(node_id: str, request: Request, payload: dict = Body(default={})):
    _require_api_token(request)
    REGISTRY.add_event(
        node_id,
        str((payload or {}).get("kind") or "event"),
        str((payload or {}).get("message") or ""),
        dict((payload or {}).get("data") or {}),
    )
    return {"ok": True}


@app.post("/api/nodes/{node_id}/commands")
def enqueue_node_command(node_id: str, request: Request, payload: dict = Body(default={})):
    """Bir worker icin calistirilacak isi merkezi kuyruğa ekler."""
    _require_api_token(request)
    try:
        return {"ok": True, "command": REGISTRY.enqueue_command(node_id, payload or {})}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/nodes/{node_id}/commands/poll")
def poll_node_commands(node_id: str, request: Request, payload: dict = Body(default={})):
    """Worker bu uctan izinli islerini alir."""
    _require_api_token(request)
    limit = int((payload or {}).get("limit") or 1)
    return {"ok": True, "commands": REGISTRY.poll_commands(node_id, limit=limit)}


@app.post("/api/commands/{command_id}/status")
def update_command_status(command_id: str, request: Request, payload: dict = Body(default={})):
    _require_api_token(request)
    command = REGISTRY.update_command_status(command_id, payload or {})
    if not command:
        return {"ok": False, "error": "komut bulunamadi"}
    return {"ok": True, "command": command}


@app.get("/api/commands")
def list_commands(request: Request, node_id: str | None = None, limit: int = 50):
    _require_api_token(request)
    return {"ok": True, "commands": REGISTRY.list_commands(node_id=node_id, limit=limit)}


@app.get("/api/sync/status")
def sync_status(request: Request):
    _require_api_token(request)
    return {
        "ok": True,
        "summary": _sync_summary(),
        "files": _patch_manifest(),
        "commands": REGISTRY.list_commands(limit=30),
    }


@app.post("/api/sync/patch")
def queue_patch_sync(request: Request, payload: dict = Body(default={})):
    _require_api_token(request)
    node_ids = payload.get("node_ids")
    if node_ids is not None and not isinstance(node_ids, list):
        node_ids = []
    eligible, waiting = _eligible_sync_nodes(node_ids=node_ids)
    if not eligible:
        return {
            "ok": False,
            "error": "Senkron icin en az bir online worker --allow-training-jobs veya --allow-remote-commands ile acik olmali.",
            "waiting": waiting,
        }
    queued = []
    for node in eligible:
        queued.append(REGISTRY.enqueue_command(node["node_id"], {
            "type": "sync_patch",
            "title": "localllm dosya senkronu",
            "patch_base_url": "/patches/localllm",
            "sync_patch": True,
        }))
    return {"ok": True, "queued": queued, "waiting": waiting, "summary": _sync_summary()}


@app.post("/api/chat-train")
def queue_chat_train(request: Request, payload: dict = Body(default={})):
    """Secili worker'larda guvenli, yapilandirilmis sohbet egitimi isi kuyruga alir."""
    _require_api_token(request)
    prepared_data = str(
        payload.get("prepared_data") or payload.get("prepared_data_dir") or ""
    ).strip()
    training_source = str(payload.get("training_source") or "").strip().lower()
    use_prepared_data = bool(prepared_data) or training_source in {"prepared", "dataset", "bulk"}
    if use_prepared_data:
        prepared_data = prepared_data or "data/chat_200m_plus_bin"
        normalized = os.path.normpath(prepared_data).replace("\\", "/")
        parts = [part for part in normalized.split("/") if part]
        if os.path.isabs(prepared_data) or ".." in parts:
            return {"ok": False, "error": "Hazir veri yolu repo icinde goreli olmali."}
        clean = []
    else:
        messages = payload.get("messages")
        if not isinstance(messages, list):
            user = str(payload.get("user") or "").strip()
            assistant = str(payload.get("assistant") or "").strip()
            messages = [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
        clean = [
            {"role": str(item.get("role") or "").strip(), "content": str(item.get("content") or "").strip()}
            for item in messages
            if isinstance(item, dict)
        ]
        clean = [item for item in clean if item["role"] in {"system", "user", "assistant"} and item["content"]]
        if not any(item["role"] == "user" for item in clean) or not any(item["role"] == "assistant" for item in clean):
            return {"ok": False, "error": "Egitim icin user ve assistant mesaji gerekli."}

    node_ids = payload.get("node_ids")
    if node_ids is not None and not isinstance(node_ids, list):
        node_ids = []
    eligible, waiting = _eligible_sync_nodes(node_ids=node_ids)
    if not eligible:
        return {
            "ok": False,
            "error": "Sohbet egitimi icin worker --allow-training-jobs ile acik olmali.",
            "waiting": waiting,
        }

    queued = []
    for node in eligible:
        command = {
            "type": "chat_train",
            "title": "200M sohbet adapter egitimi" if use_prepared_data else "Sohbet egitimi",
            "auto_train": _payload_bool(payload.get("auto_train"), True),
            "max_steps": _payload_int(payload.get("max_steps"), 80, 1, 5000),
            "preset": str(payload.get("preset") or "tiny-30m"),
            "adapter": _payload_bool(payload.get("adapter"), True),
            "adapter_out": str(payload.get("adapter_out") or "checkpoints"),
            "batch_size": _payload_int(payload.get("batch_size"), 4, 1, 64),
            "grad_accum": _payload_int(payload.get("grad_accum"), 4, 1, 128),
            "sync_patch": True,
            "patch_base_url": "/patches/localllm",
        }
        if str(payload.get("adapter_resume") or "").strip():
            command["adapter_resume"] = str(payload.get("adapter_resume")).strip()
        if use_prepared_data:
            command.update({
                "training_source": "prepared",
                "prepared_data": prepared_data,
            })
        else:
            command["sample"] = {"messages": clean}
        queued.append(REGISTRY.enqueue_command(node["node_id"], command))
    return {"ok": True, "queued": queued, "waiting": waiting}


@app.get("/patches/localllm/{rel_path:path}")
def patch_file(rel_path: str, request: Request):
    _require_api_token(request)
    return FileResponse(_safe_patch_file(rel_path))


@app.get("/api/train-sessions")
def train_sessions(request: Request):
    _require_api_token(request)
    return {"ok": True, "sessions": REGISTRY.list_sessions()}


@app.post("/api/train-sessions")
def create_train_session(request: Request, payload: dict = Body(default={})):
    """LAN egitimi icin rank/komut plani uretir; komutlari calistirmaz."""
    _require_api_token(request)
    try:
        payload = dict(payload or {})
        if not str(payload.get("master_addr") or "").strip():
            payload["master_addr"] = request.url.hostname or "127.0.0.1"
        return {"ok": True, "session": REGISTRY.create_session(payload or {})}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/chat")
def chat(req: ChatRequest, request: Request):
    _require_api_token(request)
    if not STATE.get("ready"):
        def err():
            yield "data: " + json.dumps(
                {"error": "Model yuklenmedi. Once egitip checkpoint olustur."}
            ) + "\n\n"
        return StreamingResponse(err(), media_type="text/event-stream")

    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    last_user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
    quick_mode = req.quick_mode or CHAT_QUICK_MODE
    backend = _chat_backend(req.backend)
    quick_res = _quick_reply_for_messages(messages, quick_mode) if last_user.strip() else None

    # Tsetlin hakemi "kapsam disi / emin degil" derse -> Wikipedia'da CANLI ara
    # (model uydurmasin, gercek kaynaktan tam-nokta-atisi cevap gelsin).
    lookup_res = None
    gate = STATE.get("gate")
    if quick_res is None and gate is not None and last_user.strip():
        try:
            if not gate.classify(last_user).get("in_scope"):
                lookup_res = web_lookup.lookup(last_user)
        except Exception:
            pass
    if quick_res is None and lookup_res is None and last_user.strip():
        try:
            if hasattr(quick_intents, "should_lookup") and quick_intents.should_lookup(last_user):
                lookup_res = web_lookup.lookup(last_user)
        except Exception:
            pass

    def event_stream():
        if quick_res:
            for w in re.findall(r"\S+\s*|\n", quick_res):
                yield "data: " + json.dumps({"token": w}) + "\n\n"
            yield "data: " + json.dumps({"done": True, "source": "quick_intents"}) + "\n\n"
            return
        if lookup_res:
            text = (f"Kontrol ettim:\n\n{lookup_res['summary']}"
                    f"\n\nKaynak: {lookup_res['title']} (Wikipedia)")
            for w in re.findall(r"\S+\s*|\n", text):
                yield "data: " + json.dumps({"token": w}) + "\n\n"
            yield "data: " + json.dumps({"done": True, "source": lookup_res["url"]}) + "\n\n"
            return
        cleaned = False
        source = None
        try:
            answer = None
            if backend in {"worker", "auto"}:
                answer, worker_source = _worker_chat_answer(req, messages)
                if answer:
                    source = worker_source
                elif backend == "worker":
                    answer = f"Worker sohbet cevabı alınamadı: {worker_source or 'bilinmeyen hata'}"
                    cleaned = True
            if answer is None:
                answer = _generate_local_answer(messages, req)
                source = "local_model"
                answer, cleaned, fresh_source = _finalize_model_answer(last_user, answer, req)
                if fresh_source:
                    source = fresh_source
            for piece in re.findall(r"\S+\s*|\n", answer):
                yield "data: " + json.dumps({"token": piece}) + "\n\n"
        except Exception as e:  # uretimde hata olursa istemciye bildir
            yield "data: " + json.dumps({"error": str(e)}) + "\n\n"
        done = {"done": True, "cleaned": cleaned}
        if source:
            done["source"] = source
        yield "data: " + json.dumps(done) + "\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/teach")
def teach(req: TeachRequest, request: Request):
    """Modeli sohbetten gelen bir ornekle ANINDA ince-ayar eder (kendini egitir)."""
    _require_api_token(request)
    if not STATE.get("ready"):
        return {"ok": False, "error": "Model yuklenmedi."}
    learner: OnlineLearner = STATE["learner"]
    if req.messages:
        msgs = [{"role": m.role, "content": m.content} for m in req.messages]
        pair = _last_teach_pair(msgs)
        if pair is None:
            return {"ok": False, "error": "Ogretilecek user/asistan cifti bulunamadi."}
        reason = _teach_reject_reason(pair[0], pair[1])
        if reason:
            return {"ok": False, "error": reason}
        res = learner.teach(msgs, steps=req.steps)
    elif req.user is not None and req.assistant is not None:
        reason = _teach_reject_reason(req.user, req.assistant)
        if reason:
            return {"ok": False, "error": reason}
        res = learner.teach_pair(req.user, req.assistant, steps=req.steps)
    else:
        return {"ok": False, "error": "messages ya da (user+assistant) gerekli."}
    if res is None:
        return {"ok": False, "error": "Ogretilecek bir asistan cevabi bulunamadi."}
    if req.save:
        ckpt = os.environ.get("YERELLM_CKPT", "checkpoints/ckpt.pt")
        res["saved"] = learner.save(ckpt)
    return {"ok": True, **res}


@app.post("/api/save")
def save(request: Request):
    """Online ogrenmeyle guncellenen modeli checkpoint'e kalici yazar."""
    _require_api_token(request)
    if not STATE.get("ready"):
        return {"ok": False, "error": "Model yuklenmedi."}
    ckpt = os.environ.get("YERELLM_CKPT", "checkpoints/ckpt.pt")
    STATE["learner"].save(ckpt)
    return {"ok": True, "saved": ckpt}


@app.post("/api/intent")
def intent(request: Request, req: dict = Body(default={})):
    """Tsetlin hakemi: mesajin niyetini + OKUNABILIR gerekceyi dondurur."""
    _require_api_token(request)
    g = STATE.get("gate")
    if g is None:
        return {"ok": False, "error": "niyet hakemi yuklu degil"}
    text = (req or {}).get("text", "")
    if not text.strip():
        return {"ok": False, "error": "bos metin"}
    try:
        return {"ok": True, **g.classify(text)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/api/web/status")
def web_status(request: Request):
    _require_api_token(request)
    cfg = load_web_config()
    return {"enabled": cfg.get("enabled"), "running": WEB["running"],
            "interval_minutes": cfg.get("interval_minutes"),
            "last_run": WEB["last_run"], "last_error": WEB["last_error"],
            "sources": cfg.get("sources", [])}


@app.post("/api/web/study")
def web_study_now(request: Request):
    """Web ogrenmesini HEMEN (arka planda) tetikler."""
    _require_api_token(request)
    if not STATE.get("ready"):
        return {"ok": False, "error": "Model yuklenmedi."}
    if WEB["running"]:
        return {"ok": False, "error": "Web ogrenmesi zaten calisiyor."}
    threading.Thread(target=run_web_study, daemon=True).start()
    return {"ok": True, "started": True}


@app.post("/api/web/config")
def web_config(request: Request, patch: dict = Body(default={})):
    """web_sources.json'i gunceller (enabled, interval_minutes, sources, ...)."""
    _require_api_token(request)
    cfg = load_web_config()
    for k, v in (patch or {}).items():
        if k in DEFAULT_WEB_CONFIG:
            cfg[k] = v
    save_web_config(cfg)
    return {"ok": True, "config": cfg}


# statik dosyalar (varsa)
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def run():
    import argparse
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/ckpt.pt")
    ap.add_argument("--adapter", default="auto")
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--api-token", default=None, help="LAN/uzak erisim icin X-YerelLM-Token/Bearer token")
    ap.add_argument("--cors-origins", default=None, help="virgulle ayrilmis izinli browser origin listesi")
    args = ap.parse_args()
    os.environ["YERELLM_CKPT"] = args.ckpt
    os.environ["YERELLM_ADAPTER"] = args.adapter
    os.environ["YERELLM_TOKENIZER"] = args.tokenizer
    os.environ["YERELLM_DEVICE"] = args.device
    if args.api_token is not None:
        os.environ["YERELLM_API_TOKEN"] = args.api_token
    if args.cors_origins is not None:
        os.environ["YERELLM_CORS_ORIGINS"] = args.cors_origins
    print(f"[server] http://{args.host}:{args.port}  (ckpt={args.ckpt}, device={args.device})")
    uvicorn.run("server.app:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    run()
