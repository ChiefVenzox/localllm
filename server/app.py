"""
server/app.py
=============
FastAPI tabanli web sohbet sunucusu. Modeli yukler ve token token (streaming)
cevap veren bir /api/chat ucu sunar. Tarayicidan kullanilan arayuz: static/index.html

Kullanim:
    python -m server.app --ckpt checkpoints/ckpt.pt --device cuda
    # sonra tarayicidan: http://127.0.0.1:8000
"""
from __future__ import annotations
import json
import os
import re
import threading
import time
import urllib.parse

from fastapi import FastAPI, Body
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

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(HERE, "static")

app = FastAPI(title="yerelLLM")
STATE: dict = {}

# ---- Web'den kendi kendine ogrenme (otomatik/zamanli) --------------------
WEB: dict = {"running": False, "last_run": None, "last_error": None,
             "config_path": os.environ.get("YERELLM_WEB_CONFIG", "web_sources.json")}
DEFAULT_WEB_CONFIG = {
    "enabled": False, "interval_minutes": 60, "max_pages_per_run": 4,
    "max_depth": 0, "request_delay_sec": 2.0, "respect_robots": True,
    "qa_steps": 6, "doc_passes": 1, "study_docs": False, "auto_save": True,
    "sources": [],
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


class TeachRequest(BaseModel):
    # ya tam bir konusma (messages) ya da tek soru-cevap (user+assistant) ver
    messages: list[Message] | None = None
    user: str | None = None
    assistant: str | None = None
    steps: int = 6
    save: bool = False     # True -> ogrenilen modeli checkpoint'e de yaz


@app.on_event("startup")
def _startup():
    ckpt = os.environ.get("YERELLM_CKPT", "checkpoints/ckpt.pt")
    tok_path = os.environ.get("YERELLM_TOKENIZER", "tokenizer/tokenizer.json")
    device = os.environ.get("YERELLM_DEVICE",
                            "cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(ckpt):
        print(f"[server] UYARI: checkpoint yok ({ckpt}). Once train.py calistir.")
        STATE["ready"] = False
        return
    tok = load_tokenizer(tok_path)
    model, cfg = load_model(ckpt, device)
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
    print(f"[server] model hazir: {model.num_params()/1e6:.0f}M, device={device}")
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


@app.get("/api/health")
def health():
    return {"ready": STATE.get("ready", False),
            "device": STATE.get("device"),
            "params_m": (STATE["model"].num_params() / 1e6) if STATE.get("ready") else None}


@app.post("/api/chat")
def chat(req: ChatRequest):
    if not STATE.get("ready"):
        def err():
            yield "data: " + json.dumps(
                {"error": "Model yuklenmedi. Once egitip checkpoint olustur."}
            ) + "\n\n"
        return StreamingResponse(err(), media_type="text/event-stream")

    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    last_user = messages[-1]["content"] if messages else ""

    # Tsetlin hakemi "kapsam disi / emin degil" derse -> Wikipedia'da CANLI ara
    # (model uydurmasin, gercek kaynaktan tam-nokta-atisi cevap gelsin).
    lookup_res = None
    gate = STATE.get("gate")
    if gate is not None and last_user.strip():
        try:
            if not gate.classify(last_user).get("in_scope"):
                lookup_res = web_lookup.lookup(last_user)
        except Exception:
            pass

    def event_stream():
        if lookup_res:
            text = (f"Bunu kendim bilmiyordum, araştırdım 🔎\n\n{lookup_res['summary']}"
                    f"\n\n— Kaynak: {lookup_res['title']} (Wikipedia)")
            for w in re.findall(r"\S+\s*|\n", text):
                yield "data: " + json.dumps({"token": w}) + "\n\n"
            yield "data: " + json.dumps({"done": True, "source": lookup_res["url"]}) + "\n\n"
            return
        try:
            for piece in chat_stream(
                STATE["model"], STATE["tok"], messages, STATE["device"],
                system=req.system, max_new_tokens=req.max_new_tokens,
                temperature=req.temperature, top_k=req.top_k, top_p=req.top_p,
            ):
                yield "data: " + json.dumps({"token": piece}) + "\n\n"
        except Exception as e:  # uretimde hata olursa istemciye bildir
            yield "data: " + json.dumps({"error": str(e)}) + "\n\n"
        yield "data: " + json.dumps({"done": True}) + "\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/teach")
def teach(req: TeachRequest):
    """Modeli sohbetten gelen bir ornekle ANINDA ince-ayar eder (kendini egitir)."""
    if not STATE.get("ready"):
        return {"ok": False, "error": "Model yuklenmedi."}
    learner: OnlineLearner = STATE["learner"]
    if req.messages:
        msgs = [{"role": m.role, "content": m.content} for m in req.messages]
        res = learner.teach(msgs, steps=req.steps)
    elif req.user is not None and req.assistant is not None:
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
def save():
    """Online ogrenmeyle guncellenen modeli checkpoint'e kalici yazar."""
    if not STATE.get("ready"):
        return {"ok": False, "error": "Model yuklenmedi."}
    ckpt = os.environ.get("YERELLM_CKPT", "checkpoints/ckpt.pt")
    STATE["learner"].save(ckpt)
    return {"ok": True, "saved": ckpt}


@app.post("/api/intent")
def intent(req: dict = Body(default={})):
    """Tsetlin hakemi: mesajin niyetini + OKUNABILIR gerekceyi dondurur."""
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
def web_status():
    cfg = load_web_config()
    return {"enabled": cfg.get("enabled"), "running": WEB["running"],
            "interval_minutes": cfg.get("interval_minutes"),
            "last_run": WEB["last_run"], "last_error": WEB["last_error"],
            "sources": cfg.get("sources", [])}


@app.post("/api/web/study")
def web_study_now():
    """Web ogrenmesini HEMEN (arka planda) tetikler."""
    if not STATE.get("ready"):
        return {"ok": False, "error": "Model yuklenmedi."}
    if WEB["running"]:
        return {"ok": False, "error": "Web ogrenmesi zaten calisiyor."}
    threading.Thread(target=run_web_study, daemon=True).start()
    return {"ok": True, "started": True}


@app.post("/api/web/config")
def web_config(patch: dict = Body(default={})):
    """web_sources.json'i gunceller (enabled, interval_minutes, sources, ...)."""
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
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    os.environ["YERELLM_CKPT"] = args.ckpt
    os.environ["YERELLM_TOKENIZER"] = args.tokenizer
    os.environ["YERELLM_DEVICE"] = args.device
    print(f"[server] http://{args.host}:{args.port}  (ckpt={args.ckpt}, device={args.device})")
    uvicorn.run("server.app:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    run()
