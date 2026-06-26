"""
data/build_bilge_data.py
========================
Workflow ciktisindaki (JSON) Bilge konusmalarini egitim verisine cevirir:
  * data/chat_bilge/bilge.jsonl  -> SFT (sohbet) verisi (her satir bir konusma)
  * data/raw_bilge/bilge_text.txt -> tokenizer/pretrain icin duz Turkce diyalog

Kullanim:
    python -m data.build_bilge_data <workflow_output.json>
"""
from __future__ import annotations
import json
import os
import sys


def _find_conversations(obj):
    """JSON icinde 'conversations' anahtarini herhangi bir derinlikte bulur."""
    if isinstance(obj, dict):
        if isinstance(obj.get("conversations"), list):
            return obj
        for v in obj.values():
            r = _find_conversations(v)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _find_conversations(v)
            if r is not None:
                return r
    return None


def load_result(path: str) -> dict:
    raw = open(path, "r", encoding="utf-8").read().strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        i, j = raw.find("{"), raw.rfind("}")
        data = json.loads(raw[i:j + 1])
    found = _find_conversations(data)
    if found is None:
        raise SystemExit("Ciktida 'conversations' bulunamadi.")
    return found


def valid(conv) -> bool:
    msgs = conv.get("messages")
    if not isinstance(msgs, list) or len(msgs) < 2:
        return False
    if msgs[0].get("role") != "user":
        return False
    expect = "user"
    for m in msgs:
        if m.get("role") != expect or not str(m.get("content", "")).strip():
            return False
        expect = "assistant" if expect == "user" else "user"
    return msgs[-1].get("role") == "assistant"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="workflow ciktisi (JSON)")
    ap.add_argument("--out-jsonl", default="data/chat_bilge/bilge.jsonl")
    ap.add_argument("--out-text", default="data/raw_bilge/bilge_text.txt")
    args = ap.parse_args()

    res = load_result(args.input)
    convos = res.get("conversations", [])
    print(f"[build] ham konusma: {len(convos)} | counts={res.get('counts')} "
          f"| cikarilan={res.get('removed')}")

    seen, clean = set(), []
    for c in convos:
        if not valid(c):
            continue
        key = c["messages"][0]["content"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        clean.append(c)
    print(f"[build] gecerli + tekil: {len(clean)} konusma")

    for p in (args.out_jsonl, args.out_text):
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)

    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for c in clean:
            f.write(json.dumps({"messages": c["messages"]}, ensure_ascii=False) + "\n")

    # duz metin (tokenizer + opsiyonel pretrain). Diyaloglar bos satirla ayrilir.
    with open(args.out_text, "w", encoding="utf-8") as f:
        for c in clean:
            for m in c["messages"]:
                who = "Kullanici" if m["role"] == "user" else "Bilge"
                f.write(f"{who}: {m['content'].strip()}\n")
            f.write("\n")

    n_turns = sum(len(c["messages"]) for c in clean)
    print(f"[build] yazildi -> {args.out_jsonl} ({len(clean)} konusma, {n_turns} mesaj)")
    print(f"[build] yazildi -> {args.out_text}")


if __name__ == "__main__":
    main()
