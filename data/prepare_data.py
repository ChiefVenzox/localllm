"""
data/prepare_data.py
====================
Ham metni token id'lerine cevirip egitim icin ikili (binary) dosyalara yazar.
Iki mod:

  --mode pretrain   (.txt dosyalari)  her dosya/paragraf bir dokuman; aralara
                    <|endoftext|> konur.
  --mode chat       (.jsonl dosyalari) her satir {"messages":[...]} ; sohbet
                    sablonuyla kodlanir.

Cikti: <out>/train.bin, <out>/val.bin (memmap), <out>/meta.json

Kullanim:
    python -m data.prepare_data --mode pretrain --input data/raw
    python -m data.prepare_data --mode chat --input data/chat --out data/chat_bin
"""
from __future__ import annotations
import argparse
import glob
import hashlib
import json
import os
import stat
import shutil
import tarfile
import urllib.parse
import urllib.request
import zipfile

import numpy as np
from tqdm import tqdm

from tokenizer import load_tokenizer
from chat_template import encode_chat, DEFAULT_SYSTEM


DEFAULT_CHAT_SYNC_URL = ""
DEFAULT_CHAT_SYNC_SHA256 = ""
AUTO_CHAT_SYNC_INPUTS = {"chat_remote", "chat_sync"}


def list_files(inputs, exts):
    files = []
    for item in inputs:
        if os.path.isdir(item):
            for e in exts:
                files += glob.glob(os.path.join(item, "**", f"*{e}"), recursive=True)
        elif os.path.isfile(item):
            files.append(item)
    return sorted(set(files))


def is_url(value: str) -> bool:
    parsed = urllib.parse.urlparse(value)
    return parsed.scheme in {"http", "https"}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_safe_archive_path(member_name: str, destination: str):
    root = os.path.realpath(destination)
    target = os.path.realpath(os.path.join(destination, member_name))
    try:
        is_inside = os.path.commonpath([root, target]) == root
    except ValueError:
        is_inside = False
    if not is_inside:
        raise SystemExit(f"Guvensiz arsiv yolu: {member_name}")


def safe_extract_tar(tar: tarfile.TarFile, destination: str):
    for member in tar.getmembers():
        assert_safe_archive_path(member.name, destination)
        if member.issym() or member.islnk() or member.isdev() or member.isfifo():
            raise SystemExit(f"Guvensiz arsiv girdisi: {member.name}")
        if not (member.isfile() or member.isdir()):
            raise SystemExit(f"Desteklenmeyen arsiv girdisi: {member.name}")
    tar.extractall(destination)


def safe_extract_zip(zf: zipfile.ZipFile, destination: str):
    for info in zf.infolist():
        assert_safe_archive_path(info.filename, destination)
        mode = info.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise SystemExit(f"Guvensiz arsiv girdisi: {info.filename}")
    zf.extractall(destination)


def download_chat_bundle(url: str, expected_sha256: str, cache_dir: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    key = expected_sha256 or hashlib.sha256(url.encode("utf-8")).hexdigest()
    bundle_path = os.path.join(cache_dir, f"{key}.bundle")
    extract_dir = os.path.join(cache_dir, key)
    marker = os.path.join(extract_dir, ".ready")

    if not os.path.exists(bundle_path) or (
        expected_sha256 and sha256_file(bundle_path) != expected_sha256
    ):
        tmp = bundle_path + ".tmp"
        print(f"[data] merkezi sohbet paketi indiriliyor: {url}")
        with urllib.request.urlopen(url, timeout=60) as response, open(tmp, "wb") as handle:
            shutil.copyfileobj(response, handle)
        if expected_sha256:
            got = sha256_file(tmp)
            if got != expected_sha256:
                os.remove(tmp)
                raise SystemExit(
                    f"Merkezi sohbet paketi SHA256 uyusmadi: {got} != {expected_sha256}"
                )
        os.replace(tmp, bundle_path)

    if os.path.exists(marker):
        return extract_dir

    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    os.makedirs(extract_dir, exist_ok=True)

    try:
        if tarfile.is_tarfile(bundle_path):
            with tarfile.open(bundle_path, "r:*") as tar:
                safe_extract_tar(tar, extract_dir)
        elif zipfile.is_zipfile(bundle_path):
            with zipfile.ZipFile(bundle_path) as zf:
                safe_extract_zip(zf, extract_dir)
        else:
            # Tek dosyali jsonl bundle.
            dst = os.path.join(extract_dir, "bundle.jsonl")
            shutil.copy2(bundle_path, dst)
        with open(marker, "w", encoding="utf-8") as handle:
            handle.write(url + "\n")
    except Exception:
        shutil.rmtree(extract_dir, ignore_errors=True)
        raise
    return extract_dir


def should_auto_chat_sync(args) -> bool:
    if args.mode != "chat" or args.no_chat_sync:
        return False
    if args.chat_sync_url:
        return True
    if os.environ.get("YERELLM_CHAT_SYNC_URL"):
        return True
    enabled = (
        args.chat_sync_auto
        or os.environ.get("YERELLM_ENABLE_CHAT_SYNC_AUTO", "").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    if not enabled:
        return False
    if os.environ.get("YERELLM_DISABLE_CHAT_SYNC", "").strip().lower() in {"1", "true", "yes"}:
        return False
    if not (os.environ.get("YERELLM_CHAT_SYNC_URL") or DEFAULT_CHAT_SYNC_URL):
        return False
    for item in args.input:
        if is_url(item):
            return True
        name = os.path.basename(os.path.normpath(item))
        if name in AUTO_CHAT_SYNC_INPUTS:
            return True
    return False


def materialize_inputs(args):
    inputs = []
    for item in args.input:
        if is_url(item):
            inputs.append(download_chat_bundle(item, "", args.chat_sync_cache))
        else:
            inputs.append(item)

    if should_auto_chat_sync(args):
        url = (
            args.chat_sync_url
            or os.environ.get("YERELLM_CHAT_SYNC_URL")
            or DEFAULT_CHAT_SYNC_URL
        )
        expected_sha = (
            args.chat_sync_sha256
            or os.environ.get("YERELLM_CHAT_SYNC_SHA256")
            or DEFAULT_CHAT_SYNC_SHA256
        )
        if url:
            synced = download_chat_bundle(url, expected_sha, args.chat_sync_cache)
            print(f"[data] merkezi sohbet paketi hazir: {synced}")
            inputs.append(synced)
    return inputs


def iter_pretrain_docs(files):
    """Her .txt dosyasini blank-line ile dokumanlara boler."""
    for fp in files:
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            buf = []
            for line in f:
                if line.strip() == "" and buf:
                    yield "".join(buf).strip()
                    buf = []
                else:
                    buf.append(line)
            if buf:
                yield "".join(buf).strip()


def iter_chat_convos(files):
    for fp in files:
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                yield obj["messages"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pretrain", "chat"], default="pretrain")
    ap.add_argument("--input", nargs="+", default=["data/raw"])
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--out", default="data/bin")
    ap.add_argument("--val-ratio", type=float, default=0.01)
    ap.add_argument("--min-split-tokens", type=int, default=2050,
                    help="train/val dosyalarinda gereken minimum token; chat smoke egitimi icin kucuk veri tekrar edilir.")
    ap.add_argument("--system", default=None,
                    help="chat modunda her konusmaya eklenecek sistem mesaji")
    ap.add_argument("--default-system", action="store_true",
                    help="chat modunda Bilge persona'sini (DEFAULT_SYSTEM) ekle")
    ap.add_argument("--chat-sync-url", default=None,
                    help="chat modunda indirilecek merkezi jsonl/tar/zip paket URL'si")
    ap.add_argument("--chat-sync-sha256", default=None,
                    help="merkezi sohbet paketinin beklenen SHA256 degeri")
    ap.add_argument("--chat-sync-cache", default="data/.cache/chat_sync",
                    help="indirilen merkezi sohbet paketleri icin cache klasoru")
    ap.add_argument("--chat-sync-auto", action="store_true",
                    help="chat_remote/chat_sync input adlari icin env/default sync URL'sini otomatik dene")
    ap.add_argument("--no-chat-sync", action="store_true",
                    help="otomatik merkezi sohbet paketi senkronunu kapat")
    args = ap.parse_args()
    system = DEFAULT_SYSTEM if args.default_system else args.system

    tok = load_tokenizer(args.tokenizer)
    dtype = np.uint16 if tok.vocab_size <= 65535 else np.uint32
    print(f"[data] tokenizer vocab={tok.vocab_size}, dtype={np.dtype(dtype).name}")

    exts = [".txt"] if args.mode == "pretrain" else [".jsonl"]
    inputs = materialize_inputs(args)
    files = list_files(inputs, exts)
    if not files:
        raise SystemExit(f"Girdi bulunamadi ({exts}): {args.input}")
    print(f"[data] {len(files)} dosya, mod={args.mode}")

    all_ids = []
    if args.mode == "pretrain":
        for doc in tqdm(iter_pretrain_docs(files), desc="kodlaniyor"):
            if not doc:
                continue
            all_ids.extend(tok.encode(doc))
            all_ids.append(tok.eot_id)
    else:
        for messages in tqdm(iter_chat_convos(files), desc="kodlaniyor"):
            ids = encode_chat(tok, messages, add_generation_prompt=False,
                              system=system)
            all_ids.extend(ids)
            all_ids.append(tok.eot_id)

    if not all_ids:
        raise SystemExit("Hic token uretilemedi (girdi bos olabilir).")

    min_split_tokens = max(1, args.min_split_tokens)
    if len(all_ids) < min_split_tokens and args.mode == "chat":
        original = list(all_ids)
        while len(all_ids) < min_split_tokens:
            all_ids.extend(original)
        all_ids = all_ids[:min_split_tokens]
        print(
            f"[data] chat verisi kucuk; smoke egitim icin "
            f"{len(original)} -> {len(all_ids)} token tekrarlandi"
        )

    arr = np.array(all_ids, dtype=dtype)
    n = len(arr)
    if n < min_split_tokens:
        raise SystemExit(
            f"Veri cok kucuk: {n} token. En az {min_split_tokens} token gerekli "
            "(daha uzun cevaplar veya daha fazla sohbet ornegi ekle)."
        )
    if args.val_ratio <= 0 or n < min_split_tokens * 2:
        n_val = 0
        train, val = arr, arr
    else:
        n_val = max(min_split_tokens, int(n * args.val_ratio))
        n_val = min(n_val, n - min_split_tokens)
        train, val = arr[: n - n_val], arr[n - n_val:]

    os.makedirs(args.out, exist_ok=True)
    train.tofile(os.path.join(args.out, "train.bin"))
    if n_val:
        val.tofile(os.path.join(args.out, "val.bin"))
    else:
        # cok kucuk veri: val = train (sadece duman testi icin)
        train.tofile(os.path.join(args.out, "val.bin"))

    meta = {
        "mode": args.mode,
        "dtype": np.dtype(dtype).name,
        "vocab_size": tok.vocab_size,
        "n_tokens_total": int(n),
        "n_train": int(len(train)),
        "n_val": int(n_val),
    }
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[data] toplam {n:,} token  (train={len(train):,}, val={n_val:,})")
    print(f"[data] yazildi -> {args.out}/  (train.bin, val.bin, meta.json)")


if __name__ == "__main__":
    main()
