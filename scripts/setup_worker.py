"""
scripts/setup_worker.py
=======================
Worker kurulumunu tek komuta indirir.

Ornek:
    python scripts/setup_worker.py --server http://SUNUCU_IP:8000 --token TOKEN --name macbook-pro

Tekrar baslatma:
    python scripts/setup_worker.py
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / ".yerellm_worker.env"


def read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def env_or_config(config: dict[str, str], key: str, default: str = "") -> str:
    return os.environ.get(key) or config.get(key) or default


def env_bool(value: str | bool, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on", "evet"}


def write_env_file(path: Path, values: dict[str, str]) -> None:
    lines = [
        "# yerelLLM worker ayarlari",
        "# Bu dosya scripts/setup_worker.py tarafindan yazilir.",
    ]
    for key, value in values.items():
        safe = str(value).replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'{key}="{safe}"')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def request_json(url: str, token: str = "", timeout: int = 5) -> tuple[int, dict]:
    headers = {"X-YerelLM-Token": token} if token else {}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return response.status, json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = {"error": body}
        return exc.code, payload


def check_server(server: str, token: str) -> bool:
    ok = True
    server = server.rstrip("/")
    try:
        status, payload = request_json(f"{server}/api/health")
        if status != 200:
            print(f"[setup-worker] health beklenmeyen durum: HTTP {status} {payload}")
            ok = False
        else:
            ready = "hazir" if payload.get("ready") else "model bekliyor"
            print(f"[setup-worker] sunucu erisilebilir: {ready}")

        status, payload = request_json(f"{server}/api/nodes", token=token)
        if status == 401:
            print("[setup-worker] API token eksik veya hatali; worker kayit olamaz.")
            ok = False
        elif status != 200:
            print(f"[setup-worker] node API beklenmeyen durum: HTTP {status} {payload}")
            ok = False
    except Exception as exc:
        print(f"[setup-worker] sunucu kontrolu yapilamadi: {exc}")
        ok = False
    return ok


def main() -> int:
    config_path = Path(os.environ.get("YERELLM_WORKER_CONFIG", DEFAULT_CONFIG))
    existing = read_env_file(config_path)

    parser = argparse.ArgumentParser(description="yerelLLM worker kur ve baslat")
    parser.add_argument("--server", default=env_or_config(existing, "YERELLM_SERVER_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--token", default=env_or_config(existing, "YERELLM_API_TOKEN", ""))
    parser.add_argument("--name", default=env_or_config(existing, "YERELLM_WORKER_NAME", socket.gethostname()))
    parser.add_argument("--node-id", default=env_or_config(existing, "YERELLM_NODE_ID", ""))
    parser.add_argument("--role", default=env_or_config(existing, "YERELLM_NODE_ROLE", "worker"))
    parser.add_argument("--repo", default=env_or_config(existing, "YERELLM_REPO", str(ROOT)))
    parser.add_argument("--device", default=env_or_config(existing, "YERELLM_DEVICE", "auto"),
                        choices=("auto", "cuda", "mps", "cpu"))
    parser.add_argument("--config", default=str(config_path))
    parser.add_argument("--heartbeat-sec", type=int, default=int(env_or_config(existing, "YERELLM_HEARTBEAT_SEC", "10")))
    parser.add_argument("--training-jobs", action=argparse.BooleanOptionalAction,
                        default=env_bool(env_or_config(existing, "YERELLM_ALLOW_TRAINING_JOBS", "true")),
                        help="Yapilandirilmis egitim/senkron islerini ac veya kapat.")
    parser.add_argument("--allow-remote-commands", action="store_true",
                        default=env_bool(env_or_config(existing, "YERELLM_ALLOW_REMOTE_COMMANDS", "false")))
    parser.add_argument("--save-only", action="store_true", help="Ayar dosyasini yaz, worker'i baslatma.")
    parser.add_argument("--skip-check", action="store_true", help="Sunucu/API kontrolunu atla.")
    args = parser.parse_args()

    values = {
        "YERELLM_SERVER_URL": args.server.rstrip("/"),
        "YERELLM_API_TOKEN": args.token,
        "YERELLM_WORKER_NAME": args.name,
        "YERELLM_NODE_ID": args.node_id,
        "YERELLM_NODE_ROLE": args.role,
        "YERELLM_REPO": str(Path(args.repo).resolve()),
        "YERELLM_DEVICE": args.device,
        "YERELLM_HEARTBEAT_SEC": str(max(3, args.heartbeat_sec)),
        "YERELLM_ALLOW_TRAINING_JOBS": "true" if args.training_jobs else "false",
        "YERELLM_ALLOW_REMOTE_COMMANDS": "true" if args.allow_remote_commands else "false",
    }
    config_path = Path(args.config)
    write_env_file(config_path, values)
    print(f"[setup-worker] ayarlar yazildi: {config_path}")

    if not args.skip_check:
        check_server(values["YERELLM_SERVER_URL"], values["YERELLM_API_TOKEN"])

    command = [sys.executable, "-m", "worker.local_node", "--config", str(config_path)]
    print("[setup-worker] baslatma komutu:")
    print(" ".join(command))

    if args.save_only:
        return 0
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
