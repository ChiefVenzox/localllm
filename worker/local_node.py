from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from typing import Any

WORKER_VERSION = "1.1"
DEFAULT_CHAT_MIN_SPLIT_TOKENS = 2050
TRAINING_PATCH_FILES = (
    ("chat_template.py", "chat_template.py"),
    ("config.py", "config.py"),
    ("generate.py", "generate.py"),
    ("quick_intents.py", "quick_intents.py"),
    ("web_search.py", "web_search.py"),
    ("train.py", "train.py"),
    ("data/__init__.py", "data/__init__.py"),
    ("data/make_chat_tokens.py", "data/make_chat_tokens.py"),
    ("data/prepare_data.py", "data/prepare_data.py"),
    ("model/__init__.py", "model/__init__.py"),
    ("model/gpt.py", "model/gpt.py"),
    ("tokenizer/__init__.py", "tokenizer/__init__.py"),
    ("tokenizer/tokenizer.json", "tokenizer/tokenizer.json"),
    ("worker/__init__.py", "worker/__init__.py"),
    ("worker/local_node.py", "worker/local_node.py"),
    ("server/__init__.py", "server/__init__.py"),
    ("server/app.py", "server/app.py"),
    ("server/node_registry.py", "server/node_registry.py"),
    ("server/static/index.html", "server/static/index.html"),
    ("server/static/cluster.html", "server/static/cluster.html"),
    ("scripts/lan_train.sh", "scripts/lan_train.sh"),
    ("scripts/lan_train.ps1", "scripts/lan_train.ps1"),
    ("scripts/system_check.py", "scripts/system_check.py"),
    ("scripts/vram_probe.py", "scripts/vram_probe.py"),
)


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "evet"}


def load_env_file(path: str) -> None:
    if not path or not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, value)


def total_ram_bytes() -> int | None:
    try:
        if sys.platform == "win32":
            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MemoryStatus()
            stat.dwLength = ctypes.sizeof(MemoryStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return int(stat.ullTotalPhys)
        if sys.platform == "darwin":
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
            return int(out)
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        return int(page_size * pages)
    except Exception:
        return None


def detect_torch() -> dict[str, Any]:
    info: dict[str, Any] = {"available": False, "cuda": False, "mps": False, "gpus": []}
    try:
        import torch
    except Exception as exc:
        info["error"] = str(exc)
        return info

    info["available"] = True
    info["version"] = getattr(torch, "__version__", None)
    info["cuda"] = bool(torch.cuda.is_available())
    if info["cuda"]:
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            info["gpus"].append(
                {
                    "index": idx,
                    "name": props.name,
                    "total_vram_bytes": int(props.total_memory),
                    "capability": f"{props.major}.{props.minor}",
                }
            )
    mps_backend = getattr(torch.backends, "mps", None)
    info["mps"] = bool(mps_backend and mps_backend.is_available())
    return info


def detect_nvidia_smi() -> dict[str, Any]:
    info: dict[str, Any] = {"present": False, "ok": False, "gpus": []}
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=5,
        ).strip()
    except FileNotFoundError:
        if sys.platform.startswith("linux"):
            try:
                lspci = subprocess.check_output(["lspci"], text=True, stderr=subprocess.DEVNULL, timeout=5)
                if "NVIDIA" in lspci.upper():
                    info["present"] = True
                    info["error"] = "nvidia-smi yok; NVIDIA driver kurulumu gerekli olabilir"
            except Exception:
                pass
        return info
    except subprocess.CalledProcessError as exc:
        info["present"] = True
        info["error"] = (exc.output or str(exc)).strip()
        return info
    except Exception as exc:
        info["error"] = str(exc)
        return info

    info["present"] = True
    info["ok"] = True
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3:
            try:
                memory_mb = int(float(parts[1]))
            except ValueError:
                memory_mb = 0
            info["gpus"].append(
                {
                    "name": parts[0],
                    "memory_total_mb": memory_mb,
                    "driver_version": parts[2],
                }
            )
    return info


def preferred_device(torch_info: dict[str, Any]) -> str:
    if torch_info.get("cuda"):
        return "cuda"
    if torch_info.get("mps"):
        return "mps"
    return "cpu"


def read_or_create_node_id(path: str, forced_node_id: str = "") -> str:
    if forced_node_id:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(forced_node_id + "\n")
        return forced_node_id
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            value = handle.read().strip()
            if value:
                return value
    value = str(uuid.uuid4())
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(value + "\n")
    return value


def _auth_headers(api_token: str = "") -> dict[str, str]:
    headers: dict[str, str] = {}
    if api_token:
        headers["X-YerelLM-Token"] = api_token
    return headers


def post_json(server_url: str, path: str, payload: dict[str, Any], api_token: str = "") -> dict[str, Any]:
    url = server_url.rstrip("/") + path
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", **_auth_headers(api_token)}
    req = urllib.request.Request(
        url,
        data=body,
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def build_payload(args: argparse.Namespace, node_id: str) -> dict[str, Any]:
    torch_info = detect_torch()
    return {
        "node_id": node_id,
        "name": args.name or socket.gethostname(),
        "hostname": socket.gethostname(),
        "platform": platform.system(),
        "os": platform.platform(),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "repo_path": os.path.abspath(args.repo),
        "base_url": args.base_url,
        "worker_version": WORKER_VERSION,
        "node_role": args.node_role,
        "status": "online",
        "resources": {
            "cpu_count": os.cpu_count(),
            "ram_bytes": total_ram_bytes(),
            "torch": torch_info,
            "nvidia": detect_nvidia_smi(),
        },
        "capabilities": {
            "preferred_device": args.device if args.device != "auto" else preferred_device(torch_info),
            "can_train": True,
            "can_infer": True,
            "allow_remote_commands": bool(args.allow_remote_commands),
            "allow_training_jobs": bool(args.allow_training_jobs),
        },
    }


def heartbeat_payload(args: argparse.Namespace, node_id: str) -> dict[str, Any]:
    capabilities = {
        "can_train": True,
        "can_infer": True,
        "allow_remote_commands": bool(args.allow_remote_commands),
        "allow_training_jobs": bool(args.allow_training_jobs),
    }
    if args.device != "auto":
        capabilities["preferred_device"] = args.device
    return {
        "node_id": node_id,
        "name": args.name or socket.gethostname(),
        "hostname": socket.gethostname(),
        "platform": platform.system(),
        "os": platform.platform(),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "repo_path": os.path.abspath(args.repo),
        "base_url": args.base_url,
        "worker_version": WORKER_VERSION,
        "node_role": args.node_role,
        "status": "online",
        "capabilities": capabilities,
    }


def send_heartbeat(args: argparse.Namespace, node_id: str) -> None:
    post_json(args.server, f"/api/nodes/{node_id}/heartbeat", heartbeat_payload(args, node_id), args.api_token)


def command_heartbeat_loop(args: argparse.Namespace, node_id: str, stop_event: threading.Event, title: str) -> None:
    interval = max(3, args.heartbeat_sec)
    while not stop_event.wait(interval):
        try:
            send_heartbeat(args, node_id)
            print(f"[local_node] heartbeat: busy ({title})")
        except Exception as exc:
            print(f"[local_node] busy heartbeat gonderilemedi: {exc}")


def tail_output(value: str, limit: int = 12000) -> str:
    return value[-limit:]


def resolve_repo_path(repo: str, value: str) -> str:
    if os.path.isabs(value):
        return os.path.abspath(value)
    return os.path.abspath(os.path.join(repo, value))


def bool_from_payload(value: Any, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() not in {"0", "false", "no", "off", "hayir", "hayır"}


def quick_mode_from_payload(value: Any) -> str:
    mode = str(value or os.environ.get("YERELLM_CHAT_QUICK_MODE", "model_first")).strip().lower()
    return mode if mode in {"off", "none", "model", "model_only", "tools", "tool", "model_first", "safe", "legacy", "quick", "quick_first"} else "model_first"


def clamp_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def absolute_server_url(server_url: str, value: str) -> str:
    if value.startswith("http://") or value.startswith("https://"):
        return value
    return server_url.rstrip("/") + "/" + value.lstrip("/")


def open_url(url: str, args: argparse.Namespace, timeout: int):
    req = urllib.request.Request(url, headers=_auth_headers(args.api_token))
    return urllib.request.urlopen(req, timeout=timeout)


def sync_training_files(command: dict[str, Any], args: argparse.Namespace) -> list[str]:
    if not bool_from_payload(command.get("sync_patch"), True):
        return []

    repo = os.path.abspath(args.repo)
    base = str(command.get("patch_base_url") or "/patches/localllm")
    updated: list[str] = []
    for url_path, local_path in TRAINING_PATCH_FILES:
        url = absolute_server_url(args.server, f"{base.rstrip('/')}/{url_path}")
        destination = resolve_repo_path(repo, local_path)
        try:
            with open_url(url, args, timeout=15) as response:
                content = response.read()
            old = b""
            if os.path.exists(destination):
                with open(destination, "rb") as handle:
                    old = handle.read()
            if old != content:
                os.makedirs(os.path.dirname(destination), exist_ok=True)
                with open(destination, "wb") as handle:
                    handle.write(content)
                updated.append(local_path)
        except Exception as exc:
            updated.append(f"{local_path}: sync hata ({exc})")
    return updated


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_entries(command: dict[str, Any]) -> list[dict[str, Any]]:
    manifest = command.get("artifact_manifest") or {}
    files = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(files, list):
        files = command.get("files")
    return [item for item in files or [] if isinstance(item, dict)]


def sync_artifact_files(command: dict[str, Any], args: argparse.Namespace) -> list[str]:
    repo = os.path.abspath(args.repo)
    manifest = command.get("artifact_manifest") or {}
    base = str(command.get("artifact_base_url") or "")
    if isinstance(manifest, dict) and not base:
        base = str(manifest.get("artifact_base_url") or manifest.get("base_url") or "")
    if not base:
        base = "/artifacts/localllm/ddp"

    results: list[str] = []
    for item in artifact_entries(command):
        local_path = str(item.get("local_path") or "").strip()
        url_path = str(item.get("url_path") or item.get("path") or "").strip()
        expected_sha = str(item.get("sha256") or "").strip().lower()
        expected_size = item.get("size")
        if not local_path or not url_path:
            results.append(f"gecersiz artifact girdisi: {item}")
            continue

        destination = resolve_repo_path(repo, local_path)
        try:
            if os.path.exists(destination) and expected_sha:
                current_sha = sha256_file(destination)
                if current_sha == expected_sha:
                    results.append(f"{local_path}: zaten guncel")
                    continue

            url = absolute_server_url(args.server, f"{base.rstrip('/')}/{url_path.lstrip('/')}")
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            tmp_path = destination + ".part"
            downloaded = 0
            with open_url(url, args, timeout=60) as response, open(tmp_path, "wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    downloaded += len(chunk)

            if expected_size is not None and downloaded != int(expected_size):
                raise RuntimeError(f"boyut uyusmadi: {downloaded} != {expected_size}")
            if expected_sha:
                actual_sha = sha256_file(tmp_path)
                if actual_sha != expected_sha:
                    raise RuntimeError(f"sha256 uyusmadi: {actual_sha} != {expected_sha}")

            os.replace(tmp_path, destination)
            results.append(f"{local_path}: indirildi ({downloaded} bayt)")
        except Exception as exc:
            try:
                if os.path.exists(destination + ".part"):
                    os.remove(destination + ".part")
            except Exception:
                pass
            results.append(f"{local_path}: sync hata ({exc})")
    return results


def run_sync_artifacts(command: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if not args.allow_training_jobs:
        return {
            "status": "refused",
            "exit_code": None,
            "output": "",
            "error": "--allow-training-jobs olmadan artifact senkronu calistirilmaz.",
        }

    started = time.time()
    logs: list[str] = []
    synced = sync_training_files(command, args)
    if synced:
        logs.append("[sync] panel patch senkronu: " + ", ".join(synced))
    files = artifact_entries(command)
    if not files:
        return {
            "status": "failed",
            "exit_code": None,
            "output": "\n".join(logs),
            "error": "artifact manifest bos",
            "duration_sec": round(time.time() - started, 2),
        }
    results = sync_artifact_files(command, args)
    logs.extend(f"[sync] {line}" for line in results)
    failed = [line for line in results if "sync hata" in line or "gecersiz" in line]
    return {
        "status": "failed" if failed else "done",
        "exit_code": 1 if failed else 0,
        "output": tail_output("\n".join(logs)),
        "error": "; ".join(failed),
        "duration_sec": round(time.time() - started, 2),
    }


def run_sync_patch(command: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if not args.allow_training_jobs and not args.allow_remote_commands:
        return {
            "status": "refused",
            "exit_code": None,
            "output": "",
            "error": "Patch senkronu icin --allow-training-jobs veya --allow-remote-commands gerekli.",
        }

    started = time.time()
    results = sync_training_files(command, args)
    if not results:
        results = ["degisen patch dosyasi yok"]
    failed = [line for line in results if "sync hata" in line]
    return {
        "status": "failed" if failed else "done",
        "exit_code": 1 if failed else 0,
        "output": tail_output("\n".join(f"[sync] {line}" for line in results)),
        "error": "; ".join(failed),
        "duration_sec": round(time.time() - started, 2),
    }


def run_process(argv: list[str], cwd: str, timeout_sec: int) -> dict[str, Any]:
    started = time.time()
    try:
        proc = subprocess.run(
            argv,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=max(1, timeout_sec),
        )
        output = (proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
        return {
            "status": "done" if proc.returncode == 0 else "failed",
            "exit_code": proc.returncode,
            "output": tail_output(output),
            "duration_sec": round(time.time() - started, 2),
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return {
            "status": "failed",
            "exit_code": None,
            "output": tail_output(stdout + (("\n[stderr]\n" + stderr) if stderr else "")),
            "error": f"timeout: {timeout_sec}s",
            "duration_sec": round(time.time() - started, 2),
        }
    except Exception as exc:
        return {
            "status": "failed",
            "exit_code": None,
            "output": "",
            "error": str(exc),
            "duration_sec": round(time.time() - started, 2),
        }


def run_command(command: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cmd = str(command.get("command") or "")
    cwd = str(command.get("cwd") or args.repo)
    if not os.path.isabs(cwd):
        cwd = os.path.abspath(os.path.join(args.repo, cwd))

    started = time.time()
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=max(1, args.command_timeout_sec),
        )
        output = (proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
        return {
            "status": "done" if proc.returncode == 0 else "failed",
            "exit_code": proc.returncode,
            "output": tail_output(output),
            "duration_sec": round(time.time() - started, 2),
        }
    except subprocess.TimeoutExpired as exc:
        output = ((exc.stdout or "") + "\n" + (exc.stderr or "")) if isinstance(exc.stdout, str) else ""
        return {
            "status": "failed",
            "exit_code": None,
            "output": tail_output(output),
            "error": f"timeout: {args.command_timeout_sec}s",
            "duration_sec": round(time.time() - started, 2),
        }
    except Exception as exc:
        return {
            "status": "failed",
            "exit_code": None,
            "output": "",
            "error": str(exc),
            "duration_sec": round(time.time() - started, 2),
        }


def append_chat_sample(command: dict[str, Any], args: argparse.Namespace) -> str:
    sample = command.get("sample") or {}
    messages = sample.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("chat_train sample.messages bos veya gecersiz")

    clean_messages = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").strip()
        content = str(message.get("content") or "").strip()
        if role and content:
            clean_messages.append({"role": role, "content": content})
    if not clean_messages:
        raise ValueError("chat_train icinde yazilabilir mesaj yok")

    repo = os.path.abspath(args.repo)
    data_dir = resolve_repo_path(repo, str(command.get("data_dir") or "data/chat_remote"))
    os.makedirs(data_dir, exist_ok=True)
    sample_path = os.path.join(data_dir, "server_chat.jsonl")
    row = {"messages": clean_messages}
    with open(sample_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return sample_path


def run_chat_train(command: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if not args.allow_training_jobs:
        return {
            "status": "refused",
            "exit_code": None,
            "output": "",
            "error": "--allow-training-jobs olmadan sohbet egitimi calistirilmaz.",
        }

    started = time.time()
    repo = os.path.abspath(args.repo)
    logs: list[str] = []
    try:
        synced = sync_training_files(command, args)
        if synced:
            logs.append("[chat_train] panel patch senkronu: " + ", ".join(synced))
        prepared_data = str(
            command.get("prepared_data") or command.get("prepared_data_dir") or ""
        ).strip()
        use_prepared_data = bool(prepared_data)

        if use_prepared_data:
            logs.append(f"[chat_train] hazir veri secildi: {prepared_data}")
        else:
            sample_path = append_chat_sample(command, args)
            logs.append(f"[chat_train] ornek yazildi: {sample_path}")

        if not bool_from_payload(command.get("auto_train"), True):
            return {
                "status": "done",
                "exit_code": 0,
                "output": "\n".join(logs),
                "duration_sec": round(time.time() - started, 2),
            }

        if use_prepared_data:
            out_dir = resolve_repo_path(repo, prepared_data)
            missing = [
                name for name in ("meta.json", "train.bin", "val.bin")
                if not os.path.exists(os.path.join(out_dir, name))
            ]
            if missing:
                return {
                    "status": "failed",
                    "exit_code": None,
                    "output": tail_output("\n".join(logs)),
                    "error": f"hazir veri eksik: {', '.join(missing)} ({out_dir})",
                    "duration_sec": round(time.time() - started, 2),
                }
        else:
            data_dir = resolve_repo_path(repo, str(command.get("data_dir") or "data/chat_remote"))
            out_dir = resolve_repo_path(repo, str(command.get("out_dir") or "data/chat_remote_bin"))
        ckpt = resolve_repo_path(repo, str(command.get("ckpt") or "checkpoints/ckpt.pt"))
        max_steps = clamp_int(command.get("max_steps"), 80, 1, 5000)
        min_split_tokens = clamp_int(
            command.get("min_split_tokens"),
            DEFAULT_CHAT_MIN_SPLIT_TOKENS,
            258,
            20000,
        )
        device = str(command.get("device") or args.device)
        if device == "auto":
            device = preferred_device(detect_torch())

        if not use_prepared_data:
            prepare_cmd = [
                sys.executable,
                "-m",
                "data.prepare_data",
                "--mode",
                "chat",
                "--input",
                data_dir,
                "--out",
                out_dir,
                "--min-split-tokens",
                str(min_split_tokens),
                "--no-chat-sync",
            ]
            logs.append(f"[chat_train] veri hazirlaniyor: min_split_tokens={min_split_tokens}")
            prepare = run_process(prepare_cmd, repo, min(args.training_timeout_sec, 900))
            logs.append(prepare.get("output", ""))
            if prepare.get("status") != "done":
                return {
                    "status": "failed",
                    "exit_code": prepare.get("exit_code"),
                    "output": tail_output("\n".join(logs)),
                    "error": prepare.get("error") or "veri hazirlama basarisiz",
                    "duration_sec": round(time.time() - started, 2),
                }

        preset = str(command.get("preset") or "tiny-30m")
        train_cmd = [sys.executable, "train.py"]
        use_adapter = bool_from_payload(command.get("adapter"), True)
        if os.path.exists(ckpt):
            train_cmd.extend(["--resume", ckpt])
            if use_adapter:
                adapter_out = resolve_repo_path(repo, str(command.get("adapter_out") or "checkpoints"))
                train_cmd.extend(["--adapter", "--adapter-out", adapter_out])
                adapter_resume_value = str(command.get("adapter_resume") or "").strip()
                adapter_resume = resolve_repo_path(repo, adapter_resume_value) if adapter_resume_value else ""
                if adapter_resume and os.path.exists(adapter_resume):
                    train_cmd.extend(["--adapter-resume", adapter_resume])
        else:
            logs.append(f"[chat_train] checkpoint bulunamadi, sifirdan preset kullaniliyor: {preset}")
            train_cmd.extend(["--preset", preset])

        train_cmd.extend([
            "--data",
            out_dir,
            "--max-steps",
            str(max_steps),
            "--reset-best",
            "--device",
            device,
        ])
        batch_size = command.get("batch_size")
        if batch_size is not None:
            train_cmd.extend(["--batch-size", str(clamp_int(batch_size, 1, 1, 128))])
        grad_accum = command.get("grad_accum")
        if grad_accum is not None:
            train_cmd.extend(["--grad-accum", str(clamp_int(grad_accum, 1, 1, 128))])
        adapter_lr = command.get("adapter_lr")
        if adapter_lr is not None:
            train_cmd.extend(["--adapter-lr", str(adapter_lr)])
        adapter_dim = command.get("adapter_dim")
        if adapter_dim is not None:
            train_cmd.extend(["--adapter-dim", str(clamp_int(adapter_dim, 64, 1, 4096))])
        logs.append(
            f"[chat_train] egitim basliyor: max_steps={max_steps}, "
            f"device={device}, adapter={use_adapter and os.path.exists(ckpt)}"
        )
        train = run_process(train_cmd, repo, args.training_timeout_sec)
        logs.append(train.get("output", ""))
        return {
            "status": train.get("status", "failed"),
            "exit_code": train.get("exit_code"),
            "output": tail_output("\n".join(logs)),
            "error": train.get("error", ""),
            "duration_sec": round(time.time() - started, 2),
        }
    except Exception as exc:
        logs.append(f"[chat_train] hata: {exc}")
        return {
            "status": "failed",
            "exit_code": None,
            "output": tail_output("\n".join(logs)),
            "error": str(exc),
            "duration_sec": round(time.time() - started, 2),
        }


def run_ddp_train(command: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if not args.allow_training_jobs:
        return {
            "status": "refused",
            "exit_code": None,
            "output": "",
            "error": "--allow-training-jobs olmadan DDP egitimi calistirilmaz.",
        }

    started = time.time()
    repo = os.path.abspath(args.repo)
    logs: list[str] = []
    try:
        synced = sync_training_files(command, args)
        if synced:
            logs.append("[ddp] panel patch senkronu: " + ", ".join(synced))
        if bool_from_payload(command.get("sync_artifacts"), True):
            artifact_results = sync_artifact_files(command, args)
            logs.extend(f"[ddp] {line}" for line in artifact_results)
            failed = [line for line in artifact_results if "sync hata" in line or "gecersiz" in line]
            if failed:
                return {
                    "status": "failed",
                    "exit_code": 1,
                    "output": tail_output("\n".join(logs)),
                    "error": "; ".join(failed),
                    "duration_sec": round(time.time() - started, 2),
                }

        rank = clamp_int(command.get("rank"), 0, 0, 1024)
        local_rank = clamp_int(command.get("local_rank"), 0, 0, 128)
        nnodes = clamp_int(command.get("nnodes"), 1, 1, 1024)
        nproc_per_node = clamp_int(command.get("nproc_per_node"), 1, 1, 128)
        world_size = clamp_int(command.get("world_size"), nnodes * nproc_per_node, 1, 131072)
        master_addr = str(command.get("master_addr") or "").strip()
        master_port = str(clamp_int(command.get("master_port"), 29500, 1024, 65535))
        if not master_addr:
            raise ValueError("master_addr bos")

        preset = str(command.get("preset") or "tiny-30m")
        data_path = resolve_repo_path(repo, str(command.get("data_path") or "data/ddp_bin"))
        device = str(command.get("device") or args.device)
        if device == "auto":
            device = preferred_device(detect_torch())
        backend = str(command.get("dist_backend") or "gloo")
        timeout_minutes = clamp_int(command.get("dist_timeout_minutes"), 60, 1, 24 * 60)

        train_cmd = [
            sys.executable,
            "train.py",
            "--preset",
            preset,
            "--data",
            data_path,
            "--device",
            device,
            "--dist-backend",
            backend,
            "--dist-timeout-minutes",
            str(timeout_minutes),
        ]
        max_steps = command.get("max_steps")
        if max_steps is not None:
            train_cmd.extend(["--max-steps", str(clamp_int(max_steps, 20, 1, 500000))])
        batch_size = command.get("batch_size")
        if batch_size is not None:
            train_cmd.extend(["--batch-size", str(clamp_int(batch_size, 1, 1, 1024))])
        grad_accum = command.get("grad_accum")
        if grad_accum is not None:
            train_cmd.extend(["--grad-accum", str(clamp_int(grad_accum, 1, 1, 4096))])
        resume = str(command.get("resume") or "").strip()
        if resume:
            train_cmd.extend(["--resume", resolve_repo_path(repo, resume)])
        out_dir = str(command.get("out_dir") or "").strip()
        if out_dir:
            train_cmd.extend(["--out", resolve_repo_path(repo, out_dir)])
        if bool_from_payload(command.get("adapter"), False) or bool_from_payload(command.get("adapter_mode"), False):
            train_cmd.append("--adapter")
            adapter_dim = command.get("adapter_dim")
            if adapter_dim is not None:
                train_cmd.extend(["--adapter-dim", str(clamp_int(adapter_dim, 64, 1, 4096))])
            adapter_out = str(command.get("adapter_out") or "").strip()
            if adapter_out:
                train_cmd.extend(["--adapter-out", resolve_repo_path(repo, adapter_out)])
            adapter_resume = str(command.get("adapter_resume") or "").strip()
            if adapter_resume:
                train_cmd.extend(["--adapter-resume", resolve_repo_path(repo, adapter_resume)])
            adapter_lr = command.get("adapter_lr")
            if adapter_lr is not None:
                train_cmd.extend(["--adapter-lr", str(float(adapter_lr))])
        if bool_from_payload(command.get("reset_best"), False):
            train_cmd.append("--reset-best")

        env = os.environ.copy()
        env.update({
            "MASTER_ADDR": master_addr,
            "MASTER_PORT": master_port,
            "WORLD_SIZE": str(world_size),
            "RANK": str(rank),
            "LOCAL_RANK": str(local_rank),
            "USE_LIBUV": "0",
        })
        logs.append(
            f"[ddp] basliyor: rank={rank}/{world_size} master={master_addr}:{master_port} "
            f"backend={backend} device={device} data={data_path}"
        )
        proc = subprocess.run(
            train_cmd,
            cwd=repo,
            env=env,
            text=True,
            capture_output=True,
            timeout=max(1, args.training_timeout_sec),
        )
        output = (proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
        logs.append(output)
        return {
            "status": "done" if proc.returncode == 0 else "failed",
            "exit_code": proc.returncode,
            "output": tail_output("\n".join(logs)),
            "error": "",
            "duration_sec": round(time.time() - started, 2),
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        logs.append(stdout + (("\n[stderr]\n" + stderr) if stderr else ""))
        return {
            "status": "failed",
            "exit_code": None,
            "output": tail_output("\n".join(logs)),
            "error": f"timeout: {args.training_timeout_sec}s",
            "duration_sec": round(time.time() - started, 2),
        }
    except Exception as exc:
        logs.append(f"[ddp] hata: {exc}")
        return {
            "status": "failed",
            "exit_code": None,
            "output": tail_output("\n".join(logs)),
            "error": str(exc),
            "duration_sec": round(time.time() - started, 2),
        }


def clean_chat_messages(messages: Any) -> list[dict[str, str]]:
    if not isinstance(messages, list):
        return []
    clean: list[dict[str, str]] = []
    for message in messages[-24:]:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").strip()
        content = str(message.get("content") or "").strip()
        if role in {"system", "user", "assistant"} and content:
            clean.append({"role": role, "content": content})
    return clean[-20:]


def run_chat_infer(command: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if not args.allow_training_jobs:
        return {
            "status": "refused",
            "exit_code": None,
            "output": "",
            "error": "--allow-training-jobs olmadan worker inference calistirilmaz.",
        }

    started = time.time()
    repo = os.path.abspath(args.repo)
    logs: list[str] = []
    try:
        synced = sync_training_files(command, args)
        if synced:
            logs.append("[infer] panel patch senkronu: " + ", ".join(synced))
        if bool_from_payload(command.get("sync_artifacts"), True) and artifact_entries(command):
            artifact_results = sync_artifact_files(command, args)
            logs.extend(f"[infer] {line}" for line in artifact_results)
            failed = [line for line in artifact_results if "sync hata" in line or "gecersiz" in line]
            if failed:
                return {
                    "status": "failed",
                    "exit_code": 1,
                    "output": tail_output("\n".join(logs)),
                    "error": "; ".join(failed),
                    "duration_sec": round(time.time() - started, 2),
                }

        messages = clean_chat_messages(command.get("messages"))
        if not any(message["role"] == "user" for message in messages):
            raise ValueError("messages icinde user mesaji yok")
        system = command.get("system")
        system_text = None if system is None else str(system)
        ckpt = resolve_repo_path(repo, str(command.get("ckpt") or command.get("resume") or "checkpoints/ckpt.pt"))
        adapter_value = str(command.get("adapter") or command.get("adapter_path") or "auto").strip()
        adapter_path = adapter_value
        if adapter_value and adapter_value.lower() not in {"auto", "off", "none", "false", "0"}:
            adapter_path = resolve_repo_path(repo, adapter_value)
        tokenizer_path = resolve_repo_path(repo, str(command.get("tokenizer") or "tokenizer/tokenizer.json"))
        device = str(command.get("device") or args.device)
        if device == "auto":
            device = preferred_device(detect_torch())
        max_new_tokens = clamp_int(command.get("max_new_tokens"), 180, 1, 1024)
        temperature = float(command.get("temperature") if command.get("temperature") is not None else 0.35)
        top_k = clamp_int(command.get("top_k"), 40, 0, 500)
        top_p = float(command.get("top_p") if command.get("top_p") is not None else 0.95)
        quick_mode = quick_mode_from_payload(command.get("quick_mode"))

        sys.path.insert(0, repo)
        import torch
        from generate import chat_stream, load_model
        from tokenizer import load_tokenizer
        import quick_intents

        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        quick = None
        if last_user.strip() and quick_mode in {"tools", "tool", "model_first", "safe"}:
            for name in ("arithmetic_reply", "weather_reply"):
                fn = getattr(quick_intents, name, None)
                if not fn:
                    continue
                quick = fn(last_user)
                if quick:
                    break
        elif last_user.strip() and quick_mode in {"legacy", "quick", "quick_first"}:
            if hasattr(quick_intents, "contextual_reply"):
                quick = quick_intents.contextual_reply(messages)
            if quick is None:
                quick = quick_intents.quick_reply(last_user)
        if quick:
            answer = quick
            source = "quick_intents"
        else:
            tok = load_tokenizer(tokenizer_path)
            model, cfg = load_model(ckpt, device, adapter_path=adapter_path)
            chat_messages = [m for m in messages if m["role"] != "system"]

            def generate_answer(temp: float, k: int, p: float) -> str:
                pieces = list(chat_stream(
                    model,
                    tok,
                    chat_messages,
                    device,
                    system=system_text,
                    max_new_tokens=max_new_tokens,
                    temperature=temp,
                    top_k=k,
                    top_p=p,
                ))
                return "".join(pieces).strip()

            answer = generate_answer(temperature, top_k, top_p)
            if hasattr(quick_intents, "is_bad_model_reply") and quick_intents.is_bad_model_reply(answer, last_user):
                retry = generate_answer(min(temperature, 0.25), min(top_k, 20), min(top_p, 0.85))
                if retry and not quick_intents.is_bad_model_reply(retry, last_user):
                    answer = retry
                elif quick_mode in {"legacy", "quick", "quick_first"} and hasattr(quick_intents, "fallback_reply"):
                    answer = quick_intents.fallback_reply(last_user)
                elif hasattr(quick_intents, "greeting_reply") and quick_intents.greeting_reply(last_user):
                    answer = quick_intents.greeting_reply(last_user)
                else:
                    answer = (
                        "Bu soruya worker modeli sağlam bir cevap üretemedi. Hazır hafıza/fallback "
                        "cevabı göstermiyorum; daha kapsamlı adapter eğitimi gerekiyor."
                    )
                logs.append("[infer] bozuk sohbet cevabi filtrelendi")
            adapter_loaded = getattr(model, "adapter_path", "")
            source = f"ckpt={ckpt}" + (f", adapter={adapter_loaded}" if adapter_loaded else "")
            del model
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
        logs.append(f"[infer] cevap uretildi: {len(answer)} karakter ({source})")
        return {
            "status": "done",
            "exit_code": 0,
            "output": tail_output("\n".join(logs)),
            "answer": answer,
            "duration_sec": round(time.time() - started, 2),
        }
    except Exception as exc:
        logs.append(f"[infer] hata: {exc}")
        return {
            "status": "failed",
            "exit_code": None,
            "output": tail_output("\n".join(logs)),
            "error": str(exc),
            "duration_sec": round(time.time() - started, 2),
        }


def run_polled_command(command: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    command_type = str(command.get("type") or "shell")
    if command_type == "sync_patch":
        return run_sync_patch(command, args)
    if command_type == "sync_artifacts":
        return run_sync_artifacts(command, args)
    if command_type == "ddp_train":
        return run_ddp_train(command, args)
    if command_type == "chat_infer":
        return run_chat_infer(command, args)
    if command_type == "chat_train":
        return run_chat_train(command, args)
    if command_type != "shell":
        return {
            "status": "refused",
            "exit_code": None,
            "output": "",
            "error": f"bilinmeyen komut tipi: {command_type}",
        }
    if not args.allow_remote_commands:
        return {
            "status": "refused",
            "exit_code": None,
            "output": "",
            "error": "--allow-remote-commands olmadan shell komutu calistirilmaz.",
        }
    return run_command(command, args)


def poll_and_run_commands(args: argparse.Namespace, node_id: str):
    if not args.allow_remote_commands and not args.allow_training_jobs:
        return

    try:
        result = post_json(args.server, f"/api/nodes/{node_id}/commands/poll", {"limit": 1}, args.api_token)
    except Exception as exc:
        print(f"[local_node] komut kuyruğu okunamadi: {exc}")
        return

    for command in result.get("commands", []):
        command_id = command.get("id")
        title = command.get("title") or command_id
        print(f"[local_node] komut calisiyor: {title}")
        stop_event = threading.Event()
        heartbeat_thread = threading.Thread(
            target=command_heartbeat_loop,
            args=(args, node_id, stop_event, str(title)),
            daemon=True,
        )
        heartbeat_thread.start()
        try:
            res = run_polled_command(command, args)
        finally:
            stop_event.set()
            heartbeat_thread.join(timeout=1)
            try:
                send_heartbeat(args, node_id)
            except Exception as exc:
                print(f"[local_node] komut sonrasi heartbeat gonderilemedi: {exc}")
        try:
            post_json(args.server, f"/api/commands/{command_id}/status", res, args.api_token)
        except Exception as exc:
            print(f"[local_node] komut sonucu gonderilemedi: {exc}")
        print(f"[local_node] komut bitti: {title} -> {res.get('status')}")


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=os.environ.get("YERELLM_WORKER_CONFIG", ".yerellm_worker.env"),
                     help="worker ayar dosyasi")
    pre_args, remaining = pre.parse_known_args()
    load_env_file(pre_args.config)

    parser = argparse.ArgumentParser(description="yerelLLM merkezi kayit worker'i", parents=[pre])
    parser.set_defaults(config=pre_args.config)
    parser.add_argument(
        "--server",
        default=os.environ.get("YERELLM_SERVER_URL") or os.environ.get("YERELLM_SERVER") or "",
        help="Merkezi FastAPI panel/API URL'i, ornek: http://192.168.1.121:8000",
    )
    parser.add_argument("--name", default=os.environ.get("YERELLM_WORKER_NAME", ""),
                        help="Panelde gorunecek cihaz adi")
    parser.add_argument("--node-id", default=os.environ.get("YERELLM_NODE_ID", ""),
                        help="Sabit node_id (server-local gibi)")
    parser.add_argument("--node-role", default=os.environ.get("YERELLM_NODE_ROLE", ""),
                        help="Panel icin rol etiketi (server-local gibi)")
    parser.add_argument("--repo", default=os.environ.get("YERELLM_REPO", "."),
                        help="Bu makinedeki localllm repo yolu")
    parser.add_argument("--device", default=os.environ.get("YERELLM_DEVICE", "auto"),
                        choices=("auto", "cuda", "mps", "cpu"))
    parser.add_argument("--base-url", default=os.environ.get("YERELLM_BASE_URL", ""),
                        help="Bu node'un kendi yerel UI/API URL'i varsa")
    parser.add_argument("--api-token", default=os.environ.get("YERELLM_API_TOKEN", ""),
                        help="Merkezi API icin X-YerelLM-Token/Bearer token")
    parser.add_argument("--node-id-file", default=os.environ.get("YERELLM_NODE_ID_FILE", ".yerellm_node_id"))
    parser.add_argument("--heartbeat-sec", type=int, default=int(os.environ.get("YERELLM_HEARTBEAT_SEC", "10")))
    parser.add_argument("--allow-remote-commands", action=argparse.BooleanOptionalAction,
                        default=env_bool("YERELLM_ALLOW_REMOTE_COMMANDS", False),
                        help="Merkezi panelden gelen shell komutlarini bu makinede calistir.")
    parser.add_argument("--allow-training-jobs", action=argparse.BooleanOptionalAction,
                        default=env_bool("YERELLM_ALLOW_TRAINING_JOBS", False),
                        help="Merkezi panelden gelen sohbet egitimi islerini bu makinede calistir.")
    parser.add_argument("--command-timeout-sec", type=int, default=int(os.environ.get("YERELLM_COMMAND_TIMEOUT_SEC", "3600")))
    parser.add_argument("--training-timeout-sec", type=int, default=int(os.environ.get("YERELLM_TRAINING_TIMEOUT_SEC", "7200")))
    args = parser.parse_args(remaining)

    if not args.server:
        raise SystemExit("--server gerekli veya .yerellm_worker.env icinde YERELLM_SERVER_URL tanimli olmali.")

    node_id_path = os.path.join(os.path.abspath(args.repo), args.node_id_file)
    node_id = read_or_create_node_id(node_id_path, args.node_id)
    payload = build_payload(args, node_id)

    print(f"[local_node] merkezi sunucu: {args.server}")
    print(f"[local_node] node_id: {node_id}")
    print(f"[local_node] cihaz: {payload['name']} ({payload['platform']}, {payload['capabilities']['preferred_device']})")
    if args.allow_remote_commands:
        print("[local_node] UYARI: uzaktan komut calistirma ACIK.")
    else:
        print("[local_node] uzaktan komut calistirma kapali. Acmak icin --allow-remote-commands kullan.")
    if args.allow_training_jobs:
        print("[local_node] sohbet egitimi isleri ACIK.")
    else:
        print("[local_node] sohbet egitimi isleri kapali. Acmak icin --allow-training-jobs kullan.")

    while True:
        try:
            path = "/api/nodes/register" if payload else f"/api/nodes/{node_id}/heartbeat"
            body = payload or build_payload(args, node_id)
            result = post_json(args.server, path, body, args.api_token)
            if payload:
                print(f"[local_node] kayit tamam: {result.get('node_id', node_id)}")
                payload = {}
            else:
                node = result.get("node") or {}
                print(f"[local_node] heartbeat: {node.get('status', 'online')} age={node.get('last_seen_age_sec', 0)}s")
                poll_and_run_commands(args, node_id)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"[local_node] sunucuya ulasilamadi: {exc}")
        time.sleep(max(3, args.heartbeat_sec))


if __name__ == "__main__":
    main()
