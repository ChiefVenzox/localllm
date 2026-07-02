"""
scripts/system_check.py
=======================
localllm icin sistem gereksinimi ve paketleme kontrolu.

Kullanim:
    python scripts/system_check.py
    python scripts/system_check.py --mode docker
    python scripts/system_check.py --mode training --require-gpu
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Result:
    level: str
    name: str
    detail: str


class Reporter:
    def __init__(self):
        self.results: list[Result] = []

    def add(self, level: str, name: str, detail: str):
        self.results.append(Result(level, name, detail))

    def ok(self, name: str, detail: str):
        self.add("OK", name, detail)

    def warn(self, name: str, detail: str):
        self.add("WARN", name, detail)

    def fail(self, name: str, detail: str):
        self.add("FAIL", name, detail)

    def print(self, as_json: bool = False):
        if as_json:
            print(json.dumps([r.__dict__ for r in self.results], indent=2, ensure_ascii=False))
            return
        width = max([len(r.name) for r in self.results] + [12])
        for r in self.results:
            print(f"[{r.level:<4}] {r.name:<{width}} {r.detail}")

    def exit_code(self, strict: bool = False) -> int:
        bad = {"FAIL"} | ({"WARN"} if strict else set())
        return 1 if any(r.level in bad for r in self.results) else 0


def run(cmd: list[str], timeout: int = 8) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        return proc.returncode, proc.stdout.strip()
    except FileNotFoundError:
        return 127, f"komut bulunamadi: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, "timeout"
    except Exception as exc:
        return 1, str(exc)


def fmt_bytes(value: int | float | None) -> str:
    if value is None:
        return "?"
    value = float(value)
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    while value >= 1024 and idx < len(units) - 1:
        value /= 1024
        idx += 1
    return f"{value:.1f} {units[idx]}"


def total_ram() -> int | None:
    try:
        if sys.platform == "darwin":
            code, out = run(["sysctl", "-n", "hw.memsize"])
            return int(out) if code == 0 else None
        if hasattr(os, "sysconf"):
            return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except Exception:
        return None
    return None


def port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def check_python(rep: Reporter):
    version = sys.version_info
    detail = f"{platform.python_version()} ({sys.executable})"
    if version >= (3, 10):
        rep.ok("python", detail)
    else:
        rep.fail("python", detail + " | Python 3.10+ gerekli")


def check_modules(rep: Reporter, modules: Iterable[str]):
    missing = []
    for name in modules:
        if importlib.util.find_spec(name) is None:
            missing.append(name)
    if missing:
        rep.fail("python-modules", "eksik: " + ", ".join(missing))
    else:
        rep.ok("python-modules", "temel moduller yuklu")


def check_torch(rep: Reporter, require_gpu: bool):
    if importlib.util.find_spec("torch") is None:
        rep.fail("torch", "kurulu degil")
        return
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
        if cuda:
            devices = []
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                devices.append(f"{props.name} ({fmt_bytes(props.total_memory)})")
            rep.ok("torch", f"{torch.__version__}, CUDA var: " + "; ".join(devices))
        elif require_gpu:
            rep.fail("torch", f"{torch.__version__}, CUDA gorunmuyor")
        else:
            rep.warn("torch", f"{torch.__version__}, CUDA yok; CPU calisir ama egitim yavas")
    except Exception as exc:
        rep.fail("torch", str(exc))


def check_nvidia(rep: Reporter, require_gpu: bool):
    code, out = run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        timeout=5,
    )
    if code != 0:
        if require_gpu:
            rep.fail("nvidia-smi", out or "NVIDIA driver/gpu gorunmuyor")
        else:
            rep.warn("nvidia-smi", out or "NVIDIA driver/gpu gorunmuyor")
        return
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    rep.ok("nvidia-smi", "; ".join(lines))


def check_docker(rep: Reporter, require_gpu: bool):
    in_container = Path("/.dockerenv").exists() or os.environ.get("container")
    if not shutil.which("docker"):
        if in_container:
            rep.ok("docker-runtime", "container icinde calisiyor; host docker CLI gerekli degil")
            if require_gpu:
                rep.warn("docker-gpu-runtime", "GPU icin hostta NVIDIA Container Toolkit gerekir")
            return
        rep.fail("docker", "docker komutu yok")
        return
    code, out = run(["docker", "--version"])
    rep.ok("docker", out if code == 0 else "surum okunamadi")
    code, out = run(["docker", "info"], timeout=10)
    if code == 0:
        rep.ok("docker-daemon", "erisilebilir")
    else:
        rep.fail("docker-daemon", out or "docker daemon erisilemiyor")

    code, out = run(["docker", "compose", "version"])
    if code == 0:
        rep.ok("docker-compose", out)
    else:
        rep.fail("docker-compose", out or "docker compose yok")

    if require_gpu:
        code, out = run(["docker", "info", "--format", "{{json .Runtimes}}"], timeout=10)
        if code == 0 and "nvidia" in out.lower():
            rep.ok("docker-gpu-runtime", "nvidia runtime gorunuyor")
        else:
            rep.warn("docker-gpu-runtime", "nvidia runtime gorunmedi; NVIDIA Container Toolkit gerekebilir")


def check_disk_and_ram(rep: Reporter, mode: str):
    usage = shutil.disk_usage(ROOT)
    rep.ok("disk-free", f"{fmt_bytes(usage.free)} bos ({ROOT})")
    if usage.free < 20 * 1024**3:
        rep.fail("disk-free", f"{fmt_bytes(usage.free)} bos; Docker image + veri icin en az 20GB onerilir")
    elif mode in {"docker", "training", "all"} and usage.free < 60 * 1024**3:
        rep.warn("disk-free", f"{fmt_bytes(usage.free)} bos; GPU image + 200M veri icin 60GB+ rahat olur")

    ram = total_ram()
    if ram is None:
        rep.warn("ram", "toplam RAM okunamadi")
    elif ram < 8 * 1024**3:
        rep.fail("ram", f"{fmt_bytes(ram)}; API icin 8GB+ onerilir")
    elif mode in {"training", "all"} and ram < 16 * 1024**3:
        rep.warn("ram", f"{fmt_bytes(ram)}; 200M veri hazirlama/egitim icin 16GB+, tercihen 32GB")
    else:
        rep.ok("ram", fmt_bytes(ram))


def check_files(rep: Reporter):
    required = [
        ("tokenizer", ROOT / "tokenizer" / "tokenizer.json"),
        ("checkpoint", ROOT / "checkpoints" / "ckpt.pt"),
    ]
    for name, path in required:
        if path.exists():
            rep.ok(name, f"{path.relative_to(ROOT)} ({fmt_bytes(path.stat().st_size)})")
        else:
            rep.fail(name, f"eksik: {path.relative_to(ROOT)}")

    adapter = ROOT / "checkpoints" / "adapter.pt"
    if adapter.exists():
        rep.ok("adapter", f"{adapter.relative_to(ROOT)} ({fmt_bytes(adapter.stat().st_size)})")
    else:
        rep.warn("adapter", "yok; model adaptersiz calisir")

    data = ROOT / "data" / "chat_200m_plus_bin" / "meta.json"
    if data.exists():
        try:
            meta = json.loads(data.read_text(encoding="utf-8"))
            rep.ok("chat-200m-bin", f"{meta.get('n_tokens_total', '?'):,} token")
        except Exception as exc:
            rep.warn("chat-200m-bin", f"meta okunamadi: {exc}")
    else:
        rep.warn("chat-200m-bin", "yok; 200M egitim secenegi hazir degil")


def check_ports(rep: Reporter, port: int):
    if port_in_use("127.0.0.1", port):
        rep.warn("port", f"{port} kullanimda; Docker icin YERELLM_PORT degistirilebilir")
    else:
        rep.ok("port", f"{port} bos")


def check_compose_config(rep: Reporter):
    if Path("/.dockerenv").exists() or os.environ.get("container"):
        rep.ok("compose-config", "container icinde atlandi")
        return
    if not (ROOT / "compose.yaml").exists():
        rep.warn("compose", "compose.yaml yok")
        return
    code, out = run(["docker", "compose", "config"], timeout=15)
    if code == 0:
        rep.ok("compose-config", "gecerli")
    else:
        rep.fail("compose-config", out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("local", "docker", "training", "all"), default="all")
    parser.add_argument("--port", type=int, default=int(os.environ.get("YERELLM_PORT", "8000")))
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--strict", action="store_true", help="WARN sonucunu da hata say")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    rep = Reporter()
    rep.ok("platform", f"{platform.system()} {platform.release()} {platform.machine()}")
    check_python(rep)
    check_disk_and_ram(rep, args.mode)
    check_files(rep)
    check_ports(rep, args.port)
    check_modules(rep, ["fastapi", "uvicorn", "numpy", "tokenizers", "tqdm"])
    check_torch(rep, args.require_gpu or args.mode in {"training"})
    if args.mode in {"docker", "training", "all"}:
        check_docker(rep, args.require_gpu)
        check_compose_config(rep)
    if args.mode in {"training", "all"} or args.require_gpu:
        check_nvidia(rep, args.require_gpu)

    rep.print(as_json=args.json)
    return rep.exit_code(strict=args.strict)


if __name__ == "__main__":
    raise SystemExit(main())
