from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from collections.abc import Iterator
from typing import Any


def _now() -> float:
    return time.time()


def _json_dump(value: Any) -> str:
    return json.dumps(value or {}, ensure_ascii=False, sort_keys=True)


def _json_load(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


class NodeRegistry:
    """SQLite tabanli merkezi kayit defteri.

    Bu katman is calistirmaz. Sadece yerel makinelerin durumunu, kaynaklarini
    ve LAN egitim oturumlari icin uretilen komutlari kaydeder.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_db()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        try:
            yield conn
        except Exception:
            conn.rollback()
            raise
        else:
            conn.commit()
        finally:
            conn.close()

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS nodes (
                    node_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    hostname TEXT,
                    platform TEXT,
                    os TEXT,
                    arch TEXT,
                    python TEXT,
                    repo_path TEXT,
                    base_url TEXT,
                    worker_version TEXT,
                    resources_json TEXT NOT NULL DEFAULT '{}',
                    capabilities_json TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL DEFAULT 'online',
                    last_seen REAL NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS node_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    message TEXT NOT NULL,
                    data_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS train_sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    preset TEXT NOT NULL,
                    data_path TEXT NOT NULL,
                    master_addr TEXT NOT NULL,
                    master_port INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    config_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS train_session_nodes (
                    session_id TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    node_rank INTEGER NOT NULL,
                    command TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'planned',
                    created_at REAL NOT NULL,
                    PRIMARY KEY (session_id, node_id)
                );

                CREATE TABLE IF NOT EXISTS node_commands (
                    command_id TEXT PRIMARY KEY,
                    node_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    type TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL DEFAULT 'queued',
                    exit_code INTEGER,
                    output TEXT NOT NULL DEFAULT '',
                    error TEXT NOT NULL DEFAULT '',
                    result_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    started_at REAL,
                    finished_at REAL
                );

                CREATE INDEX IF NOT EXISTS idx_node_commands_node_status
                ON node_commands (node_id, status, created_at);
                """
            )

    def upsert_node(self, payload: dict[str, Any]) -> dict[str, Any]:
        ts = _now()
        node_id = str(payload.get("node_id") or uuid.uuid4())
        name = str(payload.get("name") or payload.get("hostname") or node_id[:8])
        resources = payload.get("resources") or {}
        capabilities = payload.get("capabilities") or {}

        with self._connect() as conn:
            old = conn.execute(
                "SELECT created_at FROM nodes WHERE node_id = ?", (node_id,)
            ).fetchone()
            created_at = float(old["created_at"]) if old else ts
            conn.execute(
                """
                INSERT INTO nodes (
                    node_id, name, hostname, platform, os, arch, python,
                    repo_path, base_url, worker_version, resources_json,
                    capabilities_json, status, last_seen, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    name = excluded.name,
                    hostname = excluded.hostname,
                    platform = excluded.platform,
                    os = excluded.os,
                    arch = excluded.arch,
                    python = excluded.python,
                    repo_path = excluded.repo_path,
                    base_url = excluded.base_url,
                    worker_version = excluded.worker_version,
                    resources_json = excluded.resources_json,
                    capabilities_json = excluded.capabilities_json,
                    status = excluded.status,
                    last_seen = excluded.last_seen,
                    updated_at = excluded.updated_at
                """,
                (
                    node_id,
                    name,
                    payload.get("hostname"),
                    payload.get("platform"),
                    payload.get("os"),
                    payload.get("arch"),
                    payload.get("python"),
                    payload.get("repo_path"),
                    payload.get("base_url"),
                    payload.get("worker_version"),
                    _json_dump(resources),
                    _json_dump(capabilities),
                    payload.get("status") or "online",
                    ts,
                    created_at,
                    ts,
                ),
            )
        self.add_event(node_id, "register", f"{name} kaydoldu", payload)
        return self.get_node(node_id) or {"node_id": node_id}

    def heartbeat(self, node_id: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        ts = _now()
        status = str(payload.get("status") or "online")
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM nodes WHERE node_id = ?", (node_id,)).fetchone()
            if not row:
                return None
            resources = _json_load(row["resources_json"], {})
            capabilities = _json_load(row["capabilities_json"], {})
            resources.update(payload.get("resources") or {})
            capabilities.update(payload.get("capabilities") or {})
            conn.execute(
                """
                UPDATE nodes
                SET status = ?, last_seen = ?, updated_at = ?,
                    resources_json = ?, capabilities_json = ?
                WHERE node_id = ?
                """,
                (
                    status,
                    ts,
                    ts,
                    _json_dump(resources),
                    _json_dump(capabilities),
                    node_id,
                ),
            )
        return self.get_node(node_id)

    def add_event(self, node_id: str, kind: str, message: str, data: dict[str, Any] | None = None):
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO node_events (node_id, kind, message, data_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (node_id, kind, message, _json_dump(data), _now()),
            )

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM nodes WHERE node_id = ?", (node_id,)).fetchone()
        return self._row_to_node(row) if row else None

    def list_nodes(self, stale_after_sec: int = 45) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM nodes ORDER BY last_seen DESC").fetchall()
        return [self._row_to_node(row, stale_after_sec=stale_after_sec) for row in rows]

    def recent_events(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM node_events ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [
            {
                "id": row["id"],
                "node_id": row["node_id"],
                "kind": row["kind"],
                "message": row["message"],
                "data": _json_load(row["data_json"], {}),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def enqueue_command(self, node_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.get_node(node_id):
            raise ValueError("node kayitli degil")

        ts = _now()
        command_id = str(payload.get("id") or payload.get("command_id") or uuid.uuid4())
        command_type = str(payload.get("type") or "shell")
        title = str(payload.get("title") or command_type)
        clean_payload = dict(payload or {})
        clean_payload.update({"id": command_id, "type": command_type, "title": title})

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO node_commands (
                    command_id, node_id, title, type, payload_json, status,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    command_id,
                    node_id,
                    title,
                    command_type,
                    _json_dump(clean_payload),
                    "queued",
                    ts,
                    ts,
                ),
            )
        self.add_event(node_id, "command_queued", f"{title} kuyruğa alındı", {"command_id": command_id})
        return self.get_command(command_id) or {"id": command_id}

    def poll_commands(self, node_id: str, limit: int = 1) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit or 1), 10))
        ts = _now()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM node_commands
                WHERE node_id = ? AND status = 'queued'
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (node_id, limit),
            ).fetchall()
            command_ids = [row["command_id"] for row in rows]
            for command_id in command_ids:
                conn.execute(
                    """
                    UPDATE node_commands
                    SET status = 'running', started_at = COALESCE(started_at, ?), updated_at = ?
                    WHERE command_id = ? AND status = 'queued'
                    """,
                    (ts, ts, command_id),
                )
        return [command for command_id in command_ids if (command := self.get_command(command_id))]

    def update_command_status(self, command_id: str, result: dict[str, Any]) -> dict[str, Any] | None:
        existing = self.get_command(command_id)
        if not existing:
            return None

        ts = _now()
        status = str(result.get("status") or existing.get("status") or "done")
        exit_code = result.get("exit_code")
        if exit_code is not None:
            try:
                exit_code = int(exit_code)
            except (TypeError, ValueError):
                exit_code = None
        output = str(result.get("output") or "")
        error = str(result.get("error") or "")
        finished_at = ts if status in {"done", "failed", "refused", "cancelled"} else None

        with self._connect() as conn:
            conn.execute(
                """
                UPDATE node_commands
                SET status = ?, exit_code = ?, output = ?, error = ?,
                    result_json = ?, updated_at = ?,
                    finished_at = COALESCE(?, finished_at)
                WHERE command_id = ?
                """,
                (
                    status,
                    exit_code,
                    output[-12000:],
                    error[-4000:],
                    _json_dump(result),
                    ts,
                    finished_at,
                    command_id,
                ),
            )
        node_id = str(existing.get("node_id") or "")
        if node_id:
            self.add_event(node_id, "command_status", f"{existing.get('title')} -> {status}", {
                "command_id": command_id,
                "status": status,
                "error": error,
            })
        return self.get_command(command_id)

    def get_command(self, command_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM node_commands WHERE command_id = ?", (command_id,)
            ).fetchone()
        return self._row_to_command(row) if row else None

    def list_commands(self, node_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit or 50), 200))
        with self._connect() as conn:
            if node_id:
                rows = conn.execute(
                    """
                    SELECT * FROM node_commands
                    WHERE node_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (node_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM node_commands ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [self._row_to_command(row) for row in rows]

    def create_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        nodes = self.list_nodes(stale_after_sec=10**9)
        selected_ids = payload.get("node_ids") or [n["node_id"] for n in nodes if n["status"] == "online"]
        node_by_id = {n["node_id"]: n for n in nodes}
        selected = [node_by_id[node_id] for node_id in selected_ids if node_id in node_by_id]
        if not selected:
            raise ValueError("Oturum icin en az bir kayitli node gerekli.")

        session_id = str(uuid.uuid4())
        title = str(payload.get("title") or "LAN egitim oturumu")
        preset = str(payload.get("preset") or "small-100m")
        data_path = str(payload.get("data_path") or "data/bin")
        master_addr = str(payload.get("master_addr") or "127.0.0.1")
        master_port = int(payload.get("master_port") or 29500)
        ts = _now()

        commands = []
        for rank, node in enumerate(selected):
            command = self._build_train_command(
                node=node,
                node_rank=rank,
                nnodes=len(selected),
                master_addr=master_addr,
                master_port=master_port,
                preset=preset,
                data_path=data_path,
            )
            commands.append({"node_id": node["node_id"], "node_rank": rank, "command": command})

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO train_sessions (
                    session_id, title, preset, data_path, master_addr, master_port,
                    status, config_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    title,
                    preset,
                    data_path,
                    master_addr,
                    master_port,
                    "planned",
                    _json_dump({"node_ids": selected_ids}),
                    ts,
                    ts,
                ),
            )
            for item in commands:
                conn.execute(
                    """
                    INSERT INTO train_session_nodes (
                        session_id, node_id, node_rank, command, status, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        item["node_id"],
                        item["node_rank"],
                        item["command"],
                        "planned",
                        ts,
                    ),
                )
        return self.get_session(session_id) or {"session_id": session_id}

    def list_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT session_id FROM train_sessions ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [session for row in rows if (session := self.get_session(row["session_id"]))]

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            session = conn.execute(
                "SELECT * FROM train_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if not session:
                return None
            nodes = conn.execute(
                """
                SELECT tsn.*, n.name, n.platform, n.hostname
                FROM train_session_nodes tsn
                LEFT JOIN nodes n ON n.node_id = tsn.node_id
                WHERE tsn.session_id = ?
                ORDER BY tsn.node_rank
                """,
                (session_id,),
            ).fetchall()
        return {
            "session_id": session["session_id"],
            "title": session["title"],
            "preset": session["preset"],
            "data_path": session["data_path"],
            "master_addr": session["master_addr"],
            "master_port": session["master_port"],
            "status": session["status"],
            "config": _json_load(session["config_json"], {}),
            "created_at": session["created_at"],
            "updated_at": session["updated_at"],
            "nodes": [
                {
                    "node_id": row["node_id"],
                    "node_rank": row["node_rank"],
                    "name": row["name"],
                    "platform": row["platform"],
                    "hostname": row["hostname"],
                    "command": row["command"],
                    "status": row["status"],
                }
                for row in nodes
            ],
        }

    def _row_to_node(self, row: sqlite3.Row, stale_after_sec: int = 45) -> dict[str, Any]:
        age = max(0.0, _now() - float(row["last_seen"]))
        status = row["status"]
        if age > stale_after_sec:
            status = "offline"
        return {
            "node_id": row["node_id"],
            "name": row["name"],
            "hostname": row["hostname"],
            "platform": row["platform"],
            "os": row["os"],
            "arch": row["arch"],
            "python": row["python"],
            "repo_path": row["repo_path"],
            "base_url": row["base_url"],
            "worker_version": row["worker_version"],
            "resources": _json_load(row["resources_json"], {}),
            "capabilities": _json_load(row["capabilities_json"], {}),
            "status": status,
            "last_seen": row["last_seen"],
            "last_seen_age_sec": round(age, 1),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _row_to_command(self, row: sqlite3.Row) -> dict[str, Any]:
        payload = _json_load(row["payload_json"], {})
        command = dict(payload)
        command.update({
            "id": row["command_id"],
            "command_id": row["command_id"],
            "node_id": row["node_id"],
            "title": row["title"],
            "type": row["type"],
            "status": row["status"],
            "exit_code": row["exit_code"],
            "output": row["output"],
            "error": row["error"],
            "result": _json_load(row["result_json"], {}),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
        })
        return command

    def _build_train_command(
        self,
        node: dict[str, Any],
        node_rank: int,
        nnodes: int,
        master_addr: str,
        master_port: int,
        preset: str,
        data_path: str,
    ) -> str:
        device = node.get("capabilities", {}).get("preferred_device") or "auto"
        platform = (node.get("platform") or "").lower()
        if "windows" in platform:
            return (
                "powershell -ExecutionPolicy Bypass -File .\\scripts\\lan_train.ps1 "
                f"-NodeRank {node_rank} -MasterAddr {master_addr} -Nodes {nnodes} "
                f"-MasterPort {master_port} -Device {device} -Preset {preset} -Data {data_path}"
            )
        return (
            f"NODE_RANK={node_rank} MASTER_ADDR={master_addr} NNODES={nnodes} "
            f"MASTER_PORT={master_port} DEVICE={device} PRESET={preset} DATA={data_path} "
            "bash scripts/lan_train.sh"
        )
