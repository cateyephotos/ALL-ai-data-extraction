#!/usr/bin/env python3
"""
extract_opencode.py -- export OpenCode session history to JSONL

Reads OpenCode's on-disk session store directly (pure file I/O) and flattens
each session to one JSON line. This was chosen over shelling out to
`opencode export <id>` because, although the CLI export is the "stable
contract", in practice each export call boots a heavy opencode runtime
(~10-15s) that does not reliably tear down -- batching it across dozens of
sessions hangs and leaves orphaned `opencode-cli` server processes. Direct
parsing is fast, deterministic, spawns nothing, and the storage schema below
was reverse-engineered and verified against opencode v1.1.48 storage on
2026-05.

Pipeline position:
  EXTRACTOR. Writes `extracted_data/opencode_conversations_<ts>.jsonl` --
  same convention as extract_codex.py / extract_augment.py. Downstream
  wiki/scripts/ingest/ingest_opencode.py reads the latest JSONL.

Storage schema (verified v1.1.48):
  <storage>/session/<projectID>/ses_*.json
      {id, slug, version, projectID, directory, parentID?, title,
       time:{created,updated}, summary:{additions,deletions,files}}
  <storage>/message/<sessionID>/msg_*.json
      {id, sessionID, role, time:{created}, summary:{title,diffs},
       agent?, model:{providerID,modelID}?, tokens?, cost?}
  <storage>/part/<messageID>/prt_*.json
      text:      {type:"text", text}
      reasoning: {type:"reasoning", text}
      tool:      {type:"tool"|"tool-call", tool|name, callID, state:{input,status,output}}
      code:      {type:"code", text, language}

Storage discovery (cross-platform):
  OpenCode (Bun/Node runtime) uses XDG-style data dirs even on Windows, so
  the canonical store is `~/.local/share/opencode/storage` on all three
  platforms, with `$XDG_DATA_HOME` and `%APPDATA%/opencode` as fallbacks.
  The desktop app (ai.opencode.desktop / ai.opencode.app) is a WebView2/Tauri
  shell over the same opencode server and shares this store -- there is no
  separate desktop transcript store to scrape.

No-op behavior:
  If no storage/sessions exist, prints a clear message and exits 0 (clean
  no-op) so the nightly orchestrator on machines without OpenCode records no
  false failure.

Output conversation shape (one JSON object per line), mirroring the
codex/augment convention so the ingester stays consistent:
  {
    "messages": [
      {"role","content","timestamp","model"?,"provider"?,"agent"?,
       "reasoning"?,"tool_calls"?,"tool_results"?,"msg_title"?,"tokens"?,"cost"?}
    ],
    "session_id","title","cwd","directory","project_id","parent_session_id"?,
    "slug"?,"version","created_at","updated_at","timestamp","source":"opencode",
    "session_file","installation","coding_platform":"opencode",
    "platform_variant","opencode_storage_kind","retrace_surface","summary"?
  }
"""

from __future__ import annotations

import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


# --- UTF-8 stdio guard (Windows scheduled-task cp1252 crash prevention) ----
def _ensure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):  # pragma: no cover
            pass


_ensure_utf8_stdio()


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def find_storage_dirs() -> list[Path]:
    """Return existing OpenCode storage directories. XDG-style on all OSes."""
    home = Path.home()
    candidate_roots = [
        Path(os.environ["XDG_DATA_HOME"]) / "opencode"
        if os.environ.get("XDG_DATA_HOME")
        else home / ".local/share/opencode",
        home / ".local/share/opencode",
    ]
    if platform.system() == "Windows":
        candidate_roots.append(
            Path(os.environ.get("APPDATA", home / "AppData/Roaming")) / "opencode"
        )
    if platform.system() == "Darwin":
        candidate_roots.append(home / "Library/Application Support/opencode")

    seen: set[Path] = set()
    out: list[Path] = []
    for root in candidate_roots:
        storage = root / "storage"
        if storage.exists() and storage not in seen:
            seen.add(storage)
            out.append(storage)
    return out


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return None


def index_sessions(storage: Path) -> dict[str, tuple[dict, Path]]:
    """Map session_id -> (session_metadata, session_file_path) from
    storage/session/<projectID>/ses_*.json. Filenames carry the id."""
    out: dict[str, tuple[dict, Path]] = {}
    session_root = storage / "session"
    if not session_root.exists():
        return out
    for path in session_root.rglob("ses_*.json"):
        sid = path.stem
        meta = _load_json(path)
        if isinstance(meta, dict):
            out.setdefault(sid, (meta, path))
    return out


# ---------------------------------------------------------------------------
# Message + part assembly
# ---------------------------------------------------------------------------


def _dget(obj: object, key: str, default=None):
    """Safe nested .get -- returns default if obj isn't a dict. OpenCode
    occasionally stores a bare bool where a {} is expected (e.g. message
    `summary`), so guard every nested access."""
    return obj.get(key, default) if isinstance(obj, dict) else default


def _ms_to_iso(ms: object) -> str | None:
    if not isinstance(ms, (int, float)):
        return None
    try:
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()
    except (ValueError, OverflowError, OSError):
        return None


def _flatten_parts(part_dir: Path) -> dict:
    """Collapse a message's parts into content/reasoning/tool fields."""
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict] = []
    tool_results: list[dict] = []

    if not part_dir.exists():
        return {"content": ""}

    for part_file in sorted(part_dir.glob("prt_*.json")):
        part = _load_json(part_file)
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        text = part.get("text", "") or ""

        if ptype == "text":
            if text:
                content_parts.append(text)
        elif ptype == "reasoning":
            if text:
                reasoning_parts.append(text)
        elif ptype in ("tool", "tool-call"):
            state = part.get("state", {}) if isinstance(part.get("state"), dict) else {}
            tool_name = part.get("tool") or part.get("name")
            tool_calls.append({
                "id": part.get("callID") or part.get("id"),
                "name": tool_name,
                "input": state.get("input", part.get("input")),
            })
            if state.get("status") == "completed" and "output" in state:
                tool_results.append({
                    "tool_call_id": part.get("callID"),
                    "tool": tool_name,
                    "output": state.get("output"),
                })
        elif ptype == "tool-result":
            tool_results.append({
                "tool_call_id": part.get("toolCallID"),
                "output": part.get("output"),
            })
        elif ptype == "code":
            lang = part.get("language", "")
            if text:
                content_parts.append(f"```{lang}\n{text}\n```")

    out: dict = {"content": "\n".join(content_parts)}
    if reasoning_parts:
        out["reasoning"] = "\n".join(reasoning_parts)
    if tool_calls:
        out["tool_calls"] = tool_calls
    if tool_results:
        out["tool_results"] = tool_results
    return out


def build_conversation(
    session_id: str,
    storage: Path,
    session_index: dict[str, tuple[dict, Path]],
) -> dict | None:
    """Assemble one conversation from message + part files for a session."""
    msg_dir = storage / "message" / session_id
    if not msg_dir.exists():
        return None

    part_root = storage / "part"
    messages: list[dict] = []
    first_ts = None
    last_ts = None

    for msg_file in sorted(msg_dir.glob("msg_*.json")):
        minfo = _load_json(msg_file)
        if not isinstance(minfo, dict):
            continue
        msg_id = minfo.get("id") or msg_file.stem
        created = _dget(minfo.get("time"), "created")
        if isinstance(created, (int, float)):
            first_ts = created if first_ts is None else min(first_ts, created)
            last_ts = created if last_ts is None else max(last_ts, created)

        flat = _flatten_parts(part_root / msg_id)
        msg: dict = {
            "role": minfo.get("role", "assistant"),
            "content": flat.get("content", ""),
            "timestamp": created,
        }
        model = minfo.get("model")
        if isinstance(model, dict):
            if model.get("modelID"):
                msg["model"] = model["modelID"]
            if model.get("providerID"):
                msg["provider"] = model["providerID"]
        if minfo.get("agent"):
            msg["agent"] = minfo["agent"]
        msg_title = _dget(minfo.get("summary"), "title")
        if msg_title:
            msg["msg_title"] = msg_title
        if minfo.get("tokens"):
            msg["tokens"] = minfo["tokens"]
        if minfo.get("cost") is not None:
            msg["cost"] = minfo["cost"]
        for key in ("reasoning", "tool_calls", "tool_results"):
            if key in flat:
                msg[key] = flat[key]

        if msg["content"] or "tool_calls" in msg or "reasoning" in msg:
            messages.append(msg)

    if not messages:
        return None

    meta, session_file = session_index.get(session_id, ({}, msg_dir))
    time_obj = meta.get("time", {}) if isinstance(meta.get("time"), dict) else {}
    created = time_obj.get("created", first_ts)
    updated = time_obj.get("updated", last_ts)
    directory = meta.get("directory")

    conv: dict = {
        "messages": messages,
        "session_id": session_id,
        "title": meta.get("title"),
        "cwd": directory,
        "directory": directory,
        "project_id": meta.get("projectID"),
        "slug": meta.get("slug"),
        "version": meta.get("version") or "unknown",
        "created_at": created,
        "updated_at": updated,
        "timestamp": _ms_to_iso(created) or "unknown",
        "source": "opencode",
        "session_file": str(session_file),
        "installation": f"opencode {meta.get('version') or 'unknown'}".strip(),
        "coding_platform": "opencode",
        "platform_variant": "opencode_cli",
        "opencode_storage_kind": "storage_files",
        "retrace_surface": "opencode_export",
    }
    if meta.get("parentID"):
        conv["parent_session_id"] = meta["parentID"]
    if isinstance(meta.get("summary"), dict):
        conv["summary"] = meta["summary"]
    return conv


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    print("=" * 80)
    print("OPENCODE EXTRACTION (direct storage parse)")
    print("=" * 80)

    storage_dirs = find_storage_dirs()
    if not storage_dirs:
        print("No opencode storage directory found -- nothing to extract (clean no-op).")
        return 0

    conversations: list[dict] = []
    seen_ids: set[str] = set()

    for storage in storage_dirs:
        print(f"storage: {storage}")
        session_index = index_sessions(storage)
        # Enumerate sessions by message dir so we capture sessions even if the
        # session metadata file is missing (reconstruct from messages).
        msg_root = storage / "message"
        session_ids = (
            sorted(d.name for d in msg_root.iterdir() if d.is_dir() and d.name.startswith("ses_"))
            if msg_root.exists() else []
        )
        print(f"  sessions with messages: {len(session_ids)} "
              f"(metadata files: {len(session_index)})")

        for i, sid in enumerate(session_ids, 1):
            if sid in seen_ids:
                continue
            seen_ids.add(sid)
            conv = build_conversation(sid, storage, session_index)
            if conv:
                conversations.append(conv)
            if i <= 3 or i % 10 == 0 or i == len(session_ids):
                n = len(conv["messages"]) if conv else 0
                title = (conv.get("title") if conv else None) or "(no metadata)"
                print(f"  [{i}/{len(session_ids)}] {sid} -> {n} msg(s)  {str(title)[:60]}")

    if not conversations:
        print("Exported 0 conversations (sessions empty).")
        return 0

    total_messages = sum(len(c["messages"]) for c in conversations)
    with_tools = sum(1 for c in conversations
                     if any("tool_calls" in m for m in c["messages"]))
    with_parent = sum(1 for c in conversations if c.get("parent_session_id"))
    with_meta = sum(1 for c in conversations if c.get("title"))

    print()
    print(f"Total conversations: {len(conversations)}")
    print(f"Total messages:      {total_messages}")
    print(f"With tool use:       {with_tools}")
    print(f"Subagent/child:      {with_parent}")
    print(f"With session title:  {with_meta}")

    output_dir = Path("extracted_data")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"opencode_conversations_{timestamp}.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")

    size_kb = output_file.stat().st_size / 1024
    print(f"Saved: {output_file} ({size_kb:.1f} KB, JSONL one-conversation-per-line)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
