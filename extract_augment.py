#!/usr/bin/env python3
"""
Extract Augment (vscode-augment) chat/agent exchanges from LevelDB stores.

Augment stores its chat history in Chromium LevelDB files under each VS Code
workspace's extension storage:

    %APPDATA%/Code/User/workspaceStorage/<ws-hash>/
        Augment.vscode-augment/
            augment-kv-store/
                *.ldb       SSTable data (one or more)
                *.log       Pending writes (current mem-table)
            Augment-Memories            Per-workspace agent memories (may be empty)

Records are keyed by the Augment-internal pattern ``exchange:<conv-uuid>:<exchange-uuid>``
with JSON values containing ``request_message``, ``response_text``,
``conversation_id``, ``exchange_id``, timestamps, and tool traces.

``plyvel`` (the canonical LevelDB Python binding) does not build on Windows
without significant toolchain setup, so this script implements a minimal
LevelDB SSTable reader directly:

  1. Walk each ``.ldb`` file, read the 48-byte footer, decode the
     ``metaindex`` / ``index`` block handles, validate the LevelDB magic
     (``0xdb4775248b80fb57``).
  2. Read the index block, iterate its entries to recover every data-block
     handle (offset + length).
  3. Decompress each data block (Snappy raw — not Snappy-framed — via
     ``cramjam.snappy.decompress_raw``) and iterate its restart-point-
     delimited entries.
  4. Keep only keys starting with ``exchange:`` and parse the JSON value
     for each.
  5. Deduplicate by ``(conversation_id, exchange_id)`` across all ``.ldb``
     files, grouped per workspace.
  6. Group surviving exchanges by ``conversation_id``, sort by request
     timestamp, and emit one conversation object per group in the toolkit's
     standard JSONL schema (``messages``, ``source``, ``name``,
     ``conversation_id``, ``workspace``, ``created_at``).

Output:
    extracted_data/augment_conversations_<ts>.jsonl
    extracted_data/augment_conversations_<ts>.inventory.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

try:
    import cramjam  # noqa: F401
except ImportError:  # pragma: no cover
    print(
        "ERROR: cramjam is required for Snappy decompression. "
        "Install with: pip install cramjam",
        file=sys.stderr,
    )
    raise


_LEVELDB_MAGIC = 0xDB4775248B80FB57
_ROCKSDB_MAGIC_V1 = 0xEB3481F02DE9F1B4

_EXCHANGE_KEY_PREFIX = b"exchange:"
# LevelDB "internal keys" append an 8-byte (seq<<8 | type) suffix to every
# user key inside data blocks, so we match on prefix only.
_EXCHANGE_KEY_RE = re.compile(
    rb"^exchange:"
    rb"(?P<conv>[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"
    rb":"
    rb"(?P<ex>[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"
)


# ---------------------------------------------------------------------------
# Minimal LevelDB SSTable reader
# ---------------------------------------------------------------------------


def _read_varint(data: bytes, pos: int) -> tuple[int, int]:
    """Decode a little-endian base128 varint starting at ``pos``."""
    result = 0
    shift = 0
    while pos < len(data):
        b = data[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if b < 0x80:
            return result, pos
        shift += 7
        if shift >= 64:
            raise ValueError("varint too long")
    raise ValueError("truncated varint")


def _read_block_handle(data: bytes, pos: int) -> tuple[tuple[int, int], int]:
    offset, pos = _read_varint(data, pos)
    length, pos = _read_varint(data, pos)
    return (offset, length), pos


def _parse_footer(data: bytes) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return (metaindex_handle, index_handle). Raises on bad magic."""
    if len(data) < 48:
        raise ValueError("file too short to contain a LevelDB footer")
    magic = struct.unpack("<Q", data[-8:])[0]
    if magic == _LEVELDB_MAGIC:
        footer_size = 48
    elif magic == _ROCKSDB_MAGIC_V1:
        footer_size = 53
    else:
        raise ValueError(f"unrecognized SSTable magic 0x{magic:016x}")
    footer = data[-footer_size:]
    metaindex, pos = _read_block_handle(footer, 0)
    index, _ = _read_block_handle(footer, pos)
    return metaindex, index


def _read_raw_block(data: bytes, handle: tuple[int, int]) -> bytes:
    """Read + decompress a block. Trailer = 1-byte compression + 4-byte CRC."""
    from cramjam import snappy

    offset, length = handle
    if offset < 0 or length < 0 or offset + length > len(data):
        raise ValueError(f"block handle out of range: {handle}")
    raw = data[offset : offset + length]
    trailer = data[offset + length : offset + length + 5]
    if not trailer:
        return raw
    compression = trailer[0]
    if compression == 0:
        return raw
    if compression == 1:
        return bytes(snappy.decompress_raw(raw))
    # zstd/zlib/lz4 not used by vscode-augment
    raise ValueError(f"unsupported compression type {compression}")


def _iter_block_entries(block: bytes) -> Iterable[tuple[bytes, bytes]]:
    """Yield (key, value) for every entry in a decompressed block."""
    n = len(block)
    if n < 4:
        return
    num_restarts = struct.unpack("<I", block[n - 4 : n])[0]
    entries_end = n - 4 - 4 * num_restarts
    if entries_end < 0:
        return
    pos = 0
    last_key = b""
    while pos < entries_end:
        try:
            shared, pos = _read_varint(block, pos)
            unshared, pos = _read_varint(block, pos)
            value_len, pos = _read_varint(block, pos)
        except ValueError:
            break
        if pos + unshared + value_len > entries_end:
            break
        key = last_key[:shared] + block[pos : pos + unshared]
        pos += unshared
        value = block[pos : pos + value_len]
        pos += value_len
        last_key = key
        yield key, value


def _default_code_roots() -> list[Path]:
    """Roots where VS Code (and forks) might keep Augment workspaceStorage."""
    appdata = os.environ.get("APPDATA")
    roots: list[Path] = []
    if appdata:
        for flavor in (
            "Code",
            "Code - Insiders",
            "Cursor",
            "Windsurf",
            "Trae",
            "VSCodium",
        ):
            root = Path(appdata) / flavor / "User" / "workspaceStorage"
            if root.exists():
                roots.append(root)
    home_roots = [
        Path.home() / "Library/Application Support/Code/User/workspaceStorage",
        Path.home() / ".config/Code/User/workspaceStorage",
    ]
    for r in home_roots:
        if r.exists():
            roots.append(r)
    return roots


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


@dataclass
class Exchange:
    """A single request/response pair inside an Augment conversation."""

    conversation_id: str
    exchange_id: str
    request_text: str = ""
    response_text: str = ""
    request_timestamp: str | None = None
    response_timestamp: str | None = None
    raw: dict = field(default_factory=dict)

    def sort_key(self) -> str:
        # Prefer request timestamp, fall back to exchange_id
        return self.request_timestamp or self.exchange_id


def _pull_request_text(obj: dict) -> str:
    """Augment stores the user request under various nested keys — try them."""
    for key in ("request_message", "user_message", "request_text"):
        v = obj.get(key)
        if isinstance(v, str) and v.strip():
            return v
    nodes = obj.get("request_nodes") or []
    for n in nodes:
        if isinstance(n, dict):
            txt = n.get("text") or n.get("content") or n.get("message")
            if isinstance(txt, str) and txt.strip():
                return txt
    return ""


def _pull_response_text(obj: dict) -> str:
    for key in ("response_text", "assistant_message", "response_message"):
        v = obj.get(key)
        if isinstance(v, str) and v.strip():
            return v
    nodes = obj.get("response_nodes") or []
    parts = []
    for n in nodes:
        if isinstance(n, dict):
            txt = n.get("text") or n.get("content")
            if isinstance(txt, str) and txt.strip():
                parts.append(txt)
    return "\n\n".join(parts)


def _pull_timestamp(obj: dict, kind: str) -> str | None:
    """kind in {'request','response'}. Timestamps live in *_nodes[0] typically."""
    nodes = obj.get(f"{kind}_nodes") or []
    for n in nodes:
        if isinstance(n, dict):
            for k in ("start_timestamp", "timestamp", "created_at"):
                v = n.get(k)
                if isinstance(v, str):
                    return v
                if isinstance(v, (int, float)):
                    return datetime.fromtimestamp(
                        v / (1000 if v > 1e12 else 1),
                        tz=timezone.utc,
                    ).isoformat()
    return None


def _scan_ldb_file(path: Path) -> Iterable[Exchange]:
    """Iterate exchanges in a single LevelDB SSTable (.ldb) file.

    Walks the index block, decompresses each data block, and yields one
    ``Exchange`` for every key matching ``exchange:<conv_uuid>:<ex_uuid>``.
    Bad blocks (truncated / unknown compression) are skipped with a warning
    rather than aborting the whole file.
    """
    try:
        data = path.read_bytes()
    except OSError:
        return

    try:
        _metaindex, index_h = _parse_footer(data)
        index_block = _read_raw_block(data, index_h)
    except (ValueError, Exception) as exc:  # noqa: BLE001
        print(f"  [warn] {path.name}: cannot parse footer: {exc}")
        return

    data_handles: list[tuple[int, int]] = []
    for _sep_key, value in _iter_block_entries(index_block):
        try:
            handle, _ = _read_block_handle(value, 0)
            data_handles.append(handle)
        except ValueError:
            continue

    for handle in data_handles:
        try:
            block = _read_raw_block(data, handle)
        except (ValueError, Exception):  # noqa: BLE001
            continue

        for key, value in _iter_block_entries(block):
            if not key.startswith(_EXCHANGE_KEY_PREFIX):
                continue
            m = _EXCHANGE_KEY_RE.match(key)
            if not m:
                continue
            try:
                obj = json.loads(value.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(obj, dict):
                continue

            conv_id = m.group("conv").decode("ascii")
            ex_id = m.group("ex").decode("ascii")

            yield Exchange(
                conversation_id=conv_id,
                exchange_id=ex_id,
                request_text=_pull_request_text(obj),
                response_text=_pull_response_text(obj),
                request_timestamp=_pull_timestamp(obj, "request"),
                response_timestamp=_pull_timestamp(obj, "response"),
                raw=obj,
            )


def _iter_augment_stores(roots: list[Path]) -> Iterable[tuple[Path, Path]]:
    """Yield (workspace_dir, augment_kv_store_dir) pairs."""
    for root in roots:
        for ws in sorted(root.iterdir()):
            if not ws.is_dir():
                continue
            kv = ws / "Augment.vscode-augment" / "augment-kv-store"
            if kv.exists():
                yield ws, kv


def _workspace_label(ws: Path) -> str:
    """Best-effort human-readable label for a workspace dir."""
    wjson = ws / "workspace.json"
    if wjson.exists():
        try:
            data = json.loads(wjson.read_text(encoding="utf-8"))
            folder = data.get("folder") or data.get("workspace") or ""
            if isinstance(folder, str) and folder:
                # file:///c%3A/Users/.../project → project
                from urllib.parse import unquote

                tail = unquote(folder).rstrip("/").split("/")[-1]
                return tail or ws.name
        except (OSError, json.JSONDecodeError):
            pass
    return ws.name


# ---------------------------------------------------------------------------
# Grouping to conversations + output shaping
# ---------------------------------------------------------------------------


def group_exchanges(
    exchanges: list[Exchange], workspace_label: str
) -> list[dict]:
    """Collapse exchanges into the toolkit's conversation JSONL schema."""
    by_conv: dict[str, list[Exchange]] = defaultdict(list)
    for ex in exchanges:
        by_conv[ex.conversation_id].append(ex)

    conversations: list[dict] = []
    for conv_id, items in by_conv.items():
        items.sort(key=Exchange.sort_key)
        messages: list[dict] = []
        for ex in items:
            if ex.request_text:
                msg = {"role": "user", "content": ex.request_text}
                if ex.request_timestamp:
                    msg["timestamp"] = ex.request_timestamp
                messages.append(msg)
            if ex.response_text:
                msg = {"role": "assistant", "content": ex.response_text}
                if ex.response_timestamp:
                    msg["timestamp"] = ex.response_timestamp
                messages.append(msg)

        if not messages:
            continue

        first_user = next(
            (m["content"] for m in messages if m["role"] == "user"), ""
        )
        name = first_user.strip().splitlines()[0][:120] if first_user else (
            f"Augment conversation {conv_id[:8]}"
        )

        conv = {
            "messages": messages,
            "source": "augment-vscode",
            "name": name,
            "conversation_id": conv_id,
            "workspace": workspace_label,
            "exchange_count": len(items),
        }
        ts = items[0].request_timestamp
        if ts:
            try:
                conv["created_at"] = int(
                    datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() * 1000
                )
            except ValueError:
                pass
        conversations.append(conv)

    return conversations


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_markdown_corpus(
    conversations: list[dict], output_dir: Path
) -> int:
    """Emit one .md per conversation, ready for `mempalace mine --mode convos`."""
    md_dir = output_dir / "augment_conversations_markdown"
    md_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for conv in conversations:
        conv_id = conv["conversation_id"]
        ws = (conv.get("workspace") or "unknown").replace("/", "_").replace(
            "\\", "_"
        )
        safe_name = re.sub(r"[^\w\-]+", "_", conv.get("name", ""))[:60]
        fname = f"augment-{ws}-{conv_id[:8]}-{safe_name or 'conv'}.md"
        fpath = md_dir / fname
        lines: list[str] = [
            f"# {conv.get('name', 'Untitled')}",
            "",
            f"- source: augment-vscode",
            f"- workspace: {conv.get('workspace', 'unknown')}",
            f"- conversation_id: {conv_id}",
            f"- exchanges: {conv.get('exchange_count', 0)}",
            "",
        ]
        for msg in conv["messages"]:
            role = msg["role"]
            ts = msg.get("timestamp", "")
            ts_tag = f" _({ts})_" if ts else ""
            lines.append(f"## {role}{ts_tag}")
            lines.append("")
            lines.append(msg["content"])
            lines.append("")
        fpath.write_text("\n".join(lines), encoding="utf-8")
        written += 1
    return written


def write_outputs(
    conversations: list[dict], inventory: dict, output_dir: Path
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = output_dir / f"augment_conversations_{ts}.jsonl"
    inv_path = output_dir / f"augment_conversations_{ts}.inventory.json"

    with jsonl_path.open("w", encoding="utf-8") as fh:
        for conv in conversations:
            fh.write(json.dumps(conv, ensure_ascii=False) + "\n")

    with inv_path.open("w", encoding="utf-8") as fh:
        json.dump(inventory, fh, indent=2)

    md_count = write_markdown_corpus(conversations, output_dir)
    md_dir = output_dir / "augment_conversations_markdown"
    print(f"  markdown corpus: {md_count} files -> {md_dir}")

    return jsonl_path, inv_path, md_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(roots: list[Path], output_dir: Path) -> int:
    print("=" * 78)
    print("AUGMENT (vscode-augment) DATA EXTRACTION")
    print("=" * 78)
    print(f"roots: {len(roots)} root(s)")
    for r in roots:
        print(f"  - {r}")
    print(f"output: {output_dir}")
    print()

    all_conversations: list[dict] = []
    per_workspace_stats: list[dict] = []

    stores = list(_iter_augment_stores(roots))
    if not stores:
        print("No Augment stores found under any VS Code-family root.")
        return 1

    print(f"Found {len(stores)} Augment workspace store(s).")
    for ws_dir, kv_dir in stores:
        label = _workspace_label(ws_dir)
        files = sorted(kv_dir.glob("*.ldb"))
        exchanges: list[Exchange] = []
        for f in files:
            exchanges.extend(_scan_ldb_file(f))
        # Dedup by (conv_id, exchange_id), keep the longest raw dict
        dedup: dict[tuple[str, str], Exchange] = {}
        for ex in exchanges:
            key = (ex.conversation_id, ex.exchange_id)
            prev = dedup.get(key)
            if prev is None or len(str(ex.raw)) > len(str(prev.raw)):
                dedup[key] = ex
        deduped = list(dedup.values())

        convs = group_exchanges(deduped, label)
        all_conversations.extend(convs)

        per_workspace_stats.append(
            {
                "workspace": label,
                "ws_hash": ws_dir.name,
                "ldb_files": len(files),
                "exchanges_found": len(exchanges),
                "exchanges_deduped": len(deduped),
                "conversations": len(convs),
            }
        )
        print(
            f"  {label:40s} ldb={len(files):2d} ex={len(exchanges):4d} "
            f"dedup={len(deduped):4d} conv={len(convs):3d}"
        )

    # Global dedup across workspaces (same convo may appear if user copied state)
    seen_conv_ids: set[str] = set()
    final: list[dict] = []
    for conv in all_conversations:
        if conv["conversation_id"] in seen_conv_ids:
            continue
        seen_conv_ids.add(conv["conversation_id"])
        final.append(conv)

    inventory = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "extractor_version": "1.0.0",
        "workspace_count": len(stores),
        "total_conversations": len(final),
        "total_messages": sum(len(c["messages"]) for c in final),
        "per_workspace": per_workspace_stats,
    }

    jsonl_path, inv_path, md_dir = write_outputs(final, inventory, output_dir)

    print()
    print("=" * 78)
    print("EXTRACTION COMPLETE")
    print("=" * 78)
    print(f"  conversations: {inventory['total_conversations']}")
    print(f"  messages:      {inventory['total_messages']}")
    print(f"  jsonl:         {jsonl_path}")
    print(f"  inventory:     {inv_path}")
    print(f"  markdown:      {md_dir}")
    print()
    print("Next: mempalace mine", md_dir, "--mode convos --wing augment-archive")
    return 0 if final else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract Augment chat history from VS Code-family "
        "workspace leveldb stores into JSONL + per-conversation markdown."
    )
    parser.add_argument(
        "--root",
        type=Path,
        action="append",
        default=None,
        help="Override VS Code workspaceStorage root (may be repeated). "
        "Defaults to %%APPDATA%%/Code/User/workspaceStorage and sibling "
        "Code-Insiders/Cursor/Windsurf/Trae/VSCodium if present.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("extracted_data"),
        help="Output directory (default: ./extracted_data).",
    )
    args = parser.parse_args(argv)

    roots = args.root if args.root else _default_code_roots()
    if not roots:
        print(
            "ERROR: no VS Code-family workspaceStorage directory found.",
            file=sys.stderr,
        )
        return 2
    return run(roots, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
