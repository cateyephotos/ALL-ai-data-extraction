#!/usr/bin/env python3
"""
Extract Windsurf (Codeium) conversation + plan data — v2.

The upstream ``extract_windsurf.py`` targets VSCode ``ItemTable`` /
``cursorDiskKV`` keys that current Windsurf releases no longer populate
with chat content. This v2 extractor targets the real on-disk layout used
by Windsurf Cascade (the agentic chat):

    ~/.codeium/windsurf/
        cascade/    <uuid>.pb                  # Full Cascade trajectories (ENCRYPTED)
        memories/   <uuid>.pb                  # Entity/fact memories       (ENCRYPTED)
        implicit/   <uuid>.pb                  # Implicit context           (ENCRYPTED)
        brain/      <uuid>/plan.md             # Plan mode notes            (PLAINTEXT)
                    <uuid>/plan_metadata.pbtxt # Plan metadata              (PLAINTEXT)
        user_settings.pb                       # Settings                   (PLAINTEXT)

The ``.pb`` files under ``cascade/``, ``memories/`` and ``implicit/`` are
encrypted at rest. Their first bytes are high-entropy random (no protobuf
tags, no gzip/zlib magic, no ``v10``/``v11`` Chromium OSCrypt prefix).
Attempting AES-GCM with the Chromium ``Local State`` DPAPI key fails, so
Cascade appears to use a separate key scheme that is not publicly documented
at the time of writing.

Rather than block on that, this script:

* **Extracts everything that is plaintext** — ``brain/*/plan.md`` and their
  ``plan_metadata.pbtxt`` pairs become JSONL conversations with clear
  ``source = "windsurf-brain"`` tagging and ``is_partial = true`` flags.
* **Produces a deterministic inventory** of the encrypted material so the
  user knows the scope of what is still locked (count, size, mtimes, ids).
  The inventory is written alongside the JSONL as ``<stem>.inventory.json``.
* **Output schema matches the rest of the toolkit** (``messages``,
  ``source``, ``name``, ``created_at``) so downstream tooling (e.g.
  ``mempalace mine --mode convos``) can ingest the result without knowing
  the difference.

Decryption of Cascade trajectory ``.pb`` files is left as a follow-up:
see the "Decryption status" section in the written inventory for what we
have tried and ruled out.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


_DEFAULT_ROOT_CANDIDATES = [
    Path.home() / ".codeium" / "windsurf",
]


@dataclass
class InventoryEntry:
    """One encrypted-file record for the inventory JSON."""

    kind: str  # "cascade" | "memories" | "implicit"
    uuid: str
    path: str
    size_bytes: int
    mtime_iso: str
    head_hex: str
    has_known_magic: bool


@dataclass
class WindsurfExtraction:
    conversations: list[dict] = field(default_factory=list)
    inventory: list[InventoryEntry] = field(default_factory=list)
    brain_plans: int = 0
    encrypted_files: int = 0
    encrypted_bytes: int = 0

    def to_json_summary(self, root: Path) -> dict:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "extractor_version": "2.0.0",
            "platform": platform.platform(),
            "windsurf_root": str(root),
            "counts": {
                "plaintext_conversations": len(self.conversations),
                "brain_plans": self.brain_plans,
                "encrypted_files": self.encrypted_files,
                "encrypted_bytes": self.encrypted_bytes,
            },
            "encrypted_inventory": [
                {
                    "kind": e.kind,
                    "uuid": e.uuid,
                    "path": e.path,
                    "size_bytes": e.size_bytes,
                    "mtime_iso": e.mtime_iso,
                    "head_hex": e.head_hex,
                    "has_known_magic": e.has_known_magic,
                }
                for e in self.inventory
            ],
            "decryption_status": {
                "status": "LOCKED — unknown key derivation",
                "ruled_out": [
                    "raw protobuf (no recognizable field tags in first bytes)",
                    "gzip / zlib (no magic: 1f8b, 789c, 78da, 7801)",
                    "Chromium OSCrypt key from Windsurf 'Local State' "
                    "(AES-GCM decrypt fails with both 12- and 16-byte IV "
                    "hypotheses)",
                    "DPAPI blob prefix (no 01000000D08C9DDF... header)",
                ],
                "candidates": [
                    "per-file key derived from installation_id + file UUID",
                    "key retrieved at runtime from Codeium backend over mTLS",
                    "key held in OS keychain under a non-default service name",
                ],
                "next_steps": [
                    "Dump strings from language_server_windows_x64.exe for "
                    "AES/Chacha/SecretBox symbol references",
                    "Watch localhost traffic between editor and language "
                    "server on startup; trajectories are returned "
                    "decrypted there",
                    "Request a user-facing export API from Windsurf / Codeium",
                ],
            },
        }


# ---------------------------------------------------------------------------
# Locate Windsurf data root
# ---------------------------------------------------------------------------


def find_windsurf_root(explicit: Path | None) -> Path | None:
    """Return the first existing Windsurf/Cascade data root."""
    if explicit is not None:
        return explicit if explicit.exists() else None
    for cand in _DEFAULT_ROOT_CANDIDATES:
        if cand.exists():
            return cand
    return None


# ---------------------------------------------------------------------------
# Brain plans (plaintext)
# ---------------------------------------------------------------------------


_PBTXT_KV_RE = re.compile(r'^\s*(\w+)\s*:\s*"?([^"\n]*?)"?\s*$')


def _parse_pbtxt(path: Path) -> dict[str, str]:
    """Very small text-protobuf parser — sufficient for plan_metadata.pbtxt."""
    out: dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            m = _PBTXT_KV_RE.match(line)
            if m:
                out[m.group(1)] = m.group(2)
    except OSError:
        pass
    return out


def extract_brain_plans(root: Path) -> list[dict]:
    """Convert each ``brain/<uuid>/plan.md`` into a JSONL conversation.

    These are the AI's planning notes from a Cascade session. They are the
    only plaintext conversational artifact Windsurf writes to disk, so they
    are the most informative thing we can feed mempalace today.
    """
    conversations: list[dict] = []
    brain_root = root / "brain"
    if not brain_root.exists():
        return conversations

    for session_dir in sorted(brain_root.iterdir()):
        if not session_dir.is_dir():
            continue
        plan_md = session_dir / "plan.md"
        if not plan_md.exists():
            continue
        meta_pb = session_dir / "plan_metadata.pbtxt"
        metadata = _parse_pbtxt(meta_pb) if meta_pb.exists() else {}

        try:
            body = plan_md.read_text(encoding="utf-8")
        except OSError:
            continue
        mtime = datetime.fromtimestamp(plan_md.stat().st_mtime, tz=timezone.utc)

        name = body.splitlines()[0].lstrip("# ").strip() if body.strip() else (
            f"Cascade plan {session_dir.name[:8]}"
        )

        conversations.append(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": body,
                        "content_type": "plan_md",
                        "timestamp": mtime.isoformat(),
                    }
                ],
                "source": "windsurf-brain",
                "name": name,
                "session_id": session_dir.name,
                "created_at": int(mtime.timestamp() * 1000),
                "metadata": metadata,
                "is_partial": True,
                "partial_reason": (
                    "Plan-mode notes only; full Cascade trajectory lives in "
                    "encrypted cascade/*.pb and cannot be decrypted by this "
                    "extractor."
                ),
            }
        )
    return conversations


# ---------------------------------------------------------------------------
# Encrypted-file inventory
# ---------------------------------------------------------------------------


_KNOWN_MAGICS = (
    b"\x1f\x8b",          # gzip
    b"\x78\x9c",          # zlib default
    b"\x78\xda",          # zlib best
    b"\x78\x01",          # zlib none
    b"v10",               # Chromium OSCrypt v10
    b"v11",               # Chromium OSCrypt v11
    b"\x28\xb5\x2f\xfd",  # zstd
)


def _walk_pb(root: Path) -> Iterable[Path]:
    if not root.exists():
        return
    for f in root.iterdir():
        if f.is_file() and f.suffix == ".pb":
            yield f


def inventory_encrypted_files(root: Path) -> list[InventoryEntry]:
    """Record every ``*.pb`` file in cascade/memories/implicit."""
    entries: list[InventoryEntry] = []
    for kind in ("cascade", "memories", "implicit"):
        for pb in _walk_pb(root / kind):
            try:
                size = pb.stat().st_size
                mtime = datetime.fromtimestamp(
                    pb.stat().st_mtime, tz=timezone.utc
                )
                with pb.open("rb") as fh:
                    head = fh.read(16)
            except OSError:
                continue
            has_magic = any(head.startswith(m) for m in _KNOWN_MAGICS)
            entries.append(
                InventoryEntry(
                    kind=kind,
                    uuid=pb.stem,
                    path=str(pb),
                    size_bytes=size,
                    mtime_iso=mtime.isoformat(),
                    head_hex=head.hex(),
                    has_known_magic=has_magic,
                )
            )
    return entries


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_outputs(
    extraction: WindsurfExtraction, output_dir: Path, root: Path
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = output_dir / f"windsurf_conversations_{ts}.jsonl"
    inventory_path = output_dir / f"windsurf_conversations_{ts}.inventory.json"

    with jsonl_path.open("w", encoding="utf-8") as fh:
        for conv in extraction.conversations:
            fh.write(json.dumps(conv, ensure_ascii=False) + "\n")

    with inventory_path.open("w", encoding="utf-8") as fh:
        json.dump(extraction.to_json_summary(root), fh, indent=2)

    return jsonl_path, inventory_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(root: Path, output_dir: Path) -> int:
    print("=" * 78)
    print("WINDSURF COMPLETE DATA EXTRACTION v2 (Cascade brain + inventory)")
    print("=" * 78)
    print(f"root:   {root}")
    print(f"output: {output_dir}")
    print()

    extraction = WindsurfExtraction()

    print("[1/2] Reading plaintext brain plans...")
    extraction.conversations = extract_brain_plans(root)
    extraction.brain_plans = len(extraction.conversations)
    print(f"      {extraction.brain_plans} plan(s) extracted")

    print("[2/2] Inventorying encrypted Cascade artifacts...")
    extraction.inventory = inventory_encrypted_files(root)
    extraction.encrypted_files = len(extraction.inventory)
    extraction.encrypted_bytes = sum(e.size_bytes for e in extraction.inventory)
    print(
        f"      {extraction.encrypted_files} encrypted file(s), "
        f"{extraction.encrypted_bytes / 1e6:.1f} MB "
        f"(split: "
        + ", ".join(
            f"{k}={sum(1 for e in extraction.inventory if e.kind == k)}"
            for k in ("cascade", "memories", "implicit")
        )
        + ")"
    )

    jsonl_path, inventory_path = write_outputs(extraction, output_dir, root)
    print()
    print("Wrote:")
    print(f"  {jsonl_path}")
    print(f"  {inventory_path}")

    if extraction.brain_plans == 0 and extraction.encrypted_files == 0:
        print("\nNothing found under", root)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract Windsurf Cascade brain plans + encrypted-file "
        "inventory into JSONL for downstream ingest (e.g. "
        "`mempalace mine --mode convos`)."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Override Windsurf data root. "
        "Default: ~/.codeium/windsurf if it exists.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("extracted_data"),
        help="Directory for the JSONL + inventory JSON (default: "
        "./extracted_data).",
    )
    args = parser.parse_args(argv)

    root = find_windsurf_root(args.root)
    if root is None:
        print(
            "ERROR: no Windsurf data root found. Expected "
            "~/.codeium/windsurf or pass --root.",
            file=sys.stderr,
        )
        return 2
    return run(root, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
