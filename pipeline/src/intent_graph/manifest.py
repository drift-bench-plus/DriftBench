"""Input-dataset checksums.

Ground truth is only reproducible if the inputs are.  ``verify`` defaults to a size+mtime
fast path because the WebShop catalog is 5.1 GB and hashing it on every data-marked test
would cost minutes; ``--full`` recomputes sha256.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

MANIFEST_NAME = "data_manifest.yaml"

TRACKED = {
    "webshop_repo": ["data/items_human_ins.json", "data/items_ins_v2.json",
                     "data/items_shuffle.json", "data/items_shuffle_1000.json",
                     "data/items_ins_v2_1000.json"],
    "webshop_derived": ["catalog_index.parquet", "human_goals_joined.parquet",
                        "human_ins_products_full.jsonl"],
    "agentbench_repo": ["data/dbbench/db_out_new.jsonl", "data/dbbench/standard.jsonl",
                        "data/dbbench/extend_std.json",
                        "data/os_interaction/train_0317/training.json"],
}


def _sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def _entries(cfg: dict, *, with_hash: bool) -> dict:
    out: dict[str, dict] = {}
    for root_key, rels in TRACKED.items():
        root = Path(cfg["paths"][root_key])
        for rel in rels:
            p = root / rel
            key = f"{root_key}/{rel}"
            if not p.exists():
                out[key] = {"present": False}
                continue
            st = p.stat()
            entry = {"present": True, "size": st.st_size, "mtime": int(st.st_mtime)}
            if with_hash:
                entry["sha256"] = _sha256(p)
            out[key] = entry
    return out


def manifest_path(cfg: dict) -> Path:
    return Path(cfg["paths"]["artifacts"]).parent / "config" / MANIFEST_NAME


def write_manifest(cfg: dict) -> Path:
    path = manifest_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(_entries(cfg, with_hash=True), sort_keys=True), encoding="utf-8")
    return path


def verify_manifest(cfg: dict, *, full: bool = False) -> list[dict]:
    path = manifest_path(cfg)
    if not path.exists():
        return [{"issue": "manifest_missing", "path": str(path)}]
    recorded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    current = _entries(cfg, with_hash=full)
    problems = []
    for key, rec in recorded.items():
        cur = current.get(key, {"present": False})
        if rec.get("present") and not cur.get("present"):
            problems.append({"issue": "missing", "file": key})
        elif rec.get("present"):
            if cur["size"] != rec["size"]:
                problems.append({"issue": "size_changed", "file": key})
            elif full and cur.get("sha256") != rec.get("sha256"):
                problems.append({"issue": "hash_changed", "file": key})
    return problems
