#!/usr/bin/env bash
set -euo pipefail

SRC_ROOT="${SRC_ROOT:-/pfs/shared_eval/datasets/images/MME}"
SITE_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "[INFO] 拷贝 selected_manifest.csv 中的图像到 $SITE_DIR/images/ 下，保持相对路径结构"
python3 - << 'PYCODE'
import csv, os, shutil, sys
from pathlib import Path

SRC_ROOT = os.environ.get("SRC_ROOT", "/pfs/shared_eval/datasets/images/MME")
SITE_DIR = Path(os.path.dirname(__file__))
dst_root = SITE_DIR / "images"
dst_root.mkdir(parents=True, exist_ok=True)

manifest = SITE_DIR / "selected_manifest.csv"
total = 0
copied = 0
missing = 0

def is_abs_path(p: str) -> bool:
    p = p.strip()
    return p.startswith("/") or (len(p) > 1 and p[1] == ":" and ("\\" in p or "/" in p))

with manifest.open(newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rel = (row.get("image_relpath") or "").replace("\r\n","\n").replace("\r","\n").strip().strip('"')
        if not rel:
            continue
        total += 1
        src = Path(rel) if is_abs_path(rel) else (Path(SRC_ROOT) / rel)
        dst = dst_root / rel.lstrip("/")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            try:
                if not dst.exists():
                    shutil.copy2(src, dst)
                copied += 1
            except Exception as e:
                print(f"[WARN] copy fail: {src} -> {dst}: {e}", file=sys.stderr)
        else:
            print(f"[MISS] {src}", file=sys.stderr)
            missing += 1
print(f"[DONE] total={total}, copied={copied}, missing={missing}")
PYCODE
