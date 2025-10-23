#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_mme_gallery.py  (E/F 强制版 + D 列 question；F 整格作为 prediction，保留换行)

- E 列作为真解（gt），F 列作为模型预测（pred），D 列作为 question。
- F / D 单元格整格使用，保留内部换行；CSV 以 QUOTE_ALL 输出保障换行安全。
- 判错：仅对 pred / gt 做轻量规范化（CRLF->LF，strip），不改动 question。
"""

import argparse
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import zipfile
from typing import List, Optional
import csv

CSS_TEXT = """
:root{--bg:#0b0d12;--fg:#e8edf2;--muted:#93a1b1;--card:#141a22;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;}
header{padding:20px 24px;border-bottom:1px solid #22303a;position:sticky;top:0;background:linear-gradient(0deg, rgba(11,13,18,0.75), rgba(11,13,18,0.98));backdrop-filter:blur(6px);}
h1{margin:0;font-size:20px;font-weight:700}
.meta{color:var(--muted);font-size:12px;margin-top:4px}
.container{padding:24px;max-width:1400px;margin:0 auto}
.section{margin:24px 0}
.section h2{font-size:18px;margin:12px 0 8px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:14px}
.card{background:var(--card);border:1px solid #22303a;border-radius:14px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,.25)}
.thumb{width:100%;height:200px;object-fit:contain;background:#0f141b}
.info{padding:10px 12px;font-size:12px;line-height:1.4}
.badge{border:1px solid #2a3a46;background:#0e1319;padding:2px 6px;border-radius:999px;font-size:11px;color:#a7c1d9}
.code{font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;color:#cbd5e1}
footer{color:#778da3;font-size:12px;padding:20px 24px;border-top:1px solid #22303a}
a{color:#7cc5ff;text-decoration:none}
a:hover{text-decoration:underline}
.small{font-size:11px;color:#9ab}
"""

COPY_IMAGES_SH = r"""#!/usr/bin/env bash
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
"""

def html_escape(s: str) -> str:
    return (s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
              .replace('"',"&quot;").replace("'","&#39;"))

def detect_columns(df: pd.DataFrame) -> dict:
    cols = list(df.columns)
    def find_col(keys: List[str]) -> List[str]:
        res = []
        for c in cols:
            lc = c.lower()
            if any(k in lc for k in keys):
                res.append(c)
        return res

    class_cols = find_col(["class","cate","task","type","subcat","category"])
    image_cols = find_col(["image","img","path","file","pic"])
    score_cols = find_col(["score","logit","prob","conf","similarity"])

    def choose(prefer: List[str], cands: List[str]) -> Optional[str]:
        for key in prefer:
            for c in cands:
                if c.lower() == key:
                    return c
        return cands[0] if cands else None

    picked = {
        "class": choose(["class","category","cate","task","type","subcat"], class_cols) or (["_CLASS"] if "_CLASS" in cols else None),
        "image": choose(["image_path","img_path","image","img","path","file","image_relpath","relpath"], image_cols),
        "score": score_cols[0] if score_cols else None,
    }

    if picked["class"] is None:
        df["_CLASS"] = "ALL"
        picked["class"] = "_CLASS"

    if picked["image"] is None:
        path_like = []
        for c in cols:
            vals = df[c].astype(str).head(200)
            if any(("/" in v) or v.lower().endswith((".jpg",".jpeg",".png",".bmp",".gif",".webp")) for v in vals):
                path_like.append(c)
        picked["image"] = path_like[0] if path_like else None

    return picked

def norm_for_compare(s):
    """用于 pred/gt 的轻量规范化：保留内部换行，CRLF->LF，然后 strip。"""
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return s.strip()

def build_site(df: pd.DataFrame, per_class: int, output_dir: Path, tag: str, it: int,
               src_image_root: Optional[str], zip_output: bool,
               col_class: Optional[str], col_image: Optional[str]) -> None:
    # 强制 D/E/F 列：question / gt / pred
    if len(df.columns) < 6:
        raise ValueError("表格列数不足：需要至少 6 列以使用 D/E/F。")
    q_col = df.columns[3]   # D question
    gt_col = df.columns[4]  # E gt
    pred_col = df.columns[5]# F pred（整格，保留换行）

    # 自动/覆盖 其他列
    picks_auto = detect_columns(df)
    if col_class:
        picks_auto["class"] = col_class
    if col_image:
        picks_auto["image"] = col_image

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "assets").mkdir(exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)

    (output_dir / "assets" / "styles.css").write_text(CSS_TEXT, encoding="utf-8")

    # 错误样例：pred != gt（仅比较规范化后的文本）
    cmp_pred = df[pred_col].apply(norm_for_compare)
    cmp_gt = df[gt_col].apply(norm_for_compare)
    wrong_mask = (cmp_pred != cmp_gt)
    wrong_df = df[wrong_mask].copy()

    selected = []
    for cls, g in wrong_df.groupby(picks_auto["class"]):
        selected.append(g.head(per_class))
    sel = pd.concat(selected, axis=0) if selected else wrong_df.head(0)

    def keep(s):
        """写出 manifest 时保留原始文本（含内部换行）。"""
        return "" if pd.isna(s) else str(s)

    score_col = picks_auto.get("score")
    if score_col not in df.columns:
        score_col = None

    manifest_rows = []
    for _, r in sel.iterrows():
        manifest_rows.append({
            "class": keep(r.get(picks_auto["class"], "")),
            "image_relpath": keep(r.get(picks_auto["image"], "")),
            "question": keep(r.get(q_col, "")),  # 新增：D 列
            "gt": keep(r.get(gt_col, "")),
            "pred": keep(r.get(pred_col, "")),
            "score": keep(r.get(score_col, "")) if score_col else "",
        })
    man = pd.DataFrame(manifest_rows)
    man.to_csv(output_dir / "selected_manifest.csv", index=False, quoting=csv.QUOTE_ALL)

    # HTML
    by_class = {}
    for rec in manifest_rows:
        by_class.setdefault(rec["class"], []).append(rec)

    def pre_block(text: str) -> str:
        esc = html_escape(text)
        return f"<pre class=\"code\" style=\"white-space:pre-wrap;word-wrap:break-word;margin:6px 0 0 0;\">{esc}</pre>"

    def card(rec):
        rel = rec["image_relpath"]
        img_src = f"images/{rel}" if rel else "assets/missing.png"
        q = rec.get("question","")
        gt = rec.get("gt","")
        pd_ = rec.get("pred","")
        score = rec.get("score","")
        parts = []
        parts.append(f'<img class="thumb" loading="lazy" src="{img_src}" alt="img">')
        meta = []
        if q:
            meta.append(f'<div><span class="badge">question</span> {pre_block(q)}</div>')
        if pd_ or gt:
            meta.append(f'<div><span class="badge">pred</span> {pre_block(pd_)}</div>')
            meta.append(f'<div><span class="badge">gt</span> {pre_block(gt)}</div>')
        if score:
            meta.append(f'<div><span class="badge">score</span> <span class="code">{html_escape(score)}</span></div>')
        if rel:
            meta.append(f'<div class="small">{html_escape(rel)}</div>')
        return f'<div class="card">{"".join(parts)}<div class="info">{"".join(meta)}</div></div>'

    sections = []
    for cls in sorted(by_class.keys()):
        cards = "\n".join(card(r) for r in by_class[cls])
        sections.append(f"""
    <div class="section" id="{html_escape(cls)}">
      <h2>{html_escape(cls)} <span class="small">({len(by_class[cls])} samples)</span></h2>
      <div class="grid">
        {cards}
      </div>
    </div>""")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header_meta = f"Built {now}. Source: {{TABLE_NAME}} · tag={tag} · iter={it}"

    index_html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MME 错误样例画廊（每类{per_class}个）</title>
  <link rel="stylesheet" href="assets/styles.css" />
</head>
<body>
  <header>
    <h1>MME 错误样例画廊（每类{per_class}个）</h1>
    <div class="meta">{{html_escape(header_meta)}}</div>
    <div class="meta">部署：将 <code>{{html_escape(str(output_dir))}}</code> 推到 GitHub Pages；先运行 <code>copy_images.sh</code> 拷图到 <code>images/</code>。</div>
  </header>
  <div class="container">
    {''.join(sections) if sections else "<p>未检测到错误样例。</p>"}
  </div>
  <footer>
    <div>自动生成页面。</div>
  </footer>
</body>
</html>
"""
    (output_dir / "index.html").write_text(index_html, encoding="utf-8")

    (output_dir / "copy_images.sh").write_text(COPY_IMAGES_SH, encoding="utf-8")
    os.chmod(output_dir / "copy_images.sh", 0o755)

    # 可选打包
    if zip_output:
        zip_path = output_dir.parent / f"{output_dir.name}.zip"
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in output_dir.rglob("*"):
                zf.write(p, p.relative_to(output_dir.parent))
        print(f"[OK] 打包完成: {zip_path}")

    print(f"[INFO] D/E/F 列：question={q_col}  gt={gt_col}  pred={pred_col}（保留内部换行）")
    print(f"[INFO] 错误样例数：{wrong_mask.sum()}")
    print(f"[OK] 已输出：{output_dir}")

def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() in [".xlsx",".xls"]:
        return pd.read_excel(p)
    elif p.suffix.lower() in [".csv",".tsv"]:
        sep = "," if p.suffix.lower()==".csv" else "\t"
        return pd.read_csv(p, sep=sep)
    else:
        try:
            return pd.read_excel(p)
        except:
            return pd.read_csv(p)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True, help="结果表路径（xlsx/csv）")
    ap.add_argument("--src-image-root", default="/pfs/shared_eval/datasets/images/MME", help="原始图片根目录（可在运行 copy_images.sh 时覆盖）")
    ap.add_argument("--output-dir", default="./site", help="输出站点目录")
    ap.add_argument("--per-class", type=int, default=20, help="每类选取错误样例数上限")
    ap.add_argument("--tag", default="train_qwendit_unify_interleave_stage1p5")
    ap.add_argument("--iter", type=int, default=40000)
    ap.add_argument("--zip", action="store_true", help="额外打包成 zip")
    ap.add_argument("--col-class", default=None, help="覆盖类别列名")
    ap.add_argument("--col-image", default=None, help="覆盖图片路径列名")
    args = ap.parse_args()

    df = read_table(args.table)
    out_dir = Path(args.output_dir)
    build_site(
        df=df,
        per_class=args.per_class,
        output_dir=out_dir,
        tag=args.tag,
        it=args.iter,
        src_image_root=args.src_image_root,
        zip_output=args.zip,
        col_class=args.col_class,
        col_image=args.col_image,
    )
