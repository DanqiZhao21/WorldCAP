#!/usr/bin/env python3
"""Compare two NavSim/WoTE evaluation CSVs and export visualization-friendly tokens.

This script aligns rows by `token` and compares metrics between an *earlier* and a
*later* CSV (determined by timestamp in filename when possible).

Typical CSV columns (from navsim evaluation):
  , token, valid,
  no_at_fault_collisions,
  drivable_area_compliance,
  driving_direction_compliance,
  ego_progress,
  time_to_collision_within_bound,
  comfort,
  score,
  (optional) oncoming_progress

Outputs a new CSV with one row per token, including both scores, delta, collision
status, and a `reason` label suitable for filtering/visualization.

Example:
  python tool/smalltool/compare_eval_csvs.py \
    --csv_a /path/to/2026.02.27.20.02.31.csv \
    --csv_b /path/to/2026.02.27.21.16.35.csv \
    --out /path/to/compare_default_vs_film03.csv
    
  python tool/smalltool/compare_eval_csvs.py \
    --csv_a /mnt/data/navsim_workspace/exp/eval/WoTE/ckpt_20260225/0002/default/2026.02.26.10.01.42.csv \
    --csv_b /mnt/data/navsim_workspace/exp/eval/WoTE/ckpt_20260225/0002/attn/2026.02.26.10.43.06.csv \
    --out /home/zhaodanqi/clone/WoTE/WorldCAPdataFor0002/compare_default_vs_attn0002.csv

  python tool/smalltool/compare_eval_csvs.py \
    --csv_a /mnt/data/navsim_workspace/exp/eval/WoTE/ckpt_20260225/0025/default/2026.02.27.09.49.51.csv \
    --csv_b /mnt/data/navsim_workspace/exp/eval/WoTE/ckpt_20260225/0025/attn/2026.02.27.10.40.36.csv \
    --out /home/zhaodanqi/clone/WoTE/WorldCAPdataFor0025/compare_default_vs_attn0025.csv

  python tool/smalltool/compare_eval_csvs.py \
    --csv_a /mnt/data/navsim_workspace/exp/eval/WoTE/ckpt_20260225/0041/default/2026.02.27.11.59.48.csv \
    --csv_b /mnt/data/navsim_workspace/exp/eval/WoTE/ckpt_20260225/0041/attn/2026.02.27.12.49.58.csv \
    --out /home/zhaodanqi/clone/WoTE/WorldCAPdataFor0041/compare_default_vs_attn0041.csv

  python tool/smalltool/compare_eval_csvs.py \
    --csv_a /mnt/data/navsim_workspace/exp/eval/WoTE/ckpt_20260225/0087/default/2026.02.26.05.43.18.csv \
    --csv_b /mnt/data/navsim_workspace/exp/eval/WoTE/ckpt_20260225/0087/attn/2026.02.26.07.01.46.csv \
    --out /home/zhaodanqi/clone/WoTE/WorldCAPdataFor0087/compare_default_vs_attn0087.csv

  python tool/smalltool/compare_eval_csvs.py \
    --csv_a /mnt/data/navsim_workspace/exp/eval/WoTE/ckpt_20260225/0159/default/2026.02.26.02.18.53.csv \
    --csv_b /mnt/data/navsim_workspace/exp/eval/WoTE/ckpt_20260225/0159/attn/2026.02.26.04.10.50.csv \
    --out /home/zhaodanqi/clone/WoTE/WorldCAPdataFor0159/compare_default_vs_attn0159.csv
    
  python tool/smalltool/compare_eval_csvs.py \
    --csv_a /mnt/data/navsim_workspace/exp/eval/WoTE/ckpt_20260225/0189/default/2026.02.25.12.27.04.csv \
    --csv_b /mnt/data/navsim_workspace/exp/eval/WoTE/ckpt_20260225/0189/attn/2026.02.25.13.07.38.csv \
    --out /home/zhaodanqi/clone/WoTE/WorldCAPdataFor0189/compare_default_vs_attn0189.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple


TS_RE = re.compile(r"(?P<ts>\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2})")


METRICS = [
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "score",
]


@dataclass(frozen=True)
class CsvInfo:
    path: str
    timestamp: Optional[datetime]


def _parse_timestamp_from_path(path: str) -> Optional[datetime]:
    base = os.path.basename(path)
    m = TS_RE.search(base)
    if not m:
        return None
    ts = m.group("ts")
    try:
        return datetime.strptime(ts, "%Y.%m.%d.%H.%M.%S")
    except ValueError:
        return None


def _read_eval_csv(path: str) -> Dict[str, Dict[str, str]]:
    """Return token -> row (raw strings), skipping token=='average'."""
    token_to_row: Dict[str, Dict[str, str]] = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        if "token" not in reader.fieldnames:
            raise ValueError(f"CSV missing 'token' column: {path}. fields={reader.fieldnames}")
        for row in reader:
            tok = (row.get("token") or "").strip()
            if not tok or tok == "average":
                continue
            token_to_row[tok] = row
    return token_to_row


def _to_float(v: Optional[str]) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _to_bool(v: Optional[str]) -> Optional[bool]:
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in {"true", "1", "yes"}:
        return True
    if s in {"false", "0", "no"}:
        return False
    return None


def _is_collision_free(no_at_fault_collisions: Optional[float]) -> Optional[bool]:
    if no_at_fault_collisions is None:
        return None
    # In these logs, 1.0 means no collision, 0.0 means collision.
    return no_at_fault_collisions >= 0.5


def _choose_earlier_later(a_path: str, b_path: str) -> Tuple[CsvInfo, CsvInfo, str]:
    a_ts = _parse_timestamp_from_path(a_path)
    b_ts = _parse_timestamp_from_path(b_path)
    a = CsvInfo(a_path, a_ts)
    b = CsvInfo(b_path, b_ts)

    if a_ts and b_ts:
        if a_ts <= b_ts:
            return a, b, "filename_timestamp"
        return b, a, "filename_timestamp"

    # Fallback: mtime
    try:
        a_m = os.path.getmtime(a_path)
        b_m = os.path.getmtime(b_path)
        if a_m <= b_m:
            return a, b, "mtime"
        return b, a, "mtime"
    except OSError:
        # Last resort: keep given order
        return a, b, "given_order"


def _compute_reason(
    score_early: Optional[float],
    score_late: Optional[float],
    coll_free_early: Optional[bool],
    coll_free_late: Optional[bool],
    drivable_early: Optional[float],
    drivable_late: Optional[float],
    ttc_early: Optional[float],
    ttc_late: Optional[float],
    eps: float,
) -> str:
    # Priority: collision fixed > new collision > score improved > score regressed > unchanged
    if coll_free_early is False and coll_free_late is True:
        return "collision_fixed"
    if coll_free_early is True and coll_free_late is False:
        return "collision_regressed"

    # Additional visualization-friendly categories
    if drivable_early is not None and drivable_late is not None:
        if drivable_early < 0.5 and drivable_late >= 0.5:
            return "drivable_fixed"

    if ttc_early is not None and ttc_late is not None:
        # Higher time_to_collision_within_bound is better.
        if ttc_late > ttc_early + eps:
            return "ttc_improved"

    if score_early is None or score_late is None:
        return "missing_score"
    if score_late > score_early + eps:
        return "score_improved"
    if score_late < score_early - eps:
        return "score_regressed"
    return "score_unchanged"


def compare(csv_a: str, csv_b: str, out_csv: str, eps: float, topk: int) -> None:
    early, late, order_by = _choose_earlier_later(csv_a, csv_b)

    early_map = _read_eval_csv(early.path)
    late_map = _read_eval_csv(late.path)

    common_tokens = sorted(set(early_map.keys()) & set(late_map.keys()))

    rows: List[Dict[str, object]] = []
    for tok in common_tokens:
        r0 = early_map[tok]
        r1 = late_map[tok]

        score0 = _to_float(r0.get("score"))
        score1 = _to_float(r1.get("score"))
        delta = None
        if score0 is not None and score1 is not None:
            delta = score1 - score0

        coll0 = _to_float(r0.get("no_at_fault_collisions"))
        coll1 = _to_float(r1.get("no_at_fault_collisions"))
        free0 = _is_collision_free(coll0)
        free1 = _is_collision_free(coll1)

        drv0 = _to_float(r0.get("drivable_area_compliance"))
        drv1 = _to_float(r1.get("drivable_area_compliance"))
        ttc0 = _to_float(r0.get("time_to_collision_within_bound"))
        ttc1 = _to_float(r1.get("time_to_collision_within_bound"))

        reason = _compute_reason(score0, score1, free0, free1, drv0, drv1, ttc0, ttc1, eps)

        flag_collision_fixed = (free0 is False and free1 is True)
        flag_drivable_fixed = (drv0 is not None and drv1 is not None and drv0 < 0.5 and drv1 >= 0.5)
        flag_ttc_improved = (ttc0 is not None and ttc1 is not None and ttc1 > ttc0 + eps)

        row: Dict[str, object] = {
            "token": tok,
            "reason": reason,
            "score_early": score0,
            "score_late": score1,
            "delta_score": delta,
            "collision_free_early": free0,
            "collision_free_late": free1,
            "flag_collision_fixed": flag_collision_fixed,
            "flag_drivable_fixed": flag_drivable_fixed,
            "flag_ttc_improved": flag_ttc_improved,
            "valid_early": _to_bool(r0.get("valid")),
            "valid_late": _to_bool(r1.get("valid")),
        }

        # Keep metric deltas for visualization (optional columns)
        for k in METRICS:
            if k == "score":
                continue
            v0 = _to_float(r0.get(k))
            v1 = _to_float(r1.get(k))
            row[f"{k}_early"] = v0
            row[f"{k}_late"] = v1
            if v0 is not None and v1 is not None:
                row[f"delta_{k}"] = v1 - v0
            else:
                row[f"delta_{k}"] = None

        rows.append(row)

    # Filter: visualization-friendly token sets
    score_improved = [r for r in rows if r["reason"] == "score_improved"]
    collision_fixed = [r for r in rows if r["reason"] == "collision_fixed"]
    drivable_fixed = [r for r in rows if r["reason"] == "drivable_fixed"]
    ttc_improved = [r for r in rows if r["reason"] == "ttc_improved"]

    # Sort for visualization
    score_improved.sort(key=lambda r: (r["delta_score"] is None, -(r["delta_score"] or 0.0)))
    collision_fixed.sort(key=lambda r: (r["delta_score"] is None, -(r["delta_score"] or 0.0)))
    drivable_fixed.sort(key=lambda r: (r["delta_score"] is None, -(r["delta_score"] or 0.0)))
    ttc_improved.sort(key=lambda r: (r["delta_score"] is None, -(r["delta_score"] or 0.0)))

    picked: List[Dict[str, object]] = []
    picked.extend(collision_fixed[:topk])
    picked.extend(drivable_fixed[:topk])
    picked.extend(ttc_improved[:topk])
    picked.extend(score_improved[:topk])

    # De-duplicate preserving order
    seen = set()
    deduped: List[Dict[str, object]] = []
    for r in picked:
        tok = str(r["token"])
        if tok in seen:
            continue
        seen.add(tok)
        deduped.append(r)

    # Always write something (even if empty)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    fieldnames = [
        "token",
        "reason",
        "score_early",
        "score_late",
        "delta_score",
        "collision_free_early",
        "collision_free_late",
        "flag_collision_fixed",
        "flag_drivable_fixed",
        "flag_ttc_improved",
        "valid_early",
        "valid_late",
    ]
    for k in METRICS:
        if k == "score":
            continue
        fieldnames.extend([f"{k}_early", f"{k}_late", f"delta_{k}"])

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in deduped:
            writer.writerow({k: r.get(k) for k in fieldnames})

    # Console summary
    print("[INFO] order_by:", order_by)
    print("[INFO] early:", early.path)
    print("[INFO] late :", late.path)
    print("[INFO] tokens common:", len(common_tokens))
    print("[INFO] collision_fixed:", len(collision_fixed))
    print("[INFO] drivable_fixed:", len(drivable_fixed))
    print("[INFO] ttc_improved   :", len(ttc_improved))
    print("[INFO] score_improved :", len(score_improved))
    print("[INFO] wrote:", out_csv, "rows:", len(deduped))


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--csv_a", required=True, help="Path to first CSV (any order)")
    p.add_argument("--csv_b", required=True, help="Path to second CSV (any order)")
    p.add_argument("--out", default=None, help="Output CSV path")
    p.add_argument("--eps", type=float, default=1e-9, help="Tolerance for score compare")
    p.add_argument("--topk", type=int, default=200, help="Max tokens per category to include")
    return p


def main() -> None:
    args = build_argparser().parse_args()

    if args.out:
        out_csv = args.out
    else:
        # default near later csv
        early, late, _ = _choose_earlier_later(args.csv_a, args.csv_b)
        base0 = os.path.basename(early.path)
        base1 = os.path.basename(late.path)
        out_csv = os.path.join(os.path.dirname(late.path), f"compare_{base0}_vs_{base1}.csv")

    compare(args.csv_a, args.csv_b, out_csv, eps=float(args.eps), topk=int(args.topk))


if __name__ == "__main__":
    main()
