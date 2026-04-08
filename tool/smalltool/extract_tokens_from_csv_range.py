#!/usr/bin/env python3
"""Extract a range of tokens from a CSV and write them to a txt.

Row indexing:
- "row" refers to data rows (excluding the header).
- start=1 means the first data row right after the header.

Example:
  python tool/smalltool/extract_tokens_from_csv_range.py \
    --csv WorldCAP_pic/WorldCAPdataFor0002/compare_default_vs_attn0002.csv \
    --start 1 --end 137
    
  python tool/smalltool/extract_tokens_from_csv_range.py \
    --csv WorldCAP_pic/WorldCAPdataFor0025/compare_default_vs_attn0025.csv \
    --start 1 --end 200
  python tool/smalltool/extract_tokens_from_csv_range.py \
    --csv WorldCAP_pic/WorldCAPdataFor0041/compare_default_vs_attn0041.csv \
    --start 1 --end 200
    
  python tool/smalltool/extract_tokens_from_csv_range.py \
    --csv WorldCAP_pic/WorldCAPdataFor0087/compare_default_vs_attn0087.csv \
    --start 1 --end 182
    
  python tool/smalltool/extract_tokens_from_csv_range.py \
    --csv WorldCAP_pic/WorldCAPdataFor0159/compare_default_vs_attn0159.csv \
    --start 1 --end 173
  python tool/smalltool/extract_tokens_from_csv_range.py \
    --csv WorldCAP_pic/WorldCAPdataFor0189/compare_default_vs_attn0189.csv \
    --start 1 --end 61
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract tokens from CSV rows [start,end] into a txt file.")
    parser.add_argument("--csv", required=True, type=Path, help="Input CSV file path.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output txt path. Default: next to the CSV, named '<stem>_tokens_<start>_<end>.txt'."
        ),
    )
    parser.add_argument("--column", type=str, default="token", help="Column name to extract (default: token).")
    parser.add_argument("--start", type=int, default=1, help="Start data-row index (1-based, inclusive).")
    parser.add_argument("--end", type=int, default=137, help="End data-row index (1-based, inclusive).")
    return parser.parse_args()


def extract_tokens(csv_path: Path, column: str, start: int, end: int) -> List[str]:
    if start < 1 or end < 1:
        raise ValueError(f"start/end must be >= 1, got start={start}, end={end}")
    if end < start:
        raise ValueError(f"end must be >= start, got start={start}, end={end}")

    tokens: List[str] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header/fieldnames")
        if column not in reader.fieldnames:
            raise ValueError(f"Column '{column}' not found. Available: {reader.fieldnames}")

        for row_idx, row in enumerate(reader, start=1):
            if row_idx < start:
                continue
            if row_idx > end:
                break
            value = (row.get(column) or "").strip()
            if value:
                tokens.append(value)

    return tokens


def main() -> None:
    args = _parse_args()

    csv_path: Path = args.csv
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    out_path: Path
    if args.out is None:
        out_path = csv_path.with_name(f"{csv_path.stem}_tokens_{args.start}_{args.end}.txt")
    else:
        out_path = args.out

    tokens = extract_tokens(csv_path=csv_path, column=args.column, start=args.start, end=args.end)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(tokens) + ("\n" if tokens else ""), encoding="utf-8")

    print(f"[OK] csv: {csv_path}")
    print(f"[OK] out: {out_path}")
    print(f"[OK] extracted tokens: {len(tokens)} (rows {args.start}..{args.end})")


if __name__ == "__main__":
    main()
