"""One-shot: merge an external fm_failure_modes CSV into this repo's state.

The failure-mode pipeline's timeline graph is rebuilt from the per-frame flag
store (``fm_frame_flags/<video>.json``), not from ``fm_failure_modes.csv``.
Appending rows to the CSV alone is therefore invisible to the graph. This
script does both:

1. Appends every CSV row from ``--source-csv`` that is NOT already present in
   ``--dest-csv`` (de-duped by ``Video`` filename).
2. For each appended row, writes a synthesised minimal flag-store JSON so the
   timeline rebuild places dots for it. The synthesised payload has
   ``total_frames=2`` and writes ``[True, True]`` for every triggered FM (so
   the ``MIN_CONSECUTIVE_FRAMES=2`` rule for FM1/FM2/FM4 still passes) and
   ``[False, False]`` for untriggered ones. FM3 has no run rule, so a single
   True frame would suffice — we use two for uniformity.

Synthesised flag files are NOT a substitute for real processing: they carry
no white-pixel series, no run lengths, no debug imagery, and the
``total_frames=2`` is a stub. They exist purely so the timeline reflects the
fact that these videos were classified by an earlier run.

The dest CSV is copied to ``<dest-csv>.bak-<timestamp>`` before any writes.

Usage::

    python scripts/merge_external_fm_csv.py \\
        --source-csv path/to/other/fm_failure_modes.csv \\
        --dest-csv anomaly_classification/fm_failure_modes.csv \\
        --flag-store-dir anomaly_classification/fm_frame_flags

After the merge, regenerate the graphs::

    python pipelines/run__failure_mode_detection.py --graph-only
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.fm_state_store import (
    CSV_HEADER,
    FM_ORDER,
    append_csv_row,
    ensure_csv_header,
    flag_store_path,
    load_processed_videos,
    save_frame_flags,
)
from shared_functions.timestamp_utils import parse_video_timestamp

# Matches DEFAULT_FPS in pipelines/run__failure_mode_detection.py. The
# synthesised payload's fps is only used to derive a video end-time on the
# timeline; with total_frames=2 the resulting duration is ~13 ms, which is
# imperceptible on a multi-day axis.
DEFAULT_FPS = 155.0
SYNTHETIC_FRAME_COUNT = 2


def _read_external_rows(source_csv: Path) -> List[Tuple[str, Dict[str, bool]]]:
    """Return ``[(video_filename, {FM: triggered}), ...]`` from the source CSV.

    Tolerates extra columns; required columns are ``Video, fm1, fm2, fm3, fm4``
    (case-insensitive on the header).
    """
    with open(source_csv, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"source CSV is empty: {source_csv}")
        lower = [h.strip().lower() for h in header]
        required = [c.lower() for c in CSV_HEADER]
        try:
            col_idx = {c: lower.index(c) for c in required}
        except ValueError as exc:
            raise ValueError(
                f"source CSV header missing required column: {exc} "
                f"(header was {header})"
            )
        rows: List[Tuple[str, Dict[str, bool]]] = []
        for raw in reader:
            if not raw or not raw[col_idx["video"]]:
                continue
            video = raw[col_idx["video"]]
            flags = {
                fm: raw[col_idx[fm.lower()]].strip() == "1" for fm in FM_ORDER
            }
            rows.append((video, flags))
        return rows


def _backup_csv(dest_csv: Path) -> Path:
    """Copy the dest CSV to ``<name>.bak-<YYYYmmdd-HHMMSS>`` and return the path."""
    if not dest_csv.exists():
        return dest_csv  # nothing to back up
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = dest_csv.with_suffix(dest_csv.suffix + f".bak-{stamp}")
    shutil.copy2(dest_csv, backup)
    return backup


def _write_synthetic_flag_store(
    flag_store_dir: Path,
    video_filename: str,
    flags: Dict[str, bool],
) -> None:
    """Write a minimal valid flag-store JSON for one video.

    See module docstring for the synthesis rules.
    """
    fm_frame_flags = {
        fm: [flags[fm]] * SYNTHETIC_FRAME_COUNT for fm in FM_ORDER
    }
    start_dt = parse_video_timestamp(video_filename)
    save_frame_flags(
        flag_store_dir=flag_store_dir,
        video_filename=video_filename,
        total_frames=SYNTHETIC_FRAME_COUNT,
        fps=DEFAULT_FPS,
        start_time_iso=start_dt.isoformat() if start_dt is not None else "",
        fm_frame_flags=fm_frame_flags,
    )


def merge(
    source_csv: Path,
    dest_csv: Path,
    flag_store_dir: Path,
    overwrite_existing_flag_files: bool,
    dry_run: bool,
) -> None:
    external_rows = _read_external_rows(source_csv)
    already: Set[str] = load_processed_videos(dest_csv)

    net_new = [(v, f) for v, f in external_rows if v not in already]
    skipped = [v for v, _ in external_rows if v in already]

    # Track which flag-store files would be (re)written even when the CSV row
    # is already present, so the user can opt into back-filling missing JSONs.
    flag_only_backfill: List[Tuple[str, Dict[str, bool]]] = []
    if overwrite_existing_flag_files:
        flag_only_backfill = [
            (v, f) for v, f in external_rows
            if v in already and not flag_store_path(flag_store_dir, v).exists()
        ]

    print(f"source CSV:  {source_csv}  ({len(external_rows)} rows)")
    print(f"dest CSV:    {dest_csv}  ({len(already)} rows already present)")
    print(f"flag store:  {flag_store_dir}")
    print(
        f"plan:        append {len(net_new)} new CSV rows, "
        f"write {len(net_new) + len(flag_only_backfill)} synthetic flag files "
        f"(skipping {len(skipped) - len(flag_only_backfill)} dupes)"
    )
    if dry_run:
        print("dry-run: no files written")
        if net_new[:3]:
            print("would-add sample:")
            for v, f in net_new[:3]:
                print(f"  {v}  {f}")
        return

    if not net_new and not flag_only_backfill:
        print("nothing to do")
        return

    backup = _backup_csv(dest_csv)
    if backup != dest_csv:
        print(f"backed up dest CSV to: {backup}")

    ensure_csv_header(dest_csv)
    flag_store_dir.mkdir(parents=True, exist_ok=True)

    for video, flags in net_new:
        append_csv_row(dest_csv, video, flags)
        _write_synthetic_flag_store(flag_store_dir, video, flags)
    for video, flags in flag_only_backfill:
        _write_synthetic_flag_store(flag_store_dir, video, flags)

    print(f"appended {len(net_new)} CSV rows")
    print(
        f"wrote {len(net_new) + len(flag_only_backfill)} synthetic flag files"
    )
    print()
    print("Next: rebuild the timelines from the merged state:")
    print("  python pipelines/run__failure_mode_detection.py --graph-only")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source-csv",
        type=Path,
        required=True,
        help="external fm_failure_modes CSV to merge from",
    )
    p.add_argument(
        "--dest-csv",
        type=Path,
        default=PROJECT_ROOT / "anomaly_classification" / "fm_failure_modes.csv",
        help="this repo's fm_failure_modes.csv (default: anomaly_classification/fm_failure_modes.csv)",
    )
    p.add_argument(
        "--flag-store-dir",
        type=Path,
        default=PROJECT_ROOT / "anomaly_classification" / "fm_frame_flags",
        help="this repo's fm_frame_flags dir (default: anomaly_classification/fm_frame_flags)",
    )
    p.add_argument(
        "--backfill-flag-files",
        action="store_true",
        help=(
            "also write synthetic flag files for rows that already exist in "
            "the dest CSV but have no flag-store JSON (useful when a prior "
            "manual CSV edit left orphans)"
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan and exit without writing anything",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    merge(
        source_csv=args.source_csv,
        dest_csv=args.dest_csv,
        flag_store_dir=args.flag_store_dir,
        overwrite_existing_flag_files=args.backfill_flag_files,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
