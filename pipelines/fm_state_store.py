"""Resumable state for the failure-mode detection pipeline.

Three pieces of durable state, all keyed by video filename:

1. **The CSV** (``fm_failure_modes.csv``) is the source of truth for "which
   videos have been processed". It is appended to one row at a time, so an
   interrupted run loses at most the in-flight video. ``load_processed_videos``
   reads it back to decide what to skip.

2. **The per-frame flag store** (``fm_frame_flags/<video>.json``) holds each
   video's per-frame FM booleans, run-length encoded so the file stays small.
   The timeline graphs are rebuilt from this store, so the graph is resumable
   and never has to hold every video in memory at once.

To reprocess a video, delete its CSV row and its flag-store JSON.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

FM_ORDER: Tuple[str, ...] = ("FM1", "FM2", "FM3", "FM4")
CSV_HEADER: Tuple[str, ...] = ("Video", "fm1", "fm2", "fm3", "fm4")


# =============================================================================
# CSV (source of truth for "processed")
# =============================================================================


def load_processed_videos(csv_path: Path) -> Set[str]:
    """Return the set of video filenames already present in the CSV.

    An absent or empty CSV means nothing has been processed yet.
    """
    if not csv_path.exists():
        return set()

    processed: Set[str] = set()
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return set()
        for row in reader:
            if row and row[0]:
                processed.add(row[0])
    return processed


def ensure_csv_header(csv_path: Path) -> None:
    """Create the CSV with its header row if it does not exist yet."""
    if csv_path.exists():
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(CSV_HEADER)


def append_csv_row(csv_path: Path, video_filename: str, flags: Dict[str, bool]) -> None:
    """Append one ``Video, fm1, fm2, fm3, fm4`` row (0/1) to the CSV.

    Called immediately after each video is processed so progress is durable.
    """
    ensure_csv_header(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [video_filename] + [1 if flags.get(fm, False) else 0 for fm in FM_ORDER]
        )


# =============================================================================
# Per-frame flag store (run-length encoded; drives the timeline graphs)
# =============================================================================


def _rle_encode(flags: List[bool]) -> List[List[int]]:
    """Encode a bool list as ``[[value, run_length], ...]`` (value is 0/1)."""
    runs: List[List[int]] = []
    for flag in flags:
        value = 1 if flag else 0
        if runs and runs[-1][0] == value:
            runs[-1][1] += 1
        else:
            runs.append([value, 1])
    return runs


def _rle_decode(runs: List[List[int]]) -> List[bool]:
    """Inverse of :func:`_rle_encode`."""
    out: List[bool] = []
    for value, length in runs:
        out.extend([bool(value)] * int(length))
    return out


def flag_store_path(flag_store_dir: Path, video_filename: str) -> Path:
    """Path to the per-frame flag JSON for one video."""
    return flag_store_dir / f"{Path(video_filename).stem}.json"


@dataclass(frozen=True)
class VideoFlags:
    """One persisted video's timeline data, loaded from the flag store."""

    video_filename: str
    total_frames: int
    fps: float
    # Video start time as an ISO string (parsed from the filename), or "" if
    # the filename could not be parsed.
    start_time_iso: str
    fm_flags: Dict[str, List[bool]]


def save_frame_flags(
    flag_store_dir: Path,
    video_filename: str,
    total_frames: int,
    fps: float,
    start_time_iso: str,
    fm_frame_flags: Dict[str, List[bool]],
) -> None:
    """Persist one video's per-frame FM flags (RLE) plus timing metadata.

    ``fps`` and ``start_time_iso`` are stored so the timeline can map frame
    index -> real wall-clock time during a ``--graph-only`` rebuild without
    re-opening any video.
    """
    flag_store_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "video_filename": video_filename,
        "total_frames": total_frames,
        "fps": fps,
        "start_time_iso": start_time_iso,
        "fm_flags_rle": {
            fm: _rle_encode(fm_frame_flags.get(fm, [False] * total_frames))
            for fm in FM_ORDER
        },
    }
    path = flag_store_path(flag_store_dir, video_filename)
    with open(path, "w") as f:
        json.dump(payload, f)


def load_all_frame_flags(
    flag_store_dir: Path, ordered_video_filenames: List[str]
) -> List[VideoFlags]:
    """Load every persisted video's flags, in the given video order.

    Videos with no flag-store file (e.g. errored before flags were saved) are
    skipped so the timeline only shows fully-processed videos.
    """
    out: List[VideoFlags] = []
    for video_filename in ordered_video_filenames:
        path = flag_store_path(flag_store_dir, video_filename)
        if not path.exists():
            continue
        with open(path, "r") as f:
            payload = json.load(f)
        fm_flags = {
            fm: _rle_decode(payload["fm_flags_rle"].get(fm, []))
            for fm in FM_ORDER
        }
        out.append(
            VideoFlags(
                video_filename=payload["video_filename"],
                total_frames=int(payload["total_frames"]),
                # Older flag-store files (pre-time-axis) lack these keys; fall
                # back so a mixed store still rebuilds.
                fps=float(payload.get("fps", 0.0)) or 0.0,
                start_time_iso=str(payload.get("start_time_iso", "")),
                fm_flags=fm_flags,
            )
        )
    return out
