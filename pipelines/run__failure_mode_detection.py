"""Multi-failure-mode detection across a camera's videos (S3 or local folder).

Iterates every ``.ts`` video the configured source exposes, processing each
one and persisting per-video results immediately. The run is **resumable**:
the output CSV is the source of truth for which videos are done, so a killed
run picks up exactly where it stopped.

The video source is pluggable via ``--source``:

- ``--source s3`` (default) streams from the S3 prefix configured by the
  ``S3_VIDEO_BUCKET`` / ``S3_VIDEO_PREFIX`` env vars, downloading each file
  to a temp dir and deleting it after processing.
- ``--source <folder>`` reads ``.ts`` files directly from a local folder; no
  files are downloaded, copied, or deleted.

The pipeline body is identical in both modes; only the source plugin differs.

Two detectors per video:

1. **White-pixel metric** (Canny+dilate on the right half of the ROI). If the
   white-pixel percentage drops below ``WHITE_PCT_THRESHOLD`` on ANY frame, the
   video is flagged ``FM3`` (Infeed Collapse).

2. **YOLO object detection** (china/yolo_l ``best.pt``; classes
   ``displaced_box``, ``empty_cups``). Per frame, for detections with
   confidence >= ``YOLO_CONF_THRESHOLD``:
     - ``displaced_box`` whose centre is in the LEFT ``FM2_LEFT_FRACTION`` of
       the frame AND whose box area is >= ``FM2_LARGE_AREA_FRACTION`` of the
       frame area -> ``FM2`` (Fallen Out Infeed).
     - any other ``displaced_box`` -> ``FM1`` (Suction Release).
     - ``empty_cups`` (anywhere) -> ``FM4`` (Empty Cups).

A failure mode is set to 1 for the whole video if ANY frame triggers it.

Resumability — all three outputs survive an interrupted run:
- **CSV** (``fm_failure_modes.csv``) is appended one row per video; on restart
  already-listed videos are skipped.
- **Debug images** (``fm_failure_modes_debug/``) are written per video and
  never wiped, so prior runs' images are kept.
- **Timelines** are rebuilt every run from the per-frame flag store
  (``fm_frame_flags/<video>.json``, run-length encoded), so the graph reflects
  every video ever processed without holding them all in memory.

Outputs (under ``anomaly_classification/``):
- ``fm_failure_modes.csv`` - one row per video: ``Video, fm1, fm2, fm3, fm4``.
- ``fm_failure_modes_timeline.png`` - static matplotlib timeline.
- ``fm_failure_modes_timeline.html`` - interactive plotly timeline (zoom/pan
  the long x-axis).
- ``fm_frame_flags/`` - per-video RLE per-frame flag JSON (drives the graphs).
- ``fm_failure_modes_debug/`` - one annotated debug image per (video,
  triggered FM), showing the strongest-evidence frame.
- ``fm_no_detection_frames/`` - one mid-video frame per video where NO failure
  mode triggered, so clean videos can still be eyeballed.

Usage:
    python pipelines/run__failure_mode_detection.py
        [--source s3 | <folder-path>] [--limit N] [--graph-only]

    --source SPEC  's3' (default) or a path to a folder of .ts files
    --limit N      process at most N unprocessed videos this run (default: all)
    --graph-only   skip processing; just rebuild the timelines from the store
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
from dotenv import load_dotenv
from ultralytics import YOLO

# Load .env from the project root before any module-level env reads.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.fm_state_store import (
    append_csv_row,
    load_all_frame_flags,
    load_processed_videos,
    save_frame_flags,
)
from pipelines.fm_video_source import VideoRef, VideoSource, make_video_source
from pipelines.run__infeed_consistency_pipeline import (
    CONFIG_PATH,
    RoiConfig,
    canny_pipeline,
    crop_to_roi,
    load_config,
)
from shared_functions.timestamp_utils import parse_video_timestamp


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CAMERA_ID = os.environ.get("CAMERA_ID", "40705473")


def _resolve_yolo_weights() -> Path:
    """Return the path to the trained YOLO weights, validating it exists.

    The weights file (``best.pt``) is distributed separately from this repo —
    set the ``YOLO_WEIGHTS_PATH`` environment variable (or add it to a
    ``.env`` file) to point at the file before running detection.
    """
    env = os.environ.get("YOLO_WEIGHTS_PATH")
    if not env:
        raise RuntimeError(
            "YOLO_WEIGHTS_PATH is not set. Point it at the trained best.pt "
            "file. See README.md → 'YOLO weights' for setup details."
        )
    path = Path(env).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"YOLO weights not found at {path}. "
            f"Check YOLO_WEIGHTS_PATH or download the weights file."
        )
    return path

OUTPUT_DIR = PROJECT_ROOT / "anomaly_classification"
OUTPUT_CSV = OUTPUT_DIR / "fm_failure_modes.csv"
OUTPUT_TIMELINE_PNG = OUTPUT_DIR / "fm_failure_modes_timeline.png"
OUTPUT_TIMELINE_HTML = OUTPUT_DIR / "fm_failure_modes_timeline.html"
FLAG_STORE_DIR = OUTPUT_DIR / "fm_frame_flags"
DEBUG_DIR = OUTPUT_DIR / "fm_failure_modes_debug"
# Videos with NO failure mode triggered get a single mid-video frame exported
# here instead, so "clean" videos can still be eyeballed.
NO_DETECTION_DIR = OUTPUT_DIR / "fm_no_detection_frames"
# Scratch dir used by remote video sources (e.g. S3) for downloads; deleted
# per video after processing. Local sources ignore this — see
# :class:`pipelines.fm_local_source.LocalVideoSource`.
TEMP_VIDEO_DIR = OUTPUT_DIR / "fm_source_temp"

# White-pixel % strictly below this on any frame triggers FM3.
WHITE_PCT_THRESHOLD = 50.0
# YOLO detections below this confidence are ignored.
YOLO_CONF_THRESHOLD = 0.7
# A YOLO failure mode (FM1/FM2/FM4) must be detected on at least this many
# CONSECUTIVE frames to be flagged. FM3 is not subject to this rule.
MIN_CONSECUTIVE_FRAMES = 2

# FM2 (Fallen Out Infeed) rule: a displaced_box counts as FM2 only if its
# centre is within the left FM2_LEFT_FRACTION of the frame width AND its box
# area is at least FM2_LARGE_AREA_FRACTION of the full frame area. Every other
# displaced_box falls through to FM1.
FM2_LEFT_FRACTION = 0.25
FM2_LARGE_AREA_FRACTION = 0.02

# YOLO class names (from china/yolo_l/yolo_dataset/dataset.yaml).
CLASS_DISPLACED_BOX = "displaced_box"
CLASS_EMPTY_CUPS = "empty_cups"

# Failure mode metadata: key -> (human label, plot colour).
FM_META = {
    "FM1": ("FM1 Suction Release", "#1f77b4"),
    "FM2": ("FM2 Fallen Out Infeed", "#ff7f0e"),
    "FM3": ("FM3 Infeed Collapse", "#d62728"),
    "FM4": ("FM4 Empty Cups", "#2ca02c"),
}
FM_ORDER = ("FM1", "FM2", "FM3", "FM4")


# =============================================================================
# Per-FM evidence tracking
# =============================================================================


@dataclass
class FmEvidence:
    """Best (strongest) evidence frame for one failure mode.

    For FM3 the rule is "any single frame below threshold" and ``consider`` is
    called greedily during the frame loop. For FM1/FM2/FM4 the trigger requires
    a run of >= MIN_CONSECUTIVE_FRAMES detections; ``consider`` records every
    hit and ``resolve_run_rule`` is applied after the loop.
    """
    triggered: bool = False
    best_score: float = -1.0  # higher == stronger evidence
    best_frame_idx: Optional[int] = None
    best_frame_bgr: Optional[np.ndarray] = None
    # Optional bounding box (x1, y1, x2, y2) for YOLO-driven FMs.
    best_box: Optional[Tuple[int, int, int, int]] = None
    # The consecutive run that triggered the FM (YOLO FMs only).
    run_start_frame: Optional[int] = None
    run_length: int = 0

    def consider(
        self,
        score: float,
        frame_idx: int,
        frame_bgr: np.ndarray,
        box: Optional[Tuple[int, int, int, int]] = None,
    ) -> None:
        """Greedy update: mark triggered and keep the single strongest frame.

        Used directly for FM3. For YOLO FMs this is invoked by
        ``resolve_run_rule`` only on frames inside a qualifying run.
        """
        self.triggered = True
        if score > self.best_score:
            self.best_score = score
            self.best_frame_idx = frame_idx
            self.best_frame_bgr = frame_bgr.copy()
            self.best_box = box


@dataclass(frozen=True)
class FrameHit:
    """One per-frame YOLO detection for a failure mode."""
    frame_idx: int
    conf: float
    frame_bgr: np.ndarray
    box: Tuple[int, int, int, int]


def resolve_run_rule(
    hits: List[FrameHit], min_consecutive: int
) -> FmEvidence:
    """Flag an FM only if it has a run of >= min_consecutive consecutive frames.

    Picks the strongest frame (max confidence) within the longest qualifying
    run; ties on length are broken by the run's peak confidence.
    """
    ev = FmEvidence()
    if not hits:
        return ev

    hits_sorted = sorted(hits, key=lambda h: h.frame_idx)

    # Group into consecutive runs (frame_idx contiguous).
    runs: List[List[FrameHit]] = []
    current: List[FrameHit] = [hits_sorted[0]]
    for h in hits_sorted[1:]:
        if h.frame_idx == current[-1].frame_idx + 1:
            current.append(h)
        else:
            runs.append(current)
            current = [h]
    runs.append(current)

    qualifying = [r for r in runs if len(r) >= min_consecutive]
    if not qualifying:
        return ev

    def run_key(run: List[FrameHit]) -> Tuple[int, float]:
        return (len(run), max(h.conf for h in run))

    best_run = max(qualifying, key=run_key)
    peak = max(best_run, key=lambda h: h.conf)
    ev.consider(peak.conf, peak.frame_idx, peak.frame_bgr, peak.box)
    ev.run_start_frame = best_run[0].frame_idx
    ev.run_length = len(best_run)
    return ev


DEFAULT_FPS = 155.0  # camera 40705473 nominal frame rate; fallback only.


@dataclass
class VideoResult:
    video_filename: str
    total_frames: int
    # Frame rate reported by the video container; used to map frame index ->
    # real time for the timeline. Falls back to DEFAULT_FPS if unreadable.
    fps: float = DEFAULT_FPS
    evidence: Dict[str, FmEvidence] = field(default_factory=dict)
    # Per-frame white pct, for the timeline plot.
    white_pct_series: List[float] = field(default_factory=list)
    # Per-frame boolean for each FM, for the timeline plot.
    fm_frame_flags: Dict[str, List[bool]] = field(default_factory=dict)
    # The video's middle frame, kept so a no-detection video can still be
    # eyeballed (exported only when nothing triggered).
    mid_frame_bgr: Optional[np.ndarray] = None
    error: str = ""

    @property
    def any_triggered(self) -> bool:
        return any(ev.triggered for ev in self.evidence.values())


# =============================================================================
# Per-video processing
# =============================================================================


YOLO_FM_KEYS = ("FM1", "FM2", "FM4")


def _resolve_yolo_device() -> Tuple[str, bool]:
    """Pick the best available inference device for YOLO.

    Returns ``(device, use_half)`` where ``device`` is an ultralytics-compatible
    spec ("cuda:0" or "cpu") and ``use_half`` enables FP16 inference (GPU only).
    Falls back to CPU silently when CUDA is unavailable; ultralytics would do
    the same, but resolving it here lets us log the decision and pass an
    explicit ``device`` into ``predict`` to skip its per-call auto-detect.
    """
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.current_device()}", True
    return "cpu", False


def process_video(
    video_path: Path,
    roi: RoiConfig,
    model: YOLO,
    device: str,
    use_half: bool,
) -> VideoResult:
    evidence = {fm: FmEvidence() for fm in FM_ORDER}
    fm_frame_flags: Dict[str, List[bool]] = {fm: [] for fm in FM_ORDER}
    white_series: List[float] = []
    # Per-frame YOLO detections, resolved into runs after the loop.
    yolo_hits: Dict[str, List[FrameHit]] = {fm: [] for fm in YOLO_FM_KEYS}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return VideoResult(
            video_filename=video_path.name,
            total_frames=0,
            evidence=evidence,
            error="cv2_open_failed",
        )

    # Frame rate for the time axis. A container that reports 0 or NaN falls
    # back to the camera's nominal rate rather than producing a broken axis.
    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = raw_fps if raw_fps and raw_fps > 1.0 else DEFAULT_FPS

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_h, frame_w = frame.shape[:2]
            frame_area = float(frame_w * frame_h)
            fm2_left_x = frame_w * FM2_LEFT_FRACTION

            # --- FM3: white-pixel metric (any single frame triggers) ---
            try:
                _, _, _, _, white_pct = canny_pipeline(frame, roi)
            except ValueError:
                white_pct = 100.0  # ROI invalid for this frame; treat as "no collapse"
            white_series.append(white_pct)

            fm3_hit = white_pct < WHITE_PCT_THRESHOLD
            fm_frame_flags["FM3"].append(fm3_hit)
            if fm3_hit:
                # Lower white pct == stronger evidence -> score = (threshold - white).
                evidence["FM3"].consider(
                    score=WHITE_PCT_THRESHOLD - white_pct,
                    frame_idx=frame_idx,
                    frame_bgr=frame,
                )

            # --- FM1 / FM2 / FM4: YOLO detections ---
            # Keep at most one hit per FM per frame (the highest-confidence one)
            # so consecutive-run detection is frame-contiguous.
            best_per_fm: Dict[str, FrameHit] = {}
            results = model.predict(
                frame,
                conf=YOLO_CONF_THRESHOLD,
                device=device,
                half=use_half,
                verbose=False,
            )
            for res in results:
                boxes = res.boxes
                if boxes is None:
                    continue
                names = res.names
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i].item())
                    conf = float(boxes.conf[i].item())
                    cls_name = names.get(cls_id, str(cls_id))
                    x1, y1, x2, y2 = (
                        float(v) for v in boxes.xyxy[i].tolist()
                    )
                    box_int = (int(x1), int(y1), int(x2), int(y2))
                    center_x = (x1 + x2) / 2.0
                    box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)

                    if cls_name == CLASS_DISPLACED_BOX:
                        # FM2 only when the box is in the left FM2_LEFT_FRACTION
                        # of the frame AND large enough; otherwise FM1.
                        is_left = center_x < fm2_left_x
                        is_large = (
                            frame_area > 0
                            and box_area / frame_area >= FM2_LARGE_AREA_FRACTION
                        )
                        fm_key = "FM2" if (is_left and is_large) else "FM1"
                    elif cls_name == CLASS_EMPTY_CUPS:
                        fm_key = "FM4"
                    else:
                        continue

                    prev = best_per_fm.get(fm_key)
                    if prev is None or conf > prev.conf:
                        best_per_fm[fm_key] = FrameHit(
                            frame_idx=frame_idx,
                            conf=conf,
                            frame_bgr=frame.copy(),
                            box=box_int,
                        )

            for fm_key, hit in best_per_fm.items():
                yolo_hits[fm_key].append(hit)
            for fm_key in YOLO_FM_KEYS:
                fm_frame_flags[fm_key].append(fm_key in best_per_fm)

            frame_idx += 1
    finally:
        cap.release()

    # Apply the consecutive-run rule to the YOLO failure modes.
    for fm_key in YOLO_FM_KEYS:
        evidence[fm_key] = resolve_run_rule(
            yolo_hits[fm_key], MIN_CONSECUTIVE_FRAMES
        )

    # Grab the middle frame so a video with no detections can still be
    # eyeballed. Re-seek with a fresh capture now that the total is known;
    # this costs one decode and avoids buffering every frame during the loop.
    mid_frame_bgr = _read_middle_frame(video_path, frame_idx)

    return VideoResult(
        video_filename=video_path.name,
        total_frames=frame_idx,
        fps=fps,
        evidence=evidence,
        white_pct_series=white_series,
        fm_frame_flags=fm_frame_flags,
        mid_frame_bgr=mid_frame_bgr,
    )


def _read_middle_frame(
    video_path: Path, total_frames: int
) -> Optional[np.ndarray]:
    """Return the frame at ``total_frames // 2``, or None if it can't be read."""
    if total_frames <= 0:
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
        ret, frame = cap.read()
        return frame if ret and frame is not None else None
    finally:
        cap.release()


# =============================================================================
# Debug image
# =============================================================================


def build_fm_debug_image(
    video_result: VideoResult,
    fm_key: str,
    ev: FmEvidence,
    roi: RoiConfig,
) -> Optional[np.ndarray]:
    """2-panel debug for one triggered FM: full frame (annotated) | ROI / Canny."""
    if ev.best_frame_bgr is None or ev.best_frame_idx is None:
        return None

    frame = ev.best_frame_bgr
    label, _ = FM_META[fm_key]

    full = frame.copy()
    # Draw the ROI rectangle and the FM2 left-fraction boundary on the full
    # frame (a displaced_box left of this line + large enough -> FM2).
    rx1, ry1, rx2, ry2 = roi.as_xyxy()
    cv2.rectangle(full, (rx1, ry1), (rx2, ry2), (0, 200, 255), 2)
    fm2_x = int(full.shape[1] * FM2_LEFT_FRACTION)
    cv2.line(full, (fm2_x, 0), (fm2_x, full.shape[0]), (180, 180, 180), 1)
    if ev.best_box is not None:
        bx1, by1, bx2, by2 = ev.best_box
        cv2.rectangle(full, (bx1, by1), (bx2, by2), (0, 0, 255), 3)
        cv2.putText(
            full,
            f"{ev.best_score:.2f}",
            (bx1, max(by1 - 8, 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    # Right panel: for FM3 show the Canny+dilate mask; otherwise the ROI crop.
    if fm_key == "FM3":
        _, _, _, dilated, white_pct = canny_pipeline(frame, roi)
        right = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
        right_title = f"CANNY+DILATE  white={white_pct:.2f}%"
    else:
        right = crop_to_roi(frame, roi)
        right_title = "ROI"

    panel_h = max(full.shape[0], right.shape[0])

    def fit(img: np.ndarray) -> np.ndarray:
        if img.shape[0] == panel_h:
            return img
        scale = panel_h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * scale), panel_h))

    label_strip_h = 32
    border = 4

    def frame_panel(img: np.ndarray, title: str, color: Tuple[int, int, int]) -> np.ndarray:
        fitted = fit(img)
        out = np.zeros(
            (fitted.shape[0] + label_strip_h, fitted.shape[1], 3), dtype=np.uint8
        )
        cv2.putText(
            out, title, (10, label_strip_h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )
        out[label_strip_h:, :] = fitted
        cv2.rectangle(
            out, (0, 0), (out.shape[1] - 1, out.shape[0] - 1), color, border
        )
        return out

    left_panel = frame_panel(
        full, "FULL FRAME (ROI + FM2 boundary)", (200, 200, 200)
    )
    right_panel = frame_panel(right, right_title, (0, 200, 255))

    gap = np.zeros((left_panel.shape[0], 24, 3), dtype=np.uint8)
    row = np.hstack([left_panel, gap, right_panel])

    banner_h = 56
    banner = np.zeros((banner_h, row.shape[1], 3), dtype=np.uint8)
    text = (
        f"{video_result.video_filename}  {label}  "
        f"frame={ev.best_frame_idx}  score={ev.best_score:.3f}"
    )
    if ev.run_length:
        text += f"  run=[{ev.run_start_frame}, +{ev.run_length}]"
    cv2.putText(
        banner, text, (12, 36),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
    )

    pad = 16
    inner = np.vstack([banner, row])
    canvas = np.zeros(
        (inner.shape[0] + pad * 2, inner.shape[1] + pad * 2, 3), dtype=np.uint8
    )
    canvas[pad:pad + inner.shape[0], pad:pad + inner.shape[1]] = inner
    return canvas


# =============================================================================
# Timeline (rebuilt from the per-frame flag store each run)
# =============================================================================


@dataclass(frozen=True)
class VideoMarker:
    """One video's start time, for the timeline's video separators."""
    video_filename: str
    start: datetime


# Long gaps between recordings are collapsed on the timeline so the chart
# stays legible when datasets cover multiple disconnected days. Anything wider
# than ``GAP_BREAK_MULTIPLIER`` x the typical "long" inter-video gap is
# replaced with a narrow "..." break.
#
# The "typical long gap" is the ``GAP_BREAK_REFERENCE_QUANTILE`` of the
# observed inter-video gap distribution (75th percentile by default). Using a
# high quantile prevents bursts of near-coincident timestamps -- e.g. the
# ``__RATE_LIMITED__`` files arriving within seconds of each other -- from
# collapsing the typical-gap estimate down to a few seconds, which would then
# scale the threshold so low that ordinary within-shift idle stretches also
# get collapsed.
#
# The break's visual width is the larger of ``GAP_BREAK_DISPLAY_SECONDS_MIN``
# and a small fraction of the total kept span (``GAP_BREAK_DISPLAY_RATIO``) so
# the boundary tick labels on either side of the break never collide.
GAP_BREAK_MULTIPLIER = 5.0
GAP_BREAK_REFERENCE_QUANTILE = 0.75
GAP_BREAK_DISPLAY_SECONDS_MIN = 60.0
GAP_BREAK_DISPLAY_RATIO = 0.06
# Absolute floor: never collapse anything shorter than this, even if the
# quantile-derived threshold would. Two hours is shorter than any plausible
# multi-shift idle period but longer than any sane within-shift gap.
GAP_BREAK_MIN_REAL_SECONDS = 2 * 3600.0


@dataclass(frozen=True)
class AxisBreak:
    """One collapsed gap on the display axis: real range -> display range."""
    real_start: datetime
    real_end: datetime
    display_start: float  # seconds from t0
    display_end: float


@dataclass(frozen=True)
class TimeAxis:
    """Piecewise-linear mapping from real wall-clock time to display seconds.

    ``segments`` is a list of ``(real_start, real_end, display_start)`` triples
    sorted chronologically. Inside a segment, mapping is the identity (one real
    second = one display second). Between segments, a collapsed gap of width
    ``GAP_BREAK_DISPLAY_SECONDS`` is inserted -- those gaps are recorded in
    ``breaks`` so the renderer can draw "..." markers.

    The axis is built so any input ``dt`` inside one of the kept segments maps
    deterministically. ``dt`` values inside a collapsed gap are clamped to the
    nearest segment edge -- we do not put real activity inside gaps, so this
    only ever happens for axis bookkeeping (segment boundaries themselves).
    """
    segments: List[Tuple[datetime, datetime, float]]
    breaks: List[AxisBreak]
    display_min: float
    display_max: float

    def to_display(self, dt: datetime) -> float:
        # Linear search is fine here: a multi-day archive collapses to a
        # handful of segments, not thousands.
        for real_start, real_end, display_start in self.segments:
            if real_start <= dt <= real_end:
                return display_start + (dt - real_start).total_seconds()
        # Outside any segment (only possible for the global min/max in
        # degenerate cases): clamp to the nearest segment edge.
        first_start, _, first_disp = self.segments[0]
        if dt < first_start:
            return first_disp
        _, last_end, last_disp = self.segments[-1]
        return last_disp + (last_end - self.segments[-1][1]).total_seconds()


def build_time_axis(
    markers: List[VideoMarker],
    axis_min: datetime,
    axis_max: datetime,
) -> TimeAxis:
    """Build a piecewise-linear time axis that collapses unusually large gaps.

    The threshold is data-driven: take the median inter-video gap and collapse
    anything ``GAP_BREAK_MULTIPLIER`` times larger. With one cluster of videos
    this returns a single identity segment; with two clusters separated by an
    overnight or multi-day idle period the gap becomes a thin break.
    """
    if not markers:
        return TimeAxis(
            segments=[(axis_min, axis_max, 0.0)],
            breaks=[],
            display_min=0.0,
            display_max=(axis_max - axis_min).total_seconds(),
        )

    sorted_markers = sorted(markers, key=lambda m: m.start)
    gaps = [
        (sorted_markers[i + 1].start - sorted_markers[i].start).total_seconds()
        for i in range(len(sorted_markers) - 1)
    ]
    positive_gaps = sorted(g for g in gaps if g > 0)
    if positive_gaps:
        # Pick the high-quantile gap so bursts of near-coincident timestamps
        # don't pull the reference down and over-collapse normal idle gaps.
        idx = min(
            len(positive_gaps) - 1,
            int(GAP_BREAK_REFERENCE_QUANTILE * (len(positive_gaps) - 1)),
        )
        reference_gap = positive_gaps[idx]
    else:
        reference_gap = 0.0
    threshold = max(
        reference_gap * GAP_BREAK_MULTIPLIER, GAP_BREAK_MIN_REAL_SECONDS
    )

    # Merge zero-width marker intervals plus the axis end-caps. Without
    # spans, "activity" is just video starts -- collapse anything between
    # them that exceeds the threshold.
    points = sorted(
        {axis_min, axis_max, *(m.start for m in sorted_markers)}
    )
    merged: List[List[datetime]] = [[points[0], points[0]]]
    for p in points[1:]:
        merged.append([p, p])

    # First pass: merge intervals into "kept segments" by collapsing every gap
    # that exceeds ``threshold``. We track segment spans (real time) and the
    # real-time pairs of each collapsed gap, but defer assigning display
    # coordinates until we know the total kept span -- the break's display
    # width depends on it (see ``break_display_width`` below) and using a
    # fixed width can cause the boundary tick labels on either side of a
    # break to overlap on screen.
    kept_segments: List[List[datetime]] = [list(merged[0])]
    collapsed_gaps: List[Tuple[datetime, datetime]] = []

    for nxt_start, nxt_end in merged[1:]:
        cur_end = kept_segments[-1][1]
        if (nxt_start - cur_end).total_seconds() > threshold:
            collapsed_gaps.append((cur_end, nxt_start))
            kept_segments.append([nxt_start, nxt_end])
        else:
            if nxt_end > cur_end:
                kept_segments[-1][1] = nxt_end

    kept_span = sum(
        (s[1] - s[0]).total_seconds() for s in kept_segments
    )

    # Break width is the larger of a fixed minimum and a small fraction of the
    # total kept span. The fraction guarantees that the boundary tick labels
    # on either side of the break stay separated enough not to overlap.
    break_display_width = max(
        GAP_BREAK_DISPLAY_SECONDS_MIN, kept_span * GAP_BREAK_DISPLAY_RATIO
    )

    # Second pass: lay out segments and breaks on the display axis.
    segments: List[Tuple[datetime, datetime, float]] = []
    breaks: List[AxisBreak] = []
    cur_display = 0.0
    for i, (seg_start, seg_end) in enumerate(kept_segments):
        seg_real = (seg_end - seg_start).total_seconds()
        segments.append((seg_start, seg_end, cur_display))
        cur_display += seg_real
        if i < len(collapsed_gaps):
            gap_start, gap_end = collapsed_gaps[i]
            breaks.append(
                AxisBreak(
                    real_start=gap_start,
                    real_end=gap_end,
                    display_start=cur_display,
                    display_end=cur_display + break_display_width,
                )
            )
            cur_display += break_display_width

    return TimeAxis(
        segments=segments,
        breaks=breaks,
        display_min=0.0,
        display_max=cur_display,
    )


def axis_tick_positions(
    axis: TimeAxis, target_ticks: int = 8
) -> Tuple[List[float], List[str]]:
    """Pick ~target_ticks tick positions in display time, labelled as datetimes.

    Each kept segment gets a share of ticks proportional to its display length.
    Segment start/end ticks are tagged as "boundary" ticks so the dedupe pass
    below can preserve them when they collide with interior ticks across a
    collapsed gap.
    """
    seg_lengths = [
        (s[1] - s[0]).total_seconds() for s in axis.segments
    ]
    total_len = sum(seg_lengths) or 1.0

    # (display_pos, label, is_boundary). Boundary ticks anchor the visible
    # edges of each segment so they never get dropped in favour of an
    # interior tick that just happens to be nearby.
    candidates: List[Tuple[float, str, bool]] = []

    for (real_start, real_end, display_start), seg_len in zip(
        axis.segments, seg_lengths
    ):
        share = max(2, int(round(target_ticks * seg_len / total_len)))
        for i in range(share + 1):
            t = i / share
            dt = real_start + (real_end - real_start) * t
            disp = display_start + (dt - real_start).total_seconds()
            is_boundary = i == 0 or i == share
            candidates.append((disp, dt.strftime("%Y-%m-%d\n%H:%M"), is_boundary))

    # Two visual collision cases to dedupe:
    #   1. Exact duplicate at a segment join (within 1s).
    #   2. Tick at the END of segment N and tick at the START of segment N+1.
    #      They are separated only by the narrow break band, so labels overlap
    #      across the "..." marker. Enforce a minimum on-screen separation of
    #      ~3% of the total display axis.
    total_span = max(axis.display_max - axis.display_min, 1.0)
    min_sep = max(1.0, total_span * 0.03)

    candidates.sort(key=lambda c: c[0])
    deduped: List[Tuple[float, str, bool]] = []
    for cand in candidates:
        if deduped and (cand[0] - deduped[-1][0]) < min_sep:
            # Within the readable minimum. Prefer keeping a boundary tick if
            # either is one; if both are boundaries (segment-end vs next
            # segment-start across a break), drop the trailing interior tick
            # by replacing the kept one with the boundary that has more
            # downstream importance -- arbitrary tie-break: keep both.
            prev = deduped[-1]
            if cand[2] and not prev[2]:
                deduped[-1] = cand  # replace interior with boundary
            elif cand[2] and prev[2]:
                # Both are boundaries across a collapsed gap. Keep both --
                # they're the only labels telling the reader the gap's edges.
                deduped.append(cand)
            # else: prev is boundary or both interior -> drop cand
            continue
        deduped.append(cand)

    return [c[0] for c in deduped], [c[1] for c in deduped]


def apply_min_run(flags: List[bool], min_consecutive: int) -> List[bool]:
    """Zero out runs of True shorter than ``min_consecutive``."""
    out = list(flags)
    x = 0
    n = len(out)
    while x < n:
        if out[x]:
            run_start = x
            while x < n and out[x]:
                x += 1
            if x - run_start < min_consecutive:
                for j in range(run_start, x):
                    out[j] = False
        else:
            x += 1
    return out


@dataclass(frozen=True)
class VideoFmStatus:
    """One video's per-FM triggered flags + the time anchor used to plot it.

    Built from the per-frame flag store; the consecutive-run rule for the YOLO
    failure modes is applied here so the timeline matches the CSV's triggered
    column for column.
    """

    video_filename: str
    start: datetime
    triggered: Dict[str, bool]


def _build_timeline_data(
    flag_store_dir: Path, ordered_video_filenames: List[str]
) -> Tuple[List[VideoFmStatus], List[VideoMarker], Optional[datetime], Optional[datetime]]:
    """Build per-video triggered/clean status from the per-frame flag store.

    Each video becomes one :class:`VideoFmStatus` with a boolean per FM and a
    real wall-clock anchor (the recording start time parsed from the filename).
    FM1/FM2/FM4 use the ``MIN_CONSECUTIVE_FRAMES`` run rule; FM3 triggers on
    any single frame. Videos whose filename timestamp could not be parsed are
    skipped — they cannot be placed on the time axis.
    """
    per_video = load_all_frame_flags(flag_store_dir, ordered_video_filenames)

    statuses: List[VideoFmStatus] = []
    markers: List[VideoMarker] = []
    axis_min: Optional[datetime] = None
    axis_max: Optional[datetime] = None

    for vf in per_video:
        if vf.total_frames == 0:
            continue
        start_dt = parse_video_timestamp(vf.video_filename)
        if start_dt is None:
            continue
        fps = vf.fps if vf.fps and vf.fps > 1.0 else DEFAULT_FPS
        video_end = start_dt + timedelta(seconds=vf.total_frames / fps)

        triggered: Dict[str, bool] = {}
        for fm in FM_ORDER:
            flags = vf.fm_flags.get(fm, [False] * vf.total_frames)
            if fm in YOLO_FM_KEYS:
                flags = apply_min_run(flags, MIN_CONSECUTIVE_FRAMES)
            triggered[fm] = any(flags)

        statuses.append(VideoFmStatus(vf.video_filename, start_dt, triggered))
        markers.append(VideoMarker(vf.video_filename, start_dt))
        axis_min = start_dt if axis_min is None else min(axis_min, start_dt)
        axis_max = video_end if axis_max is None else max(axis_max, video_end)

    statuses.sort(key=lambda s: s.start)
    markers.sort(key=lambda m: m.start)
    return statuses, markers, axis_min, axis_max


def plot_timeline_png(
    statuses: List[VideoFmStatus],
    markers: List[VideoMarker],
    axis: TimeAxis,
    output_path: Path,
) -> None:
    """Static matplotlib timeline as a per-video, per-FM dot grid.

    Each FM is a row; each video is one column anchored at its real start
    time on the gap-aware display axis. Filled marker (FM colour) = the FM
    triggered for that video; hollow grey ring = clean.
    """
    fig, ax = plt.subplots(figsize=(24, 5))

    lane_of = {fm: lane for lane, fm in enumerate(FM_ORDER)}

    # Group statuses by lane / (triggered, clean) so we can issue 8 scatter
    # calls instead of one per dot.
    xs_hit: Dict[str, List[float]] = {fm: [] for fm in FM_ORDER}
    xs_clean: Dict[str, List[float]] = {fm: [] for fm in FM_ORDER}
    for s in statuses:
        x = axis.to_display(s.start)
        for fm in FM_ORDER:
            (xs_hit if s.triggered[fm] else xs_clean)[fm].append(x)

    for fm in FM_ORDER:
        lane = lane_of[fm]
        _, color = FM_META[fm]
        if xs_clean[fm]:
            ax.scatter(
                xs_clean[fm],
                [lane] * len(xs_clean[fm]),
                s=60,
                facecolors="none",
                edgecolors="#bbbbbb",
                linewidths=1.0,
                zorder=2,
            )
        if xs_hit[fm]:
            ax.scatter(
                xs_hit[fm],
                [lane] * len(xs_hit[fm]),
                s=90,
                facecolors=color,
                edgecolors=color,
                linewidths=0,
                zorder=3,
            )

    # Lane guides.
    for lane, fm in enumerate(FM_ORDER):
        ax.hlines(
            lane,
            axis.display_min,
            axis.display_max,
            color="#eeeeee",
            linewidth=0.8,
            zorder=0,
        )

    # Render collapsed gaps as a hatched band with a "..." label.
    for br in axis.breaks:
        ax.axvspan(
            br.display_start,
            br.display_end,
            facecolor="#f4f4f4",
            edgecolor="#bbbbbb",
            hatch="///",
            linewidth=0.3,
            zorder=-1,
        )
        ax.text(
            (br.display_start + br.display_end) / 2,
            len(FM_ORDER) - 0.4,
            "...",
            ha="center",
            va="bottom",
            fontsize=14,
            color="#666666",
        )
        gap_label = _format_gap_duration(
            (br.real_end - br.real_start).total_seconds()
        )
        ax.text(
            (br.display_start + br.display_end) / 2,
            -0.55,
            f"gap {gap_label}",
            ha="center",
            va="top",
            fontsize=8,
            color="#999999",
        )

    tick_pos, tick_lab = axis_tick_positions(axis)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab, rotation=20, ha="right", fontsize=8)
    ax.set_yticks(range(len(FM_ORDER)))
    ax.set_yticklabels([FM_META[fm][0] for fm in FM_ORDER])
    ax.set_ylim(-0.6, len(FM_ORDER) - 0.2)
    ax.set_xlim(axis.display_min, axis.display_max)
    ax.set_xlabel("Time (long idle gaps collapsed; see '...' markers)")
    n_hit = sum(1 for s in statuses if any(s.triggered.values()))
    ax.set_title(
        f"Per-video failure-mode classifications ({CAMERA_ID}); "
        f"{len(statuses)} videos, {n_hit} with >=1 FM triggered; "
        f"FM3 white%<{WHITE_PCT_THRESHOLD:.0f}, "
        f"YOLO conf>={YOLO_CONF_THRESHOLD}, "
        f"FM1/2/4 need >={MIN_CONSECUTIVE_FRAMES} consecutive frames"
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _format_gap_duration(seconds: float) -> str:
    """Compact human label: '23m', '4h12m', '2d6h'."""
    seconds = max(0, int(round(seconds)))
    if seconds < 3600:
        return f"{seconds // 60}m"
    if seconds < 86400:
        h, rem = divmod(seconds, 3600)
        return f"{h}h{rem // 60:02d}m"
    d, rem = divmod(seconds, 86400)
    return f"{d}d{rem // 3600:02d}h"


def plot_timeline_html(
    statuses: List[VideoFmStatus],
    markers: List[VideoMarker],
    axis: TimeAxis,
    output_path: Path,
) -> None:
    """Interactive plotly timeline as a per-video, per-FM dot grid.

    One filled marker per (video, FM) when triggered; a hollow grey ring when
    clean. Hover shows the video filename, real timestamp, and the full
    triggered set for that video.
    """
    fig = go.Figure()

    lane_of = {fm: lane for lane, fm in enumerate(FM_ORDER)}

    # Precompute per-video hover summary so it's identical on every dot from
    # the same video.
    def summary(s: VideoFmStatus) -> str:
        active = [fm for fm in FM_ORDER if s.triggered[fm]] or ["(none)"]
        return (
            f"{Path(s.video_filename).name}<br>"
            f"{s.start:%Y-%m-%d %H:%M:%S}<br>"
            f"triggered: {', '.join(active)}"
        )

    for fm in FM_ORDER:
        label, color = FM_META[fm]
        lane = lane_of[fm]
        xs_hit: List[float] = []
        hover_hit: List[str] = []
        xs_clean: List[float] = []
        hover_clean: List[str] = []
        for s in statuses:
            x = axis.to_display(s.start)
            if s.triggered[fm]:
                xs_hit.append(x)
                hover_hit.append(f"{label}<br>{summary(s)}")
            else:
                xs_clean.append(x)
                hover_clean.append(f"{label}: clean<br>{summary(s)}")

        if xs_clean:
            fig.add_trace(
                go.Scattergl(
                    x=xs_clean,
                    y=[lane] * len(xs_clean),
                    mode="markers",
                    marker=dict(
                        size=8,
                        color="rgba(0,0,0,0)",
                        line=dict(color="#bbbbbb", width=1),
                    ),
                    name=f"{label} clean",
                    hovertext=hover_clean,
                    hoverinfo="text",
                    showlegend=False,
                )
            )
        if xs_hit:
            fig.add_trace(
                go.Scattergl(
                    x=xs_hit,
                    y=[lane] * len(xs_hit),
                    mode="markers",
                    marker=dict(size=12, color=color, line=dict(width=0)),
                    name=label,
                    hovertext=hover_hit,
                    hoverinfo="text",
                )
            )

    # Collapsed-gap bands + "..." annotations.
    shapes = []
    annotations = []
    for br in axis.breaks:
        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="paper",
                x0=br.display_start,
                x1=br.display_end,
                y0=0,
                y1=1,
                fillcolor="#f0f0f0",
                line=dict(color="#bbbbbb", width=0.5),
                layer="below",
            )
        )
        gap_label = _format_gap_duration(
            (br.real_end - br.real_start).total_seconds()
        )
        annotations.append(
            dict(
                x=(br.display_start + br.display_end) / 2,
                y=1.02,
                xref="x",
                yref="paper",
                text=f"... <span style='font-size:10px;color:#999'>({gap_label})</span>",
                showarrow=False,
                font=dict(size=14, color="#666"),
            )
        )

    tick_pos, tick_lab = axis_tick_positions(axis)
    tick_lab_html = [t.replace("\n", "<br>") for t in tick_lab]

    n_hit = sum(1 for s in statuses if any(s.triggered.values()))
    fig.update_layout(
        title=(
            f"Per-video failure-mode classifications ({CAMERA_ID}); "
            f"{len(statuses)} videos, {n_hit} with >=1 FM triggered; "
            f"FM3 white%<{WHITE_PCT_THRESHOLD:.0f}, "
            f"YOLO conf>={YOLO_CONF_THRESHOLD}, "
            f"FM1/2/4 need >={MIN_CONSECUTIVE_FRAMES} consecutive frames"
        ),
        xaxis=dict(
            title="Time (long idle gaps collapsed; see '...' markers)",
            tickmode="array",
            tickvals=tick_pos,
            ticktext=tick_lab_html,
            range=[axis.display_min, axis.display_max],
            rangeslider=dict(visible=True),
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(FM_ORDER))),
            ticktext=[FM_META[fm][0] for fm in FM_ORDER],
            range=[-0.6, len(FM_ORDER) - 0.4],
        ),
        shapes=shapes,
        annotations=annotations,
        height=460,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3),
        margin=dict(l=160, r=40, t=90, b=80),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))


def rebuild_timelines(ordered_video_filenames: List[str]) -> int:
    """Rebuild both timeline outputs from the flag store. Returns video count."""
    statuses, markers, axis_min, axis_max = _build_timeline_data(
        FLAG_STORE_DIR, ordered_video_filenames
    )
    if not markers or axis_min is None or axis_max is None:
        print("  (no time-placeable videos in the flag store yet — skipping timelines)")
        return 0
    axis = build_time_axis(markers, axis_min, axis_max)
    if axis.breaks:
        gaps = ", ".join(
            f"{br.real_start:%Y-%m-%d %H:%M}->{br.real_end:%H:%M} "
            f"({_format_gap_duration((br.real_end - br.real_start).total_seconds())})"
            for br in axis.breaks
        )
        print(f"  collapsed {len(axis.breaks)} large gap(s): {gaps}")
    plot_timeline_png(statuses, markers, axis, OUTPUT_TIMELINE_PNG)
    plot_timeline_html(statuses, markers, axis, OUTPUT_TIMELINE_HTML)
    return len(markers)


# =============================================================================
# Driver
# =============================================================================


def _fmt_duration(seconds: float) -> str:
    """Human-readable H:MM:SS (or M:SS for short spans)."""
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


class ProgressTracker:
    """Tracks per-video wall time and prints progress + a rolling ETA.

    The ETA uses a rolling average over the most recent ``window`` videos so a
    few slow (long) videos early on don't permanently skew the estimate.
    """

    def __init__(self, total: int, window: int = 25) -> None:
        self.total = total
        self.done = 0
        self.start_time = time.monotonic()
        self._recent: Deque[float] = deque(maxlen=window)

    def record(self, video_seconds: float) -> None:
        self.done += 1
        self._recent.append(video_seconds)

    def summary(self) -> str:
        """One-line progress: count, %, elapsed, rolling rate, ETA, finish-at."""
        elapsed = time.monotonic() - self.start_time
        pct = (self.done / self.total * 100.0) if self.total else 0.0
        parts = [
            f"{self.done}/{self.total} ({pct:.1f}%)",
            f"elapsed {_fmt_duration(elapsed)}",
        ]
        if self._recent:
            avg = sum(self._recent) / len(self._recent)
            remaining = self.total - self.done
            eta_seconds = avg * remaining
            finish_at = datetime.now() + timedelta(seconds=eta_seconds)
            parts.append(f"avg {avg:.1f}s/video")
            parts.append(f"ETA {_fmt_duration(eta_seconds)}")
            parts.append(f"~done {finish_at:%H:%M:%S}")
        return "progress: " + " | ".join(parts)


def write_debug_images(result: VideoResult, roi: RoiConfig) -> None:
    """Write debug output for one video.

    - If ANY failure mode triggered: one annotated debug image per triggered
      FM, into ``DEBUG_DIR``.
    - If NOTHING triggered: a single mid-video frame into ``NO_DETECTION_DIR``
      so "clean" videos can still be eyeballed.

    Neither folder is wiped between runs — each run only adds images for the
    videos it processed, so they accumulate across resumable runs.
    """
    stem = Path(result.video_filename).stem

    if not result.any_triggered:
        if result.mid_frame_bgr is None:
            return
        NO_DETECTION_DIR.mkdir(parents=True, exist_ok=True)
        mid_idx = result.total_frames // 2
        out_path = NO_DETECTION_DIR / f"{stem}__no_detection__f{mid_idx:06d}.png"
        cv2.imwrite(str(out_path), result.mid_frame_bgr)
        return

    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    for fm in FM_ORDER:
        ev = result.evidence[fm]
        if not ev.triggered:
            continue
        img = build_fm_debug_image(result, fm, ev, roi)
        if img is None:
            continue
        out_path = DEBUG_DIR / f"{stem}__{fm}__f{ev.best_frame_idx:06d}.png"
        cv2.imwrite(str(out_path), img)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=str,
        default="s3",
        help=(
            "Video source: 's3' (default S3 bucket/prefix) or a path to a "
            "local folder of .ts files."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="process at most N unprocessed videos this run (default: all)",
    )
    parser.add_argument(
        "--graph-only",
        action="store_true",
        help="skip processing; just rebuild the timelines from the flag store",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 70)
    print(f"Failure-mode detection (camera {CAMERA_ID})")
    print("=" * 70)

    config = load_config(CONFIG_PATH)
    if config.roi is None:
        raise RuntimeError("ROI not configured in infeed_consistency_detection.json")

    # --graph-only: rebuild timelines from whatever is already in the store.
    if args.graph_only:
        processed = sorted(load_processed_videos(OUTPUT_CSV))
        print(f"Graph-only: rebuilding timelines from {len(processed)} processed videos")
        n = rebuild_timelines(processed)
        print(f"Timeline PNG:  {OUTPUT_TIMELINE_PNG}")
        print(f"Timeline HTML: {OUTPUT_TIMELINE_HTML}  ({n} videos)")
        return

    source: VideoSource = make_video_source(args.source)

    print(
        f"ROI:           ({config.roi.x},{config.roi.y}) "
        f"{config.roi.width}x{config.roi.height}"
    )
    print(f"FM3 rule:      white_pct < {WHITE_PCT_THRESHOLD}")
    yolo_weights = _resolve_yolo_weights()
    print(f"YOLO weights:  {yolo_weights}")
    print(f"YOLO conf:     >= {YOLO_CONF_THRESHOLD}")
    print(f"Min run:       FM1/FM2/FM4 need >= {MIN_CONSECUTIVE_FRAMES} consecutive frames")
    print(f"Video source:  {source.description}")

    model = YOLO(str(yolo_weights))
    device, use_half = _resolve_yolo_device()
    if device.startswith("cuda"):
        model.to(device)
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
        print(f"YOLO device:   {device} ({gpu_name}), half={use_half}")
    else:
        print(
            "YOLO device:   cpu  (no CUDA available — install a GPU torch "
            "build for a large speedup; see requirements.txt)"
        )
    print(f"YOLO classes:  {model.names}")

    # --- Resume: the CSV is the source of truth for "already processed". ---
    all_videos: List[VideoRef] = source.list_videos()
    processed = load_processed_videos(OUTPUT_CSV)
    pending: List[VideoRef] = [v for v in all_videos if v.filename not in processed]
    if args.limit is not None:
        pending = pending[: args.limit]

    print(
        f"Videos:        {len(all_videos)} total, "
        f"{len(processed)} already processed, {len(pending)} to process"
        + (f" (--limit {args.limit})" if args.limit is not None else "")
    )
    if not pending:
        print("Nothing to process. Rebuilding timelines from the store.")
        n = rebuild_timelines([v.filename for v in all_videos])
        print(f"Timeline PNG:  {OUTPUT_TIMELINE_PNG}")
        print(f"Timeline HTML: {OUTPUT_TIMELINE_HTML}  ({n} videos)")
        return

    TEMP_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    run_start = datetime.now()
    print(f"Run started:   {run_start:%Y-%m-%d %H:%M:%S}")
    print("-" * 70)

    progress = ProgressTracker(total=len(pending))
    processed_this_run = 0
    errored_this_run = 0
    for i, video in enumerate(pending, 1):
        video_start = time.monotonic()
        print(f"[{i:>4}/{len(pending)}] {video.filename} ...", end=" ", flush=True)

        local_path: Optional[Path] = None
        try:
            local_path = source.acquire(video, TEMP_VIDEO_DIR)
            result = process_video(local_path, config.roi, model, device, use_half)
        except Exception as exc:  # noqa: BLE001 - report, skip, keep going
            print(f"ERROR ({exc})")
            errored_this_run += 1
            if local_path is not None:
                source.release(local_path)
            progress.record(time.monotonic() - video_start)
            print(f"  {progress.summary()}")
            continue
        finally:
            # Hand the local path back to the source. S3 deletes it; the
            # local source intentionally leaves the user's file alone.
            if local_path is not None:
                source.release(local_path)

        if result.error:
            print(f"ERROR ({result.error})")
            errored_this_run += 1
            progress.record(time.monotonic() - video_start)
            print(f"  {progress.summary()}")
            continue

        flags = {fm: result.evidence[fm].triggered for fm in FM_ORDER}
        flag_str = " ".join(f"{fm}={int(flags[fm])}" for fm in FM_ORDER)
        print(f"frames={result.total_frames}  {flag_str}")

        # Persist progress immediately so the run is resumable per-video:
        #   1. debug images   2. per-frame flag store   3. CSV row.
        # The CSV row is written LAST so a video only counts as "processed"
        # once its flag store is on disk for the timeline rebuild.
        write_debug_images(result, config.roi)
        start_dt = parse_video_timestamp(result.video_filename)
        save_frame_flags(
            FLAG_STORE_DIR,
            result.video_filename,
            result.total_frames,
            result.fps,
            start_dt.isoformat() if start_dt is not None else "",
            result.fm_frame_flags,
        )
        append_csv_row(OUTPUT_CSV, result.video_filename, flags)
        processed_this_run += 1

        progress.record(time.monotonic() - video_start)
        print(f"  {progress.summary()}")

    # Rebuild the timelines from the full store (every video ever processed).
    run_elapsed = (datetime.now() - run_start).total_seconds()
    print("-" * 70)
    print(
        f"Run finished:   {datetime.now():%Y-%m-%d %H:%M:%S}  "
        f"(elapsed {_fmt_duration(run_elapsed)})"
    )
    print(
        f"Processed this run: {processed_this_run} ok, "
        f"{errored_this_run} errored, {len(pending)} attempted"
    )
    n = rebuild_timelines([v.filename for v in all_videos])
    print(f"CSV:            {OUTPUT_CSV}")
    print(f"Flag store:     {FLAG_STORE_DIR}")
    print(f"Debug (FM hit): {DEBUG_DIR}")
    print(f"Debug (clean):  {NO_DETECTION_DIR}")
    print(f"Timeline PNG:   {OUTPUT_TIMELINE_PNG}")
    print(f"Timeline HTML:  {OUTPUT_TIMELINE_HTML}  ({n} videos)")


if __name__ == "__main__":
    main()
