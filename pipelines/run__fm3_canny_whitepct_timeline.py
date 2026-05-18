"""Canny-on-Otsu white-pixel-percentage timeline across all 40705473 videos.

For every event folder under ``anomaly_classification/synced_events`` (sorted by
folder name), this pipeline processes the ``40705473`` camera video and, for
every frame:

1. Crops to the ROI from ``infeed_consistency_detection.json``.
2. Keeps only the RIGHT half of the ROI (columns roi_w/2 .. roi_w).
3. Converts to grayscale, applies Otsu thresholding.
4. Runs Canny edge detection on the Otsu result.
5. Dilates the edges with a 3x3 kernel, one iteration.
6. Records the percentage of white pixels in that final mask.

All per-frame values are concatenated across all videos onto a single x-axis
(global frame index). The background is shaded over the frame ranges that
correspond to ground-truth FM3 folders from ``failure_mode_listing.xlsx``.
Thin vertical separators and folder-ID labels mark each video boundary.

Output:
- ``anomaly_classification/fm3_canny_whitepct_timeline.png``
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.run__fm3_classification_eval import (
    CAMERA_ID,
    GROUND_TRUTH_XLSX,
    SYNCED_EVENTS_DIR,
    canonicalize_folder_id,
    find_camera_video,
    list_event_folders,
    load_ground_truth,
)
from pipelines.run__infeed_consistency_pipeline import (
    CONFIG_PATH,
    InfeedConfig,
    RoiConfig,
    load_config,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PNG = PROJECT_ROOT / "anomaly_classification" / "fm3_canny_whitepct_timeline.png"

DILATE_KERNEL_SIZE = 3
DILATE_ITERATIONS = 1


# =============================================================================
# Per-frame metric
# =============================================================================


def right_half_xyxy(roi: RoiConfig) -> Tuple[int, int, int, int]:
    """Return absolute (x1, y1, x2, y2) for the right half of the ROI."""
    x_mid = roi.x + roi.width // 2
    return (x_mid, roi.y, roi.x + roi.width, roi.y + roi.height)


def crop_right_half(frame: np.ndarray, roi: RoiConfig) -> np.ndarray:
    x1, y1, x2, y2 = right_half_xyxy(roi)
    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Right-half ROI {(x1, y1, x2, y2)} invalid for frame {(w, h)}")
    return frame[y1:y2, x1:x2]


CANNY_LOW = 30
CANNY_HIGH = 90


def compute_white_pct(frame: np.ndarray, roi: RoiConfig, kernel: np.ndarray) -> float:
    crop = crop_right_half(frame, roi)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH, L2gradient=True)
    dilated = cv2.dilate(edges, kernel, iterations=DILATE_ITERATIONS)
    total = dilated.size
    if total == 0:
        return 0.0
    return float(np.count_nonzero(dilated)) / total * 100.0


# =============================================================================
# Per-video processing
# =============================================================================


@dataclass(frozen=True)
class VideoSeries:
    folder_id: str
    video_filename: str
    white_pct: List[float]
    error: str = ""


def process_video_series(
    video_path: Path,
    folder_id: str,
    roi: RoiConfig,
    kernel: np.ndarray,
) -> VideoSeries:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return VideoSeries(
            folder_id=folder_id,
            video_filename=video_path.name,
            white_pct=[],
            error="cv2_open_failed",
        )

    values: List[float] = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            try:
                values.append(compute_white_pct(frame, roi, kernel))
            except ValueError as exc:
                return VideoSeries(
                    folder_id=folder_id,
                    video_filename=video_path.name,
                    white_pct=values,
                    error=f"roi_error: {exc}",
                )
    finally:
        cap.release()

    return VideoSeries(
        folder_id=folder_id,
        video_filename=video_path.name,
        white_pct=values,
    )


# =============================================================================
# Plotting
# =============================================================================


@dataclass(frozen=True)
class VideoSegment:
    folder_id: str
    start_x: int
    end_x: int  # exclusive
    is_fm3: bool


def plot_timeline(
    segments: List[VideoSegment],
    series: List[float],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(24, 6))

    # Background shading for FM3 segments.
    fm3_label_used = False
    for seg in segments:
        if not seg.is_fm3:
            continue
        ax.axvspan(
            seg.start_x,
            seg.end_x,
            color="#ff6666",
            alpha=0.18,
            linewidth=0,
            label="FM3 (ground truth)" if not fm3_label_used else None,
        )
        fm3_label_used = True

    # Vertical separators between videos.
    for seg in segments[1:]:
        ax.axvline(seg.start_x, color="#999999", linewidth=0.4, linestyle=":")

    # Main curve.
    if series:
        ax.plot(np.arange(len(series)), series, color="#1f77b4", linewidth=0.6)

    # Folder labels at each segment start.
    y_top = max(series) if series else 100.0
    label_y = y_top * 1.02 if y_top > 0 else 1.0
    for seg in segments:
        ax.text(
            seg.start_x,
            label_y,
            seg.folder_id,
            rotation=90,
            fontsize=7,
            color="#444444",
            verticalalignment="bottom",
            horizontalalignment="left",
        )

    ax.set_xlabel("Global frame index (videos concatenated, sorted by folder name)")
    ax.set_ylabel("White pixel % (Canny on Otsu, dilated 3x3)")
    ax.set_title(
        f"FM3 detector experiment: white-pixel % in right-half ROI for {CAMERA_ID}"
    )
    if series:
        ax.set_xlim(0, len(series))
    ax.margins(x=0)
    if fm3_label_used:
        ax.legend(loc="upper right")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# =============================================================================
# Driver
# =============================================================================


def main() -> None:
    print("=" * 70)
    print(f"FM3 Canny+Otsu white-pixel timeline (camera {CAMERA_ID})")
    print("=" * 70)

    config = load_config(CONFIG_PATH)
    if config.roi is None:
        raise RuntimeError("ROI not configured in infeed_consistency_detection.json")
    print(
        f"ROI:           ({config.roi.x},{config.roi.y}) "
        f"{config.roi.width}x{config.roi.height}"
    )
    rx1, ry1, rx2, ry2 = right_half_xyxy(config.roi)
    print(f"Right half:    ({rx1},{ry1}) -> ({rx2},{ry2})")

    folders = list_event_folders(SYNCED_EVENTS_DIR)
    known: Set[str] = set(folders)
    print(f"Event folders: {len(folders)}")

    truth_rows = load_ground_truth(GROUND_TRUTH_XLSX, known)
    fm3_folders: Set[str] = {
        r.folder_id_canonical for r in truth_rows if r.is_fm3
    }
    print(f"FM3 folders:   {sorted(fm3_folders)}")

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (DILATE_KERNEL_SIZE, DILATE_KERNEL_SIZE)
    )

    all_values: List[float] = []
    segments: List[VideoSegment] = []

    for i, folder_name in enumerate(folders, 1):
        folder_path = SYNCED_EVENTS_DIR / folder_name
        video = find_camera_video(folder_path, CAMERA_ID)
        if video is None:
            print(f"[{i:>2}/{len(folders)}] {folder_name} ... SKIP (no {CAMERA_ID})")
            continue

        print(f"[{i:>2}/{len(folders)}] {folder_name} ...", end=" ", flush=True)
        series = process_video_series(video, folder_name, config.roi, kernel)
        if series.error:
            print(f"SKIP ({series.error}) after {len(series.white_pct)} frames")
            continue

        start_x = len(all_values)
        all_values.extend(series.white_pct)
        end_x = len(all_values)
        segments.append(
            VideoSegment(
                folder_id=folder_name,
                start_x=start_x,
                end_x=end_x,
                is_fm3=(folder_name in fm3_folders),
            )
        )
        print(
            f"frames={len(series.white_pct)} "
            f"min={min(series.white_pct):.2f} "
            f"max={max(series.white_pct):.2f} "
            f"mean={sum(series.white_pct) / max(len(series.white_pct), 1):.2f}"
        )

    print("-" * 70)
    print(f"Total frames:  {len(all_values)}")
    print(f"Segments:      {len(segments)} "
          f"({sum(1 for s in segments if s.is_fm3)} FM3, "
          f"{sum(1 for s in segments if not s.is_fm3)} non-FM3)")

    plot_timeline(segments, all_values, OUTPUT_PNG)
    print(f"Plot written:  {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
