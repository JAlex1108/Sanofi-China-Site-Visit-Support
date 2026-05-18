"""Export one TP / TN / FP / FN sample with full frame, ROI, and Canny+dilate.

Reads ``anomaly_classification/fm3_eval_per_folder.csv`` produced by the FM3
classification eval, picks one folder per outcome category (TP, TN, FP, FN),
opens the 40705473 video, seeks to the max-white-% frame (recomputed here so
this script does not depend on cached state), and renders a 3-panel row:

    FULL FRAME  |  ROI  |  CANNY+DILATE (with white_pct overlay)

All rows are stacked into a single combined image:
``anomaly_classification/fm3_eval_category_samples.png``.

If a category has no rows (e.g. zero FN), it is omitted with a note.
"""
from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.run__fm3_classification_eval import (
    CAMERA_ID,
    PER_FOLDER_CSV,
    SYNCED_EVENTS_DIR,
    find_camera_video,
)
from pipelines.run__infeed_consistency_pipeline import (
    CONFIG_PATH,
    InfeedConfig,
    RoiConfig,
    canny_pipeline,
    crop_to_roi,
    load_config,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PNG = PROJECT_ROOT / "anomaly_classification" / "fm3_eval_category_samples.png"

CATEGORY_ORDER = ("TP", "TN", "FP", "FN")
CATEGORY_COLORS = {
    "TP": (0, 220, 0),       # green
    "TN": (200, 200, 200),   # gray
    "FP": (0, 80, 255),      # red-orange
    "FN": (255, 80, 255),    # magenta
}


# =============================================================================
# CSV loading
# =============================================================================


@dataclass(frozen=True)
class EvalRow:
    folder_id: str
    outcome: str
    predicted_fm3: bool
    truth_fm3: bool
    truth_group: str


def load_eval_rows(csv_path: Path) -> List[EvalRow]:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Eval CSV not found: {csv_path}. Run run__fm3_classification_eval.py first."
        )
    rows: List[EvalRow] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                EvalRow(
                    folder_id=r["folder_id"],
                    outcome=r["outcome"],
                    predicted_fm3=r["predicted_fm3"].lower() == "true",
                    truth_fm3=r["truth_fm3"].lower() == "true",
                    truth_group=r.get("truth_group", ""),
                )
            )
    return rows


def pick_one_per_category(rows: List[EvalRow]) -> Dict[str, Optional[EvalRow]]:
    """Pick the first row for each outcome category in CSV order (stable)."""
    picks: Dict[str, Optional[EvalRow]] = {c: None for c in CATEGORY_ORDER}
    for r in rows:
        if r.outcome in picks and picks[r.outcome] is None:
            picks[r.outcome] = r
    return picks


# =============================================================================
# Max-white frame discovery
# =============================================================================


@dataclass(frozen=True)
class FrameSample:
    frame_idx: int
    frame_bgr: np.ndarray
    white_pct: float


def find_max_white_frame(
    video_path: Path, roi: RoiConfig
) -> Optional[FrameSample]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    best: Optional[FrameSample] = None
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            try:
                _, _, _, _, white_pct = canny_pipeline(frame, roi)
            except ValueError:
                frame_idx += 1
                continue
            if best is None or white_pct > best.white_pct:
                best = FrameSample(
                    frame_idx=frame_idx,
                    frame_bgr=frame.copy(),
                    white_pct=white_pct,
                )
            frame_idx += 1
    finally:
        cap.release()

    return best


# =============================================================================
# Panel rendering
# =============================================================================


PANEL_HEIGHT = 360
BORDER_THICKNESS = 4
LABEL_STRIP_H = 36
GAP_W = 24
ROW_GAP = 32
SIDE_PAD = 24
TOP_PAD = 24
CATEGORY_LABEL_W = 220

PANEL_COLORS = {
    "FULL FRAME": (200, 200, 200),
    "ROI": (0, 200, 255),
    "CANNY+DILATE": (255, 80, 0),
}


def fit_height(img: np.ndarray, h: int) -> np.ndarray:
    if img.shape[0] == h:
        return img
    scale = h / img.shape[0]
    return cv2.resize(img, (int(img.shape[1] * scale), h))


def labeled_panel(
    img: np.ndarray, title: str, color: Tuple[int, int, int], overlay_text: Optional[str] = None
) -> np.ndarray:
    inner_h = PANEL_HEIGHT
    fitted = fit_height(img, inner_h)
    panel = np.zeros((inner_h + LABEL_STRIP_H, fitted.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        panel,
        title,
        (10, LABEL_STRIP_H - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )
    panel[LABEL_STRIP_H:, :] = fitted

    if overlay_text:
        text_size, _ = cv2.getTextSize(
            overlay_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        tx = panel.shape[1] - text_size[0] - 12
        ty = LABEL_STRIP_H + text_size[1] + 12
        cv2.rectangle(
            panel,
            (tx - 8, ty - text_size[1] - 8),
            (tx + text_size[0] + 8, ty + 8),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            panel,
            overlay_text,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

    cv2.rectangle(
        panel,
        (0, 0),
        (panel.shape[1] - 1, panel.shape[0] - 1),
        color,
        BORDER_THICKNESS,
    )
    return panel


def build_row(
    category: str,
    eval_row: EvalRow,
    sample: Optional[FrameSample],
    roi: RoiConfig,
    error: str = "",
) -> np.ndarray:
    panels: List[np.ndarray] = []

    if sample is not None:
        full = sample.frame_bgr
        roi_crop = crop_to_roi(sample.frame_bgr, roi)
        _, _, _, dilated, _ = canny_pipeline(sample.frame_bgr, roi)
        dilated_bgr = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)

        panels.append(
            labeled_panel(
                full,
                "FULL FRAME",
                PANEL_COLORS["FULL FRAME"],
                overlay_text=f"frame={sample.frame_idx}",
            )
        )
        panels.append(labeled_panel(roi_crop, "ROI", PANEL_COLORS["ROI"]))
        panels.append(
            labeled_panel(
                dilated_bgr,
                "CANNY+DILATE",
                PANEL_COLORS["CANNY+DILATE"],
                overlay_text=f"white={sample.white_pct:.2f}%",
            )
        )
    else:
        placeholder = np.zeros((PANEL_HEIGHT, 480, 3), dtype=np.uint8)
        cv2.putText(
            placeholder,
            f"NO DATA: {error or 'frame_unavailable'}",
            (20, PANEL_HEIGHT // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        for title in ("FULL FRAME", "ROI", "CANNY+DILATE"):
            panels.append(labeled_panel(placeholder, title, PANEL_COLORS[title]))

    panel_h = panels[0].shape[0]
    gap = np.zeros((panel_h, GAP_W, 3), dtype=np.uint8)
    interleaved: List[np.ndarray] = []
    for i, p in enumerate(panels):
        if i > 0:
            interleaved.append(gap)
        interleaved.append(p)
    row_strip = np.hstack(interleaved)

    # Category label column on the left.
    cat_color = CATEGORY_COLORS.get(category, (255, 255, 255))
    cat_strip = np.zeros((panel_h, CATEGORY_LABEL_W, 3), dtype=np.uint8)
    cv2.putText(
        cat_strip,
        category,
        (16, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.6,
        cat_color,
        4,
    )
    cv2.putText(
        cat_strip,
        eval_row.folder_id,
        (16, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        cat_strip,
        f"pred={'FM3' if eval_row.predicted_fm3 else 'ok'}",
        (16, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1,
    )
    cv2.putText(
        cat_strip,
        f"truth={'FM3' if eval_row.truth_fm3 else (eval_row.truth_group or 'ok')}",
        (16, 175),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1,
    )
    cv2.rectangle(
        cat_strip,
        (0, 0),
        (cat_strip.shape[1] - 1, cat_strip.shape[0] - 1),
        cat_color,
        BORDER_THICKNESS,
    )

    gap_col = np.zeros((panel_h, GAP_W, 3), dtype=np.uint8)
    return np.hstack([cat_strip, gap_col, row_strip])


def stack_rows(rows: List[np.ndarray]) -> np.ndarray:
    max_w = max(r.shape[1] for r in rows)
    padded: List[np.ndarray] = []
    for r in rows:
        if r.shape[1] < max_w:
            pad = np.zeros((r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8)
            r = np.hstack([r, pad])
        padded.append(r)

    row_h = padded[0].shape[0]
    spacer = np.zeros((ROW_GAP, max_w, 3), dtype=np.uint8)
    stacked: List[np.ndarray] = []
    for i, r in enumerate(padded):
        if i > 0:
            stacked.append(spacer)
        stacked.append(r)
    inner = np.vstack(stacked)

    canvas = np.zeros(
        (inner.shape[0] + TOP_PAD * 2, inner.shape[1] + SIDE_PAD * 2, 3),
        dtype=np.uint8,
    )
    canvas[TOP_PAD:TOP_PAD + inner.shape[0], SIDE_PAD:SIDE_PAD + inner.shape[1]] = inner
    return canvas


# =============================================================================
# Driver
# =============================================================================


def main() -> None:
    print("=" * 70)
    print("FM3 Category Samples (TP / TN / FP / FN)")
    print("=" * 70)

    config = load_config(CONFIG_PATH)
    if config.roi is None:
        raise RuntimeError("ROI not configured")
    print(
        f"ROI: ({config.roi.x},{config.roi.y}) "
        f"{config.roi.width}x{config.roi.height}"
    )

    eval_rows = load_eval_rows(PER_FOLDER_CSV)
    picks = pick_one_per_category(eval_rows)

    rendered_rows: List[np.ndarray] = []
    for category in CATEGORY_ORDER:
        pick = picks[category]
        if pick is None:
            print(f"{category}: (none in eval CSV - skipping)")
            continue
        folder_path = SYNCED_EVENTS_DIR / pick.folder_id
        video = find_camera_video(folder_path, CAMERA_ID)
        if video is None:
            print(f"{category}: {pick.folder_id} - no {CAMERA_ID} video")
            rendered_rows.append(
                build_row(category, pick, None, config.roi, error="no_video")
            )
            continue

        print(f"{category}: {pick.folder_id} -> {video.name}", flush=True)
        sample = find_max_white_frame(video, config.roi)
        if sample is None:
            print(f"   no readable frames")
            rendered_rows.append(
                build_row(category, pick, None, config.roi, error="no_frames")
            )
            continue
        print(
            f"   max-white frame={sample.frame_idx}  white_pct={sample.white_pct:.2f}%"
        )
        rendered_rows.append(build_row(category, pick, sample, config.roi))

    if not rendered_rows:
        raise RuntimeError("No rows rendered - nothing in eval CSV.")

    combined = stack_rows(rendered_rows)
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT_PNG), combined)
    print("-" * 70)
    print(f"Wrote: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
