"""Infeed Consistency Detection - flag anomaly type 3 on a single video.

For the target .tts video, this pipeline:

1. Loads the hue/morphology/contour/ROI config from
   ``anomaly_classification/infeed_consistency_detection.json``.
2. For every frame, crops to the ROI defined in the config, applies the hue
   mask, morphology, and contour filtering, then counts objects (contours).
3. If ANY frame contains more than one object inside the ROI, the entire
   video is flagged as anomaly_type_3 = True. Otherwise False.
4. Writes a single-row CSV whose first column is the video start time
   (parsed from the filename) and which records the anomaly flag.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared_functions.timestamp_utils import parse_video_timestamp


# =============================================================================
# PIPELINE CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Single-video FM3 standalone runner — path to the .tts file to analyse.
# Override via the ``INFEED_VIDEO_PATH`` environment variable. The default
# below is only used when this module is run directly; importing the module
# (which the main detector does) never reads this constant at run time.
VIDEO_PATH = Path(
    os.environ.get(
        "INFEED_VIDEO_PATH",
        str(PROJECT_ROOT / "anomaly_classification" / "sample.tts"),
    )
)

CONFIG_PATH = PROJECT_ROOT / "anomaly_classification" / "infeed_consistency_detection.json"

OUTPUT_CSV = (
    PROJECT_ROOT / "anomaly_classification" / "infeed_consistency_anomaly3.csv"
)

# An object count strictly greater than this in a single frame triggers
# anomaly type 3 for the whole video.
OBJECT_COUNT_TRIGGER = 1


# =============================================================================
# CONFIG LOADING
# =============================================================================


@dataclass(frozen=True)
class HueConfig:
    hue_min: int
    hue_max: int
    saturation_min: int
    saturation_max: int
    value_min: int
    value_max: int


@dataclass(frozen=True)
class MorphConfig:
    enabled: bool
    operation: str
    iterations: int
    kernel_size: int


@dataclass(frozen=True)
class ContourConfig:
    enabled: bool
    min_area: int
    max_area: int  # 0 == no max


@dataclass(frozen=True)
class RoiConfig:
    enabled: bool
    x: int
    y: int
    width: int
    height: int

    def as_xyxy(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)


@dataclass(frozen=True)
class InfeedConfig:
    hue: HueConfig
    morphology: MorphConfig
    contour: ContourConfig
    roi: Optional[RoiConfig]


def load_config(config_path: Path) -> InfeedConfig:
    with open(config_path, "r") as f:
        raw = json.load(f)

    hue_raw = raw.get("hue_detection", {})
    hue = HueConfig(
        hue_min=int(hue_raw.get("hue_min", 0)),
        hue_max=int(hue_raw.get("hue_max", 179)),
        saturation_min=int(hue_raw.get("saturation_min", 0)),
        saturation_max=int(hue_raw.get("saturation_max", 255)),
        value_min=int(hue_raw.get("value_min", 0)),
        value_max=int(hue_raw.get("value_max", 255)),
    )

    morph_raw = raw.get("morphology", {})
    morph = MorphConfig(
        enabled=bool(morph_raw.get("enabled", False)),
        operation=str(morph_raw.get("operation", "dilate")),
        iterations=int(morph_raw.get("iterations", 1)),
        kernel_size=int(morph_raw.get("kernel_size", 3)),
    )

    contour_raw = raw.get("contour_filtering", {})
    contour = ContourConfig(
        enabled=bool(contour_raw.get("enabled", True)),
        min_area=int(contour_raw.get("min_area", 0)),
        max_area=int(contour_raw.get("max_area", 0)),
    )

    roi_raw = raw.get("roi", {})
    active = roi_raw.get("active_rois", []) or []
    roi_obj: Optional[RoiConfig] = None
    if roi_raw.get("enabled", False) and active:
        first = active[0]
        if first.get("enabled", True):
            roi_obj = RoiConfig(
                enabled=True,
                x=int(first["x"]),
                y=int(first["y"]),
                width=int(first["width"]),
                height=int(first["height"]),
            )

    return InfeedConfig(hue=hue, morphology=morph, contour=contour, roi=roi_obj)


# =============================================================================
# IMAGE PROCESSING
# =============================================================================


def crop_to_roi(frame: np.ndarray, roi: Optional[RoiConfig]) -> np.ndarray:
    if roi is None:
        return frame
    x1, y1, x2, y2 = roi.as_xyxy()
    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"ROI {roi.as_xyxy()} is outside frame size {(w, h)}")
    return frame[y1:y2, x1:x2]


def apply_hue_mask(frame: np.ndarray, hue: HueConfig) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([hue.hue_min, hue.saturation_min, hue.value_min])
    upper = np.array([hue.hue_max, hue.saturation_max, hue.value_max])
    return cv2.inRange(hsv, lower, upper)


def apply_morphology(mask: np.ndarray, morph: MorphConfig) -> np.ndarray:
    if not morph.enabled:
        return mask
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (morph.kernel_size, morph.kernel_size)
    )
    op = morph.operation
    if op == "dilate":
        return cv2.dilate(mask, kernel, iterations=morph.iterations)
    if op == "erode":
        return cv2.erode(mask, kernel, iterations=morph.iterations)
    if op == "open":
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if op == "close":
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    raise ValueError(f"Unsupported morphology operation: {op}")


def count_objects(mask: np.ndarray, contour: ContourConfig) -> int:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contour.enabled:
        return len(contours)
    valid = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < contour.min_area:
            continue
        if contour.max_area and area > contour.max_area:
            continue
        valid += 1
    return valid


def count_objects_in_frame(frame: np.ndarray, config: InfeedConfig) -> int:
    cropped = crop_to_roi(frame, config.roi)
    mask = apply_hue_mask(cropped, config.hue)
    mask = apply_morphology(mask, config.morphology)
    return count_objects(mask, config.contour)


# =============================================================================
# PIPELINE
# =============================================================================


@dataclass(frozen=True)
class VideoAnomalyResult:
    video_filename: str
    video_start_time: datetime
    total_frames: int
    max_object_count: int
    max_object_frame: Optional[int]
    first_trigger_frame: Optional[int]
    anomaly_type_3: bool
    debug_image_path: Optional[Path] = None
    max_white_pct: float = 0.0
    max_white_frame: Optional[int] = None


# =============================================================================
# Canny / Otsu metric (right-half of ROI)
# =============================================================================


CANNY_DILATE_KERNEL_SIZE = 3
CANNY_DILATE_ITERATIONS = 1
CANNY_LOW_THRESHOLD = 30
CANNY_HIGH_THRESHOLD = 90


def crop_right_half_of_roi(frame: np.ndarray, roi: RoiConfig) -> np.ndarray:
    x_mid = roi.x + roi.width // 2
    x1, y1, x2, y2 = x_mid, roi.y, roi.x + roi.width, roi.y + roi.height
    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Right-half ROI {(x1, y1, x2, y2)} invalid for frame {(w, h)}")
    return frame[y1:y2, x1:x2]


def canny_pipeline(
    frame: np.ndarray, roi: RoiConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Right-half ROI -> grayscale -> Canny (sensitive) -> dilate 3x3.

    Otsu is still computed for visualization in the debug image, but the
    metric and dilation use Canny applied directly to the grayscale crop.

    Returns: (right_half_crop, otsu_viz, edges, dilated, white_pct)
    """
    crop = crop_right_half_of_roi(frame, roi)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(
        gray,
        CANNY_LOW_THRESHOLD,
        CANNY_HIGH_THRESHOLD,
        L2gradient=True,
    )
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (CANNY_DILATE_KERNEL_SIZE, CANNY_DILATE_KERNEL_SIZE)
    )
    dilated = cv2.dilate(edges, kernel, iterations=CANNY_DILATE_ITERATIONS)
    total = dilated.size
    white_pct = (
        float(np.count_nonzero(dilated)) / total * 100.0 if total > 0 else 0.0
    )
    return crop, otsu, edges, dilated, white_pct


def build_canny_debug_image(
    frame: np.ndarray,
    roi: RoiConfig,
    frame_idx: int,
    white_pct: float,
    video_filename: str,
) -> np.ndarray:
    """4-panel debug: full ROI | right-half crop | Otsu | Canny+dilate.

    Panels are spread apart with a gap and given color-coded borders so it is
    obvious which is which.
    """
    full_roi = crop_to_roi(frame, roi)
    right_crop, otsu, _, dilated, _ = canny_pipeline(frame, roi)
    otsu_bgr = cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)
    dilated_bgr = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)

    panel_titles = ("FULL ROI", "RIGHT HALF", "OTSU", "CANNY+DILATE")
    border_colors = (
        (200, 200, 200),  # FULL ROI - white-ish
        (0, 200, 255),    # RIGHT HALF - amber
        (0, 255, 0),      # OTSU - green
        (255, 80, 0),     # CANNY+DILATE - blue
    )
    panels = (full_roi, right_crop, otsu_bgr, dilated_bgr)

    h = max(p.shape[0] for p in panels)

    def fit(img: np.ndarray) -> np.ndarray:
        if img.shape[0] == h:
            return img
        scale = h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * scale), h))

    border_thickness = 4
    label_strip_h = 32
    gap_w = 24
    canvas_pad = 20
    banner_h = 60

    fitted = [fit(p) for p in panels]

    framed: List[np.ndarray] = []
    for img, color, title in zip(fitted, border_colors, panel_titles):
        labeled = np.zeros(
            (img.shape[0] + label_strip_h, img.shape[1], 3), dtype=np.uint8
        )
        cv2.putText(
            labeled,
            title,
            (10, label_strip_h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
        labeled[label_strip_h:, :] = img
        cv2.rectangle(
            labeled,
            (0, 0),
            (labeled.shape[1] - 1, labeled.shape[0] - 1),
            color,
            border_thickness,
        )
        framed.append(labeled)

    panel_h = framed[0].shape[0]
    gap = np.zeros((panel_h, gap_w, 3), dtype=np.uint8)

    interleaved: List[np.ndarray] = []
    for i, p in enumerate(framed):
        if i > 0:
            interleaved.append(gap)
        interleaved.append(p)
    row = np.hstack(interleaved)

    canvas = np.zeros(
        (panel_h + canvas_pad * 2, row.shape[1] + canvas_pad * 2, 3),
        dtype=np.uint8,
    )
    canvas[canvas_pad:canvas_pad + panel_h, canvas_pad:canvas_pad + row.shape[1]] = row

    banner = np.zeros((banner_h, canvas.shape[1], 3), dtype=np.uint8)
    text = f"{video_filename}  frame={frame_idx}  white_pct={white_pct:.2f}%"
    cv2.putText(
        banner, text, (canvas_pad, 38),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
    )

    return np.vstack([banner, canvas])


def build_debug_image(
    frame: np.ndarray,
    config: InfeedConfig,
    frame_idx: int,
    object_count: int,
    video_filename: str,
) -> np.ndarray:
    """Side-by-side debug: ROI crop | post-morph mask | ROI crop + contours."""
    cropped = crop_to_roi(frame, config.roi)
    mask = apply_hue_mask(cropped, config.hue)
    mask = apply_morphology(mask, config.morphology)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if config.contour.enabled:
        kept = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < config.contour.min_area:
                continue
            if config.contour.max_area and area > config.contour.max_area:
                continue
            kept.append(c)
        contours = kept

    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    overlay = cropped.copy()
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
    cv2.drawContours(mask_bgr, contours, -1, (0, 0, 255), 2)

    h = max(cropped.shape[0], mask_bgr.shape[0], overlay.shape[0])

    def fit(img: np.ndarray) -> np.ndarray:
        if img.shape[0] == h:
            return img
        scale = h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * scale), h))

    panel = np.hstack([fit(cropped), fit(mask_bgr), fit(overlay)])

    banner_h = 60
    banner = np.zeros((banner_h, panel.shape[1], 3), dtype=np.uint8)
    text = (
        f"{video_filename}  frame={frame_idx}  objects={object_count}  "
        f"min_area={config.contour.min_area}"
    )
    cv2.putText(
        banner, text, (10, 38),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
    )

    labels_h = 28
    labels = np.zeros((labels_h, panel.shape[1], 3), dtype=np.uint8)
    third = panel.shape[1] // 3
    for i, label in enumerate(("ROI", "MASK", "OVERLAY")):
        cv2.putText(
            labels, label, (i * third + 10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )

    return np.vstack([banner, panel, labels])


def process_video(
    video_path: Path,
    config: InfeedConfig,
    debug_dir: Optional[Path] = None,
    debug_filename: Optional[str] = None,
) -> VideoAnomalyResult:
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    start_time = parse_video_timestamp(video_path.name)
    if start_time is None:
        raise ValueError(f"Could not parse timestamp from filename: {video_path.name}")
    start_time = start_time.replace(tzinfo=timezone.utc)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    try:
        frame_idx = 0
        max_count = 0
        max_count_frame_idx: Optional[int] = None
        first_trigger: Optional[int] = None

        max_white_pct = -1.0
        max_white_frame_idx: Optional[int] = None
        max_white_frame: Optional[np.ndarray] = None

        canny_enabled = config.roi is not None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            count = count_objects_in_frame(frame, config)
            if count > max_count:
                max_count = count
                max_count_frame_idx = frame_idx
            if first_trigger is None and count > OBJECT_COUNT_TRIGGER:
                first_trigger = frame_idx

            if canny_enabled:
                _, _, _, _, white_pct = canny_pipeline(frame, config.roi)  # type: ignore[arg-type]
                if white_pct > max_white_pct:
                    max_white_pct = white_pct
                    max_white_frame_idx = frame_idx
                    if debug_dir is not None:
                        max_white_frame = frame.copy()

            frame_idx += 1

        debug_path: Optional[Path] = None
        if (
            debug_dir is not None
            and max_white_frame is not None
            and max_white_frame_idx is not None
            and config.roi is not None
        ):
            debug_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(debug_filename or video_path.name).stem
            debug_path = (
                debug_dir
                / f"{stem}__f{max_white_frame_idx:06d}_w{max_white_pct:05.1f}.png"
            )
            image = build_canny_debug_image(
                max_white_frame,
                config.roi,
                max_white_frame_idx,
                max_white_pct,
                video_path.name,
            )
            cv2.imwrite(str(debug_path), image)

        return VideoAnomalyResult(
            video_filename=video_path.name,
            video_start_time=start_time,
            total_frames=frame_idx,
            max_object_count=max_count,
            max_object_frame=max_count_frame_idx,
            first_trigger_frame=first_trigger,
            anomaly_type_3=first_trigger is not None,
            debug_image_path=debug_path,
            max_white_pct=max(max_white_pct, 0.0),
            max_white_frame=max_white_frame_idx,
        )
    finally:
        cap.release()


def write_csv(result: VideoAnomalyResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time",
            "video_filename",
            "total_frames",
            "max_object_count_in_roi",
            "first_trigger_frame",
            "anomaly_type_3",
        ])
        writer.writerow([
            result.video_start_time.isoformat(),
            result.video_filename,
            result.total_frames,
            result.max_object_count,
            result.first_trigger_frame if result.first_trigger_frame is not None else "",
            result.anomaly_type_3,
        ])


def run(
    video_path: Path = VIDEO_PATH,
    config_path: Path = CONFIG_PATH,
    output_csv: Path = OUTPUT_CSV,
) -> VideoAnomalyResult:
    print("=" * 60)
    print("Infeed Consistency Detection (anomaly type 3)")
    print("=" * 60)
    print(f"Video:  {video_path}")
    print(f"Config: {config_path}")
    print(f"Output: {output_csv}")

    config = load_config(config_path)
    roi_str = (
        f"({config.roi.x}, {config.roi.y}) {config.roi.width}x{config.roi.height}"
        if config.roi is not None
        else "FULL FRAME"
    )
    print(f"ROI:    {roi_str}")
    print(
        f"Hue:    H[{config.hue.hue_min}-{config.hue.hue_max}] "
        f"S[{config.hue.saturation_min}-{config.hue.saturation_max}] "
        f"V[{config.hue.value_min}-{config.hue.value_max}]"
    )
    print(
        f"Morph:  enabled={config.morphology.enabled} "
        f"op={config.morphology.operation} "
        f"k={config.morphology.kernel_size} "
        f"it={config.morphology.iterations}"
    )
    print(
        f"Contour: min_area={config.contour.min_area} "
        f"max_area={config.contour.max_area}"
    )

    result = process_video(video_path, config)

    print("-" * 60)
    print(f"Video start time:        {result.video_start_time.isoformat()}")
    print(f"Total frames processed:  {result.total_frames}")
    print(f"Max objects in any frame:{result.max_object_count}")
    print(f"First trigger frame:     {result.first_trigger_frame}")
    print(f"anomaly_type_3:          {result.anomaly_type_3}")

    write_csv(result, output_csv)
    print(f"CSV written to:          {output_csv}")

    return result


def main() -> None:
    run()


if __name__ == "__main__":
    main()
