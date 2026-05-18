"""Evaluate FM3 (anomaly type 3 / infeed collapse) classification on synced events.

For each event folder in ``anomaly_classification/synced_events`` that contains
a ``40705473`` camera video, run the infeed-consistency detector and compare
against ground-truth from ``failure_mode_listing.xlsx`` (Failure Mode Group ==
``FM3``).

Outputs:

- ``anomaly_classification/fm3_eval_per_folder.csv`` — per-folder predictions
  with ground-truth labels.
- ``anomaly_classification/fm3_eval_summary.csv`` — confusion-matrix summary.

Console prints precision/recall/F1 plus the misclassified folder IDs.
"""
from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import openpyxl

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.run__infeed_consistency_pipeline import (
    OBJECT_COUNT_TRIGGER,
    InfeedConfig,
    load_config,
    process_video,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_PATH = PROJECT_ROOT / "anomaly_classification" / "infeed_consistency_detection.json"
SYNCED_EVENTS_DIR = PROJECT_ROOT / "anomaly_classification" / "synced_events"
GROUND_TRUTH_XLSX = PROJECT_ROOT / "failure_mode_listing.xlsx"

PER_FOLDER_CSV = PROJECT_ROOT / "anomaly_classification" / "fm3_eval_per_folder.csv"
SUMMARY_CSV = PROJECT_ROOT / "anomaly_classification" / "fm3_eval_summary.csv"
DEBUG_MASK_DIR = PROJECT_ROOT / "anomaly_classification" / "fm3_eval_debug_masks"
DEBUG_CAM71_DIR = PROJECT_ROOT / "anomaly_classification" / "fm3_eval_debug_cam71"
DEBUG_CAM73_RAW_DIR = PROJECT_ROOT / "anomaly_classification" / "fm3_eval_debug_cam73_raw"

CAMERA_ID = "40705473"
COMPANION_CAMERA_ID = "40705471"

# Ground-truth folder IDs in the xlsx can lose a leading zero (e.g. 14682
# instead of 146822). Map any GT id to its zero-padded canonical folder name
# by checking which existing folder it matches.
FOLDER_NAME_WIDTH = 6


# =============================================================================
# Ground truth
# =============================================================================


@dataclass(frozen=True)
class GroundTruthRow:
    folder_id_raw: int
    folder_id_canonical: str
    failure_mode_group: str
    is_fm3: bool


def canonicalize_folder_id(raw_id: int, known_folders: Set[str]) -> str:
    """Map a possibly-truncated id to the matching event folder name.

    1. zero-padded id is the default canonical form
    2. if zero-pad does not match, try padding to widths up to 7
    3. if still no match, return the zero-padded form so the row can be
       reported as a "missing folder"
    """
    padded = str(raw_id).zfill(FOLDER_NAME_WIDTH)
    if padded in known_folders:
        return padded
    raw = str(raw_id)
    for width in (FOLDER_NAME_WIDTH, FOLDER_NAME_WIDTH + 1):
        candidate = raw.zfill(width)
        if candidate in known_folders:
            return candidate
    # try matching by suffix - GT may drop a leading zero from a 6-digit id
    for folder in known_folders:
        if folder.lstrip("0") == raw:
            return folder
    return padded


def load_ground_truth(
    xlsx_path: Path, known_folders: Set[str]
) -> List[GroundTruthRow]:
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws = wb[wb.sheetnames[0]]

    rows: List[GroundTruthRow] = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row or row[0] is None:
            continue
        try:
            raw_id = int(row[0])
        except (TypeError, ValueError):
            continue
        fm_group = str(row[4]).strip() if row[4] is not None else ""
        canonical = canonicalize_folder_id(raw_id, known_folders)
        rows.append(
            GroundTruthRow(
                folder_id_raw=raw_id,
                folder_id_canonical=canonical,
                failure_mode_group=fm_group,
                is_fm3=(fm_group.upper() == "FM3"),
            )
        )
    return rows


# =============================================================================
# Folder scan
# =============================================================================


def list_event_folders(events_dir: Path) -> List[str]:
    return sorted(
        p.name for p in events_dir.iterdir()
        if p.is_dir() and not p.name.startswith("_")
    )


def find_camera_video(folder: Path, camera_id: str) -> Optional[Path]:
    matches = sorted(folder.glob(f"*{camera_id}*.ts"))
    if not matches:
        # In one place earlier we saw a `.tts` extension; tolerate it.
        matches = sorted(folder.glob(f"*{camera_id}*.tts"))
    return matches[0] if matches else None


# =============================================================================
# Pipeline
# =============================================================================


@dataclass(frozen=True)
class FolderPrediction:
    folder_id: str
    video_path: Optional[Path]
    processed: bool
    total_frames: int
    max_objects: int
    max_object_frame: Optional[int]
    first_trigger_frame: Optional[int]
    predicted_fm3: bool
    error: str = ""


def predict_folder(
    folder: Path,
    config: InfeedConfig,
    debug_dir: Optional[Path] = None,
    companion_debug_dir: Optional[Path] = None,
    cam73_raw_debug_dir: Optional[Path] = None,
) -> FolderPrediction:
    video = find_camera_video(folder, CAMERA_ID)
    if video is None:
        return FolderPrediction(
            folder_id=folder.name,
            video_path=None,
            processed=False,
            total_frames=0,
            max_objects=0,
            max_object_frame=None,
            first_trigger_frame=None,
            predicted_fm3=False,
            error="no_40705473_video",
        )

    try:
        # Prefix debug filename with the event folder ID so all masks sort
        # together in a single debug folder.
        debug_filename = f"{folder.name}__{video.stem}"
        result = process_video(
            video,
            config,
            debug_dir=debug_dir,
            debug_filename=debug_filename,
        )
    except Exception as exc:  # noqa: BLE001 - report and continue
        return FolderPrediction(
            folder_id=folder.name,
            video_path=video,
            processed=False,
            total_frames=0,
            max_objects=0,
            max_object_frame=None,
            first_trigger_frame=None,
            predicted_fm3=False,
            error=f"process_error: {exc}",
        )

    # All three sibling debug folders should show the same frame as the mask
    # image (the frame with max white % under the new Canny+Otsu method).
    target_frame_idx = (
        result.max_white_frame
        if result.max_white_frame is not None
        else result.max_object_frame
    )
    if target_frame_idx is not None:
        if companion_debug_dir is not None:
            export_raw_frame(
                folder=folder,
                camera_id=COMPANION_CAMERA_ID,
                target_frame_idx=target_frame_idx,
                out_dir=companion_debug_dir,
                object_count=result.max_object_count,
            )
        if cam73_raw_debug_dir is not None:
            export_raw_frame(
                folder=folder,
                camera_id=CAMERA_ID,
                target_frame_idx=target_frame_idx,
                out_dir=cam73_raw_debug_dir,
                object_count=result.max_object_count,
            )

    return FolderPrediction(
        folder_id=folder.name,
        video_path=video,
        processed=True,
        total_frames=result.total_frames,
        max_objects=result.max_object_count,
        max_object_frame=result.max_object_frame,
        first_trigger_frame=result.first_trigger_frame,
        predicted_fm3=result.anomaly_type_3,
    )


def export_raw_frame(
    folder: Path,
    camera_id: str,
    target_frame_idx: int,
    out_dir: Path,
    object_count: int,
) -> Optional[Path]:
    """Export the raw frame from `camera_id` in `folder` at `target_frame_idx`."""
    video = find_camera_video(folder, camera_id)
    if video is None:
        return None

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            return None

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = (
            out_dir
            / f"{folder.name}__{video.stem}__f{target_frame_idx:06d}_o{object_count}.png"
        )
        cv2.imwrite(str(out_path), frame)
        return out_path
    finally:
        cap.release()


# =============================================================================
# Scoring
# =============================================================================


@dataclass(frozen=True)
class ScoredFolder:
    folder_id: str
    predicted_fm3: bool
    truth_fm3: bool
    truth_group: str
    max_objects: int
    first_trigger_frame: Optional[int]
    total_frames: int
    video_filename: str
    error: str

    @property
    def outcome(self) -> str:
        if self.error:
            return "skipped"
        if self.predicted_fm3 and self.truth_fm3:
            return "TP"
        if self.predicted_fm3 and not self.truth_fm3:
            return "FP"
        if not self.predicted_fm3 and self.truth_fm3:
            return "FN"
        return "TN"


def score(
    predictions: Dict[str, FolderPrediction],
    truth_by_folder: Dict[str, GroundTruthRow],
) -> List[ScoredFolder]:
    scored: List[ScoredFolder] = []
    all_folders = sorted(set(predictions) | set(truth_by_folder))
    for folder_id in all_folders:
        pred = predictions.get(folder_id)
        truth = truth_by_folder.get(folder_id)

        if pred is None:
            # GT references a folder we don't have on disk
            scored.append(
                ScoredFolder(
                    folder_id=folder_id,
                    predicted_fm3=False,
                    truth_fm3=truth.is_fm3 if truth else False,
                    truth_group=truth.failure_mode_group if truth else "",
                    max_objects=0,
                    first_trigger_frame=None,
                    total_frames=0,
                    video_filename="",
                    error="no_folder_on_disk",
                )
            )
            continue

        scored.append(
            ScoredFolder(
                folder_id=folder_id,
                predicted_fm3=pred.predicted_fm3,
                truth_fm3=truth.is_fm3 if truth else False,
                truth_group=truth.failure_mode_group if truth else "no_gt_label",
                max_objects=pred.max_objects,
                first_trigger_frame=pred.first_trigger_frame,
                total_frames=pred.total_frames,
                video_filename=pred.video_path.name if pred.video_path else "",
                error=pred.error,
            )
        )
    return scored


def confusion_counts(scored: List[ScoredFolder]) -> Tuple[int, int, int, int, int]:
    tp = sum(1 for s in scored if s.outcome == "TP")
    fp = sum(1 for s in scored if s.outcome == "FP")
    fn = sum(1 for s in scored if s.outcome == "FN")
    tn = sum(1 for s in scored if s.outcome == "TN")
    skipped = sum(1 for s in scored if s.outcome == "skipped")
    return tp, fp, fn, tn, skipped


def write_per_folder_csv(scored: List[ScoredFolder], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "folder_id",
            "predicted_fm3",
            "truth_fm3",
            "truth_group",
            "outcome",
            "max_objects_in_roi",
            "first_trigger_frame",
            "total_frames",
            "video_filename",
            "error",
        ])
        for s in scored:
            writer.writerow([
                s.folder_id,
                s.predicted_fm3,
                s.truth_fm3,
                s.truth_group,
                s.outcome,
                s.max_objects,
                s.first_trigger_frame if s.first_trigger_frame is not None else "",
                s.total_frames,
                s.video_filename,
                s.error,
            ])


def write_summary_csv(
    scored: List[ScoredFolder], path: Path, trigger: int
) -> None:
    tp, fp, fn, tn, skipped = confusion_counts(scored)
    total_scored = tp + fp + fn + tn
    accuracy = (tp + tn) / total_scored if total_scored else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["camera_id", CAMERA_ID])
        writer.writerow(["object_count_trigger (>)", trigger])
        writer.writerow(["true_positive", tp])
        writer.writerow(["false_positive", fp])
        writer.writerow(["false_negative", fn])
        writer.writerow(["true_negative", tn])
        writer.writerow(["skipped", skipped])
        writer.writerow(["accuracy", f"{accuracy:.4f}"])
        writer.writerow(["precision", f"{precision:.4f}"])
        writer.writerow(["recall", f"{recall:.4f}"])
        writer.writerow(["f1", f"{f1:.4f}"])


# =============================================================================
# Driver
# =============================================================================


def main() -> None:
    print("=" * 70)
    print("FM3 Classification Eval (camera 40705473, anomaly_type_3 detector)")
    print("=" * 70)

    config = load_config(CONFIG_PATH)
    print(f"Config:        {CONFIG_PATH.name}")
    print(
        f"ROI:           "
        f"({config.roi.x},{config.roi.y}) {config.roi.width}x{config.roi.height}"
        if config.roi is not None
        else "ROI: full frame"
    )
    print(f"Trigger rule:  objects_in_roi > {OBJECT_COUNT_TRIGGER}")

    folders = list_event_folders(SYNCED_EVENTS_DIR)
    print(f"Event folders: {len(folders)}")
    known = set(folders)

    truth_rows = load_ground_truth(GROUND_TRUTH_XLSX, known)
    truth_by_folder: Dict[str, GroundTruthRow] = {
        r.folder_id_canonical: r for r in truth_rows
    }
    fm3_folders = sorted(r.folder_id_canonical for r in truth_rows if r.is_fm3)
    print(f"Ground truth:  {len(truth_rows)} rows, {len(fm3_folders)} FM3 folders")
    print(f"FM3 (GT):      {fm3_folders}")

    DEBUG_MASK_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_CAM71_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_CAM73_RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Debug masks:    {DEBUG_MASK_DIR}")
    print(f"Cam 71 frames:  {DEBUG_CAM71_DIR}")
    print(f"Cam 73 raw:     {DEBUG_CAM73_RAW_DIR}")

    predictions: Dict[str, FolderPrediction] = {}
    for i, folder_name in enumerate(folders, 1):
        folder_path = SYNCED_EVENTS_DIR / folder_name
        print(f"[{i:>2}/{len(folders)}] {folder_name} ...", end=" ", flush=True)
        pred = predict_folder(
            folder_path,
            config,
            debug_dir=DEBUG_MASK_DIR,
            companion_debug_dir=DEBUG_CAM71_DIR,
            cam73_raw_debug_dir=DEBUG_CAM73_RAW_DIR,
        )
        predictions[folder_name] = pred
        if pred.error:
            print(f"SKIP ({pred.error})")
        else:
            tag = "FM3" if pred.predicted_fm3 else "ok "
            print(
                f"{tag} max={pred.max_objects} "
                f"first_trigger={pred.first_trigger_frame} "
                f"frames={pred.total_frames}"
            )

    scored = score(predictions, truth_by_folder)
    tp, fp, fn, tn, skipped = confusion_counts(scored)

    write_per_folder_csv(scored, PER_FOLDER_CSV)
    write_summary_csv(scored, SUMMARY_CSV, OBJECT_COUNT_TRIGGER)

    total_scored = tp + fp + fn + tn
    accuracy = (tp + tn) / total_scored if total_scored else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print("-" * 70)
    print(f"Confusion (FM3 = positive):")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}  skipped={skipped}")
    print(
        f"  accuracy={accuracy:.3f}  precision={precision:.3f}  "
        f"recall={recall:.3f}  f1={f1:.3f}"
    )

    fps = sorted(s.folder_id for s in scored if s.outcome == "FP")
    fns = sorted(s.folder_id for s in scored if s.outcome == "FN")
    if fps:
        print(f"  False positives: {fps}")
    if fns:
        print(f"  False negatives: {fns}")
    print(f"Per-folder CSV: {PER_FOLDER_CSV}")
    print(f"Summary CSV:    {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
