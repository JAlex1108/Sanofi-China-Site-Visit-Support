# anomaly-classification

Multi-failure-mode detection on packaging-line camera footage. Processes
`.ts` videos one at a time (from S3 or a local folder), runs two detectors per
frame, and emits per-video failure-mode flags plus debug imagery and timeline
graphs.

## Failure modes

| Mode | Trigger                                                                                          |
|------|--------------------------------------------------------------------------------------------------|
| FM1  | YOLO `displaced_box` (anywhere) on ≥ 2 consecutive frames — *Suction Release*                    |
| FM2  | YOLO `displaced_box` in left 25 % of frame **and** ≥ 2 % of frame area, ≥ 2 consecutive frames — *Fallen Out Infeed* |
| FM3  | White-pixel % (Canny+dilate on right half of ROI) drops below 50 % on any single frame — *Infeed Collapse* |
| FM4  | YOLO `empty_cups` (anywhere) on ≥ 2 consecutive frames — *Empty Cups*                            |

Tuning knobs live at the top of
[`pipelines/run__failure_mode_detection.py`](pipelines/run__failure_mode_detection.py)
(`WHITE_PCT_THRESHOLD`, `YOLO_CONF_THRESHOLD`, `MIN_CONSECUTIVE_FRAMES`,
`FM2_LEFT_FRACTION`, `FM2_LARGE_AREA_FRACTION`).

## Requirements

- Python **3.10 – 3.12** (pinned dependencies were verified on 3.12.10).
- ~3 GB free disk for the venv (ultralytics pulls torch + torchvision).
- The trained YOLO weights file `best.pt`. **Distributed separately from
  this repo** — see [YOLO weights](#yolo-weights) below.

## Setup

### 1. Clone

```bash
git clone https://github.com/JAlex1108/anomaly-classification.git
cd anomaly-classification
```

### 2. Create a venv and install dependencies

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` lists direct dependencies pinned to verified-working
versions. `requirements-lock.txt` is the full frozen environment if you need
byte-identical reproducibility.

### 3. Configure environment

```bash
cp .env.example .env
```

Then open `.env` and set `YOLO_WEIGHTS_PATH` (see next section).

### 4. Smoke-test the install

With the venv active, from the repo root:

```bash
python -c "import cv2, numpy, matplotlib, plotly, boto3, openpyxl; from ultralytics import YOLO; from dotenv import load_dotenv; from shared_functions.timestamp_utils import parse_video_timestamp; from pipelines.fm_video_source import make_video_source; from pipelines.fm_state_store import load_processed_videos; print('OK')"
```

If it prints `OK` everything is wired up. (`YOLO` import here only loads the
class; it does not need `best.pt` to be present.)

## YOLO weights

`best.pt` is **not** in this repo. Place the file you received separately
anywhere on disk and point `YOLO_WEIGHTS_PATH` at it in `.env`:

```dotenv
# .env
YOLO_WEIGHTS_PATH=C:/Users/you/weights/best.pt
```

Forward slashes work on Windows. Linux/macOS uses a regular absolute path.

The pipeline raises a clear error at startup if the variable is unset or the
file is missing, so a misconfiguration fails fast instead of mid-run.

## Running

Activate the venv first if you haven't:

| OS         | Command                                |
|------------|----------------------------------------|
| Windows    | `.\.venv\Scripts\Activate.ps1`         |
| Linux/macOS| `source .venv/bin/activate`            |

### On a local folder of `.ts` videos (recommended for first run)

```bash
python pipelines/run__failure_mode_detection.py --source /path/to/videos --limit 1
```

`--limit 1` keeps the first run short. Drop the flag once you're confident.

### On S3 (optional)

Set `AWS_PROFILE`, `S3_VIDEO_BUCKET`, `S3_VIDEO_PREFIX` in `.env`, then:

```bash
aws sso login --profile $AWS_PROFILE   # if using SSO
python pipelines/run__failure_mode_detection.py --source s3
```

Each video is downloaded to a temp dir, processed, then deleted.

### Just rebuild the timeline graphs

```bash
python pipelines/run__failure_mode_detection.py --graph-only
```

## Outputs

Everything is written under `anomaly_classification/`:

| Path                                    | Contents                                                  |
|-----------------------------------------|-----------------------------------------------------------|
| `fm_failure_modes.csv`                  | One row per video: `Video, fm1, fm2, fm3, fm4`            |
| `fm_failure_modes_timeline.png`         | Static matplotlib timeline                                |
| `fm_failure_modes_timeline.html`        | Interactive plotly timeline (zoom/pan)                    |
| `fm_failure_modes_debug/`               | One annotated debug image per (video, triggered FM)       |
| `fm_no_detection_frames/`               | Mid-video frame from each clean video for spot-checking   |
| `fm_frame_flags/<video>.json`           | Per-frame flag store (run-length encoded), drives graphs  |

Runs are **resumable**: the CSV is the source of truth, and a killed run
restarts exactly where it stopped.

## Repo layout

```
anomaly-classification/
├── pipelines/                    # Pipeline runners + shared helpers
│   ├── run__failure_mode_detection.py        # main entrypoint
│   ├── run__infeed_consistency_pipeline.py   # standalone FM3 helper
│   ├── run__fm3_*                            # eval + timeline diagnostics
│   ├── fm_video_source.py                    # pluggable source interface
│   ├── fm_s3_source.py                       # S3 implementation
│   ├── fm_local_source.py                    # local folder implementation
│   └── fm_state_store.py                     # CSV / flag-store persistence
├── shared_functions/
│   ├── timestamp_utils.py        # filename ↔ datetime helpers
│   └── s3_utils.py               # generic S3 helpers
├── anomaly_classification/
│   └── infeed_consistency_detection.json     # ROI + hue/morph config
├── requirements.txt              # direct pinned deps
├── requirements-lock.txt         # full frozen environment
└── .env.example                  # template, copy to .env and edit
```

## Troubleshooting

| Symptom                                                          | Fix                                                                                    |
|------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| `YOLO_WEIGHTS_PATH is not set`                                   | Edit `.env` and set the absolute path to `best.pt`                                     |
| `YOLO weights not found at …`                                    | Double-check the path; on Windows prefer forward slashes                               |
| `S3 video source requires environment variables`                 | Either fill the three `S3_*` vars in `.env` or use `--source <local-folder>`           |
| `AWS credentials unavailable` / `TokenRetrievalError`            | `aws sso login --profile <your-profile>`                                               |
| `ImportError: DLL load failed` (Windows, opencv)                 | Install the MSVC 2015–2022 redistributable; reinstall via `pip install --force-reinstall opencv-python` |
| `Activate.ps1 cannot be loaded` (Windows)                        | One-time: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then re-activate      |
