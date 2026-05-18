"""Manual-review GUI for failure-mode debug images.

Walks every PNG in ``anomaly_classification/fm_failure_modes_debug/`` and lets
the reviewer mark each video as include (keep on timeline) or exclude (drop
from timeline). The decision is per-video — flipping any one of a video's
debug images flips the whole video — and is persisted to the ``include``
column of ``fm_failure_modes.csv`` on every keypress so closing the window
never loses work.

Debug images are named ``<video_stem>__FM<n>__f<frame>.png`` and are emitted
by the pipeline whenever a failure mode triggers for a video. A single video
can have up to 4 images (one per triggered FM); they appear consecutively in
review order, but a single inclusion/exclusion decision applies to all of
them.

Order: chronological by video start time, then FM order within a video.

Keyboard:
  Y / Enter / Right Arrow  - mark current video INCLUDE, advance
  N / Backspace / Down     - mark current video EXCLUDE, advance
  Left Arrow / P           - previous image
  J                        - jump to next un-decided video
  S                        - save now (also auto-saves on every decision)
  Q / Esc                  - quit (state is already saved)

Run::

    python scripts/review_debug_images.py
    python scripts/review_debug_images.py --only-undecided
    python scripts/review_debug_images.py --fm FM1
"""
from __future__ import annotations

import argparse
import re
import sys
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageTk

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.fm_state_store import load_include_flags, set_include_flags
from shared_functions.timestamp_utils import parse_video_timestamp

DEFAULT_DEBUG_DIR = PROJECT_ROOT / "anomaly_classification" / "fm_failure_modes_debug"
DEFAULT_CSV_PATH = PROJECT_ROOT / "anomaly_classification" / "fm_failure_modes.csv"

# Matches ``<video_stem>__FM<n>__f<frame>.png``. The video stem is the same
# value used as the CSV's ``Video`` column (minus the ``.ts`` extension).
DEBUG_NAME_RE = re.compile(r"^(?P<stem>.+)__(?P<fm>FM\d)__f\d+\.png$")


@dataclass(frozen=True)
class DebugImage:
    """One debug image plus the video and FM it corresponds to."""

    path: Path
    video_filename: str  # the CSV key — ``<stem>.ts``
    fm: str


def _discover_images(debug_dir: Path, fm_filter: Optional[str]) -> List[DebugImage]:
    """Return debug images sorted chronologically by video, then FM.

    Files with names not matching the expected pattern are skipped with a
    warning (rather than aborting), so the reviewer can still get through
    everything else if the pipeline ever emits a one-off odd filename.
    """
    if not debug_dir.is_dir():
        raise FileNotFoundError(f"debug image dir does not exist: {debug_dir}")

    images: List[DebugImage] = []
    for p in sorted(debug_dir.glob("*.png")):
        m = DEBUG_NAME_RE.match(p.name)
        if not m:
            print(f"  WARN: skipping unrecognised debug filename: {p.name}", file=sys.stderr)
            continue
        fm = m.group("fm")
        if fm_filter and fm != fm_filter:
            continue
        video = f"{m.group('stem')}.ts"
        images.append(DebugImage(path=p, video_filename=video, fm=fm))

    # Sort key: (video start time, FM order, filename). Videos with an
    # unparseable filename sink to the end but are still reviewable.
    fm_rank = {"FM1": 0, "FM2": 1, "FM3": 2, "FM4": 3}
    far_future = float("inf")

    def key(img: DebugImage) -> Tuple[float, int, str]:
        dt = parse_video_timestamp(img.video_filename)
        t = dt.timestamp() if dt is not None else far_future
        return (t, fm_rank.get(img.fm, 99), img.path.name)

    return sorted(images, key=key)


class ReviewApp:
    """Tk GUI: one (video, FM) image at a time, save-on-keypress.

    State lives in three places:
      - ``self._include``: in-memory ``{video_filename: bool}`` reflecting the
        current CSV state plus any pending edits. The single source of truth.
      - The CSV file: written on every decision via ``set_include_flags``.
        Atomic rewrite (tempfile + os.replace) so a crash never corrupts it.
      - ``self._idx``: the current image cursor; not persisted (you start at
        the first un-decided image on relaunch when ``--only-undecided`` is
        passed; otherwise from the beginning).
    """

    BG = "#1e1e1e"
    FG = "#eaeaea"
    INCLUDE_COLOR = "#2ca02c"
    EXCLUDE_COLOR = "#d62728"
    PENDING_COLOR = "#888888"

    def __init__(
        self,
        images: List[DebugImage],
        csv_path: Path,
        start_at_undecided: bool,
        initially_decided: Dict[str, bool],
    ) -> None:
        if not images:
            raise ValueError("no debug images to review")

        self._images = images
        self._csv_path = csv_path
        # ``_include`` starts as a copy of what's on disk so that videos with no
        # debug images keep their existing decision; the GUI only ever mutates
        # videos for which the reviewer presses a key.
        self._include: Dict[str, bool] = dict(initially_decided)
        # Track which videos the *current session* has touched, so the status
        # bar can distinguish a CSV-default include=1 from an explicit decision.
        self._decided_this_session: set = set()

        # Tk setup.
        self._root = tk.Tk()
        self._root.title("FM debug-image review")
        self._root.configure(bg=self.BG)
        self._root.geometry("1400x900")

        self._image_label = tk.Label(self._root, bg=self.BG)
        self._image_label.pack(fill="both", expand=True, padx=8, pady=8)

        self._status = tk.Label(
            self._root,
            text="",
            bg=self.BG,
            fg=self.FG,
            font=("Consolas", 11),
            anchor="w",
            justify="left",
            padx=12,
            pady=8,
        )
        self._status.pack(fill="x", side="bottom")

        self._help = tk.Label(
            self._root,
            text=(
                "Y/Enter/→ include  |  N/Backspace/↓ exclude  |  "
                "←/P prev  |  J next undecided  |  S save  |  Q/Esc quit"
            ),
            bg="#111",
            fg="#aaa",
            font=("Consolas", 10),
            pady=4,
        )
        self._help.pack(fill="x", side="bottom")

        # The Tk PhotoImage reference must outlive the label, or it's GC'd
        # mid-draw and the canvas goes blank. Keep it as an instance attr.
        self._tk_image: Optional[ImageTk.PhotoImage] = None

        # Key bindings.
        for key in ("y", "Y", "Return", "Right"):
            self._root.bind(f"<{key}>", lambda _e: self._decide(True))
        for key in ("n", "N", "BackSpace", "Down"):
            self._root.bind(f"<{key}>", lambda _e: self._decide(False))
        for key in ("Left", "p", "P"):
            self._root.bind(f"<{key}>", lambda _e: self._step(-1))
        self._root.bind("<j>", lambda _e: self._jump_to_next_undecided())
        self._root.bind("<J>", lambda _e: self._jump_to_next_undecided())
        self._root.bind("<s>", lambda _e: self._save_now())
        self._root.bind("<S>", lambda _e: self._save_now())
        for key in ("q", "Q", "Escape"):
            self._root.bind(f"<{key}>", lambda _e: self._root.destroy())
        # Re-render on window resize so the image scales to fit.
        self._root.bind("<Configure>", self._on_resize)

        self._idx = 0
        if start_at_undecided:
            self._idx = self._next_undecided_from(0, default=0)
        self._render()

    # ---- decision flow ------------------------------------------------------

    def _decide(self, include: bool) -> None:
        img = self._images[self._idx]
        prior = self._include.get(img.video_filename, True)
        self._include[img.video_filename] = include
        self._decided_this_session.add(img.video_filename)
        # Persist only when the value actually changes — otherwise we'd rewrite
        # the whole CSV on every "confirm include" keypress, which is wasteful
        # on a 400+ row file.
        if prior != include:
            try:
                set_include_flags(self._csv_path, {img.video_filename: include})
            except OSError as exc:
                # Surface the error in the status bar; do not silently swallow.
                # The in-memory decision stays so the user can retry with S.
                self._flash_status(f"SAVE FAILED: {exc}", color=self.EXCLUDE_COLOR)
                return
        self._step(1)

    def _step(self, delta: int) -> None:
        new_idx = self._idx + delta
        if 0 <= new_idx < len(self._images):
            self._idx = new_idx
            self._render()
        elif new_idx >= len(self._images):
            self._flash_status("END OF QUEUE — Q to quit", color=self.INCLUDE_COLOR)

    def _next_undecided_from(self, start: int, default: int) -> int:
        for i in range(start, len(self._images)):
            v = self._images[i].video_filename
            if v not in self._decided_this_session:
                return i
        return default

    def _jump_to_next_undecided(self) -> None:
        target = self._next_undecided_from(self._idx + 1, default=self._idx)
        if target == self._idx:
            self._flash_status("no more undecided videos", color=self.PENDING_COLOR)
            return
        self._idx = target
        self._render()

    def _save_now(self) -> None:
        # ``set_include_flags`` only persists rows where the new value differs
        # from disk, so calling it with the whole in-memory map is a no-op
        # when nothing's stale — but it still confirms the CSV matches.
        try:
            changed = set_include_flags(self._csv_path, self._include)
        except OSError as exc:
            self._flash_status(f"SAVE FAILED: {exc}", color=self.EXCLUDE_COLOR)
            return
        self._flash_status(
            f"saved ({changed} row{'s' if changed != 1 else ''} updated)",
            color=self.INCLUDE_COLOR,
        )

    # ---- rendering ----------------------------------------------------------

    def _on_resize(self, event: tk.Event) -> None:
        # Only re-render on the root window resize, not child propagation.
        if event.widget is self._root:
            self._render()

    def _render(self) -> None:
        img = self._images[self._idx]

        # Fit image into the current label area while preserving aspect.
        try:
            pil = Image.open(img.path)
        except OSError as exc:
            self._image_label.configure(image="", text=f"failed to open\n{img.path}\n{exc}", fg=self.EXCLUDE_COLOR)
            return

        max_w = max(self._image_label.winfo_width(), 800)
        max_h = max(self._image_label.winfo_height(), 600)
        pil.thumbnail((max_w, max_h), Image.LANCZOS)
        self._tk_image = ImageTk.PhotoImage(pil)
        self._image_label.configure(image=self._tk_image, text="")

        decision = self._include.get(img.video_filename, True)
        decided = img.video_filename in self._decided_this_session
        if decided:
            tag = "INCLUDE" if decision else "EXCLUDE"
            color = self.INCLUDE_COLOR if decision else self.EXCLUDE_COLOR
        else:
            tag = f"{'INCLUDE' if decision else 'EXCLUDE'} (from CSV, not touched this session)"
            color = self.PENDING_COLOR

        # Count peer images for this video so the reviewer knows how many
        # detections the same decision will cover.
        peers = sum(1 for x in self._images if x.video_filename == img.video_filename)

        self._status.configure(
            text=(
                f"[{self._idx + 1}/{len(self._images)}]  {img.fm}  "
                f"{img.video_filename}  ({peers} debug image(s) for this video)\n"
                f"current decision: {tag}"
            ),
            fg=color,
        )

    def _flash_status(self, message: str, color: str) -> None:
        current = self._status.cget("text")
        self._status.configure(text=f"{message}\n{current.splitlines()[-1] if current else ''}", fg=color)
        # Restore the normal status after 1.5s so the message isn't sticky.
        self._root.after(1500, self._render)

    def run(self) -> None:
        self._root.mainloop()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--debug-dir",
        type=Path,
        default=DEFAULT_DEBUG_DIR,
        help=f"directory of FM debug PNGs (default: {DEFAULT_DEBUG_DIR})",
    )
    p.add_argument(
        "--csv-path",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help=f"fm_failure_modes.csv to update (default: {DEFAULT_CSV_PATH})",
    )
    p.add_argument(
        "--fm",
        type=str,
        choices=("FM1", "FM2", "FM3", "FM4"),
        default=None,
        help="restrict review to one failure mode (e.g. --fm FM1)",
    )
    p.add_argument(
        "--only-undecided",
        action="store_true",
        help="start at the first video that hasn't been decided this session",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    images = _discover_images(args.debug_dir, args.fm)
    if not images:
        print(
            f"No debug images found in {args.debug_dir}"
            + (f" for {args.fm}" if args.fm else ""),
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loaded {len(images)} debug image(s) from {args.debug_dir}")
    unique_videos = {img.video_filename for img in images}
    print(f"Covering {len(unique_videos)} unique video(s)")

    existing = load_include_flags(args.csv_path)
    print(
        f"CSV has {len(existing)} row(s); "
        f"{sum(1 for v in existing.values() if not v)} currently excluded"
    )

    app = ReviewApp(
        images=images,
        csv_path=args.csv_path,
        start_at_undecided=args.only_undecided,
        initially_decided=existing,
    )
    app.run()


if __name__ == "__main__":
    main()
