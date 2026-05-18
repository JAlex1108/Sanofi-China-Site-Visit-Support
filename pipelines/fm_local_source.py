"""Local-folder video source for the failure-mode detection pipeline.

Mirrors :class:`pipelines.fm_s3_source.S3VideoSource` but reads ``.ts`` files
from a directory on disk instead of S3. ``acquire`` returns the existing file
path unchanged and ``release`` is a no-op — the user's files are never
deleted or moved.

Selected by passing a folder path as ``--source`` on the runner CLI; see
:func:`pipelines.fm_video_source.make_video_source`.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from pipelines.fm_video_source import VideoRef


class LocalVideoSource:
    """A :class:`VideoSource` rooted at a folder of ``.ts`` files.

    Files are listed non-recursively (the typical layout is one flat folder
    of per-recording videos). Sorting is by filename, which embeds an
    ISO-ish timestamp and therefore also sorts chronologically — matching
    the convention used by :class:`S3VideoSource`.
    """

    def __init__(self, folder: Path) -> None:
        folder = Path(folder).expanduser().resolve()
        if not folder.is_dir():
            raise NotADirectoryError(f"Local video source path is not a folder: {folder}")
        self._folder = folder
        self.description = f"local:{folder}"

    @property
    def folder(self) -> Path:
        return self._folder

    def list_videos(self) -> List[VideoRef]:
        ts_files = sorted(
            p for p in self._folder.iterdir() if p.is_file() and p.suffix == ".ts"
        )
        return [VideoRef(filename=p.name, locator=str(p)) for p in ts_files]

    def acquire(self, video: VideoRef, temp_dir: Path) -> Path:
        # ``temp_dir`` is intentionally ignored: the file is already local,
        # so we hand back its real path. The runner must NOT delete it; that
        # is what :meth:`release` is for (a no-op here).
        path = Path(video.locator)
        if not path.exists():
            raise FileNotFoundError(f"Local video disappeared between list and acquire: {path}")
        return path

    def release(self, local_path: Path) -> None:
        # Never delete user-owned files. Intentional no-op.
        return None
