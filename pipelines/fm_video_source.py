"""Pluggable video source for the failure-mode detection pipeline.

The pipeline doesn't care where videos come from. It only needs to:

1. List videos in chronological order (one ``VideoRef`` per video).
2. Get a local ``Path`` for one video so OpenCV can read it.
3. Release that local path once processing is done (delete it if the source
   downloaded it; no-op if it was already local).

Two concrete sources implement this contract:

- :class:`pipelines.fm_s3_source.S3VideoSource` — downloads each ``.ts`` under
  the S3 camera prefix, deletes after processing.
- :class:`pipelines.fm_local_source.LocalVideoSource` — points at a folder of
  ``.ts`` files already on disk; ``release`` is a no-op.

Selection happens at the CLI via :func:`make_video_source`:

    --source s3                       -> S3VideoSource (default S3 layout)
    --source <folder-path>            -> LocalVideoSource(<folder-path>)

The runner stays source-agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Protocol


@dataclass(frozen=True)
class VideoRef:
    """A handle to one video, independent of where it lives.

    ``filename`` is what the pipeline persists in the CSV / flag-store, so it
    must be stable across runs (and identical between S3 and local sources
    pointing at the same file).
    """

    filename: str
    # Free-form identifier the backing source uses internally
    # (S3 key for S3VideoSource, absolute path for LocalVideoSource).
    locator: str


class VideoSource(Protocol):
    """Contract every video source must satisfy.

    A source is the bridge between "the universe of available videos" and "a
    local path I can hand to OpenCV". Implementations may stream from cloud
    storage or just walk a folder; the runner doesn't distinguish.
    """

    description: str

    def list_videos(self) -> List[VideoRef]:
        """Every video this source knows about, in chronological order."""
        ...

    def acquire(self, video: VideoRef, temp_dir: Path) -> Path:
        """Return a local ``Path`` ready for ``cv2.VideoCapture``.

        ``temp_dir`` is offered for sources that need scratch space (S3); a
        source that's already local may ignore it.
        """
        ...

    def release(self, local_path: Path) -> None:
        """Counterpart to :meth:`acquire`.

        S3 deletes the downloaded copy. Local sources leave the user's files
        untouched. Must be safe to call if the file is already gone.
        """
        ...


def make_video_source(spec: str) -> VideoSource:
    """Build a :class:`VideoSource` from a CLI spec.

    ``spec`` values:
        ``"s3"``                  -> default S3 source for camera 40705473.
        ``"<existing folder>"``   -> local source rooted at that folder.

    Raises ``ValueError`` for an unrecognised spec so the runner fails fast
    instead of silently picking the wrong backend.
    """
    # Imports here to avoid a hard boto3 dependency for local-only runs.
    if spec == "s3":
        from pipelines.fm_s3_source import S3VideoSource

        return S3VideoSource()

    candidate = Path(spec).expanduser()
    if candidate.is_dir():
        from pipelines.fm_local_source import LocalVideoSource

        return LocalVideoSource(candidate)

    raise ValueError(
        f"Unrecognised video source spec: {spec!r}. "
        f"Pass 's3' or a path to an existing folder of .ts files."
    )
