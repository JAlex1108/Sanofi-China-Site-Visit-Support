"""S3 video source for the failure-mode detection pipeline.

Lists and downloads ``.ts`` videos from an S3 prefix. Each video is downloaded
to a temp dir, handed to the caller, then deleted (the caller is responsible
for calling :func:`cleanup_local` once it has finished with the file).

This module implements the :class:`pipelines.fm_video_source.VideoSource`
protocol via :class:`S3VideoSource`, so the runner can swap it for a local
source without code changes.

Configuration is read from environment variables:

- ``AWS_PROFILE``       AWS SSO/credentials profile name (required for S3 mode)
- ``S3_VIDEO_BUCKET``   S3 bucket containing the videos (required for S3 mode)
- ``S3_VIDEO_PREFIX``   Key prefix to list (e.g. ``camera-id/``); required
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import boto3
from botocore.exceptions import ClientError, NoCredentialsError, TokenRetrievalError
from dotenv import load_dotenv

from pipelines.fm_video_source import VideoRef

# Pick up env-driven AWS_PROFILE / S3_VIDEO_BUCKET / S3_VIDEO_PREFIX when this
# module is imported before any runner has loaded .env itself.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# AWS / S3 location is environment-driven. The S3 source is optional — if you
# only ever run with ``--source <local-folder>`` you do not need to set these.
AWS_PROFILE = os.environ.get("AWS_PROFILE", "")
S3_BUCKET = os.environ.get("S3_VIDEO_BUCKET", "")
S3_PREFIX = os.environ.get("S3_VIDEO_PREFIX", "")


def _require_s3_config() -> None:
    """Raise a clear error if S3 env vars are missing before any S3 call."""
    missing = [
        name for name, value in (
            ("AWS_PROFILE", AWS_PROFILE),
            ("S3_VIDEO_BUCKET", S3_BUCKET),
            ("S3_VIDEO_PREFIX", S3_PREFIX),
        ) if not value
    ]
    if missing:
        raise RuntimeError(
            "S3 video source requires environment variables: "
            f"{', '.join(missing)}. Either set them (see .env.example) or "
            "run with --source <local-folder> to read videos from disk."
        )


@dataclass(frozen=True)
class S3Video:
    """One video object in S3.

    Retained for back-compat with callers that still import this name. New
    code should use :class:`VideoRef` via the :class:`VideoSource` protocol.
    """

    key: str
    size_bytes: int

    @property
    def filename(self) -> str:
        return Path(self.key).name


def get_s3_client():
    """Create an S3 client using the configured AWS SSO profile.

    Raises a clear error if credentials are missing or the SSO token has
    expired, rather than letting a cryptic botocore error surface mid-run.
    """
    _require_s3_config()
    try:
        session = boto3.Session(profile_name=AWS_PROFILE)
        return session.client("s3")
    except (NoCredentialsError, TokenRetrievalError) as exc:
        raise RuntimeError(
            f"AWS credentials unavailable for profile '{AWS_PROFILE}'. "
            f"Run: aws sso login --profile {AWS_PROFILE}\n  ({exc})"
        ) from exc


def list_camera_videos(s3_client) -> List[S3Video]:
    """List every ``.ts`` video under the camera prefix, sorted by key.

    The keys embed an ISO-ish timestamp, so a plain sort on the key is also a
    chronological sort.
    """
    videos: List[S3Video] = []
    paginator = s3_client.get_paginator("list_objects_v2")
    try:
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PREFIX):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.endswith(".ts"):
                    continue
                videos.append(S3Video(key=key, size_bytes=int(obj["Size"])))
    except (ClientError, NoCredentialsError, TokenRetrievalError) as exc:
        raise RuntimeError(
            f"Failed to list s3://{S3_BUCKET}/{S3_PREFIX}: {exc}\n"
            f"  If this is an expired token, run: "
            f"aws sso login --profile {AWS_PROFILE}"
        ) from exc

    videos.sort(key=lambda v: v.key)
    return videos


def download_video(s3_client, video: S3Video, temp_dir: Path) -> Path:
    """Download one S3 video into ``temp_dir`` and return the local path.

    If the file is already present locally (e.g. an interrupted prior run left
    it behind) it is reused rather than re-downloaded.
    """
    temp_dir.mkdir(parents=True, exist_ok=True)
    local_path = temp_dir / video.filename

    if local_path.exists() and local_path.stat().st_size == video.size_bytes:
        return local_path

    try:
        s3_client.download_file(S3_BUCKET, video.key, str(local_path))
    except (ClientError, NoCredentialsError, TokenRetrievalError) as exc:
        # Leave no partial file behind.
        local_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download {video.key}: {exc}") from exc

    return local_path


def cleanup_local(local_path: Path) -> None:
    """Delete a downloaded video. Safe to call if the file is already gone."""
    local_path.unlink(missing_ok=True)


# =============================================================================
# VideoSource implementation
# =============================================================================


class S3VideoSource:
    """The default S3-backed :class:`VideoSource`.

    Lazily creates one boto3 client and reuses it across the run. Bucket and
    prefix are read from the environment (see module docstring).
    """

    def __init__(self) -> None:
        self._client = None
        self._size_by_key: dict[str, int] = {}

    @property
    def description(self) -> str:
        return f"s3://{S3_BUCKET}/{S3_PREFIX}"

    def _ensure_client(self):
        if self._client is None:
            self._client = get_s3_client()
        return self._client

    def list_videos(self) -> List[VideoRef]:
        client = self._ensure_client()
        s3_videos = list_camera_videos(client)
        # Stash sizes so :meth:`acquire` can skip the download when an
        # interrupted prior run already wrote the full file.
        self._size_by_key = {v.key: v.size_bytes for v in s3_videos}
        return [VideoRef(filename=v.filename, locator=v.key) for v in s3_videos]

    def acquire(self, video: VideoRef, temp_dir: Path) -> Path:
        client = self._ensure_client()
        size = self._size_by_key.get(video.locator, 0)
        s3_video = S3Video(key=video.locator, size_bytes=size)
        return download_video(client, s3_video, temp_dir)

    def release(self, local_path: Path) -> None:
        cleanup_local(local_path)
