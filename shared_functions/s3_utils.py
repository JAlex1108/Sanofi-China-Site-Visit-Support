"""S3 utilities for video downloading and listing."""
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Union

import boto3
from botocore.exceptions import ClientError

from .timestamp_utils import FRAME_RATE, parse_video_timestamp


def get_s3_client(aws_profile: str):
    """Create S3 client with the specified AWS profile.

    Args:
        aws_profile: AWS profile name from ~/.aws/credentials

    Returns:
        boto3 S3 client configured with the specified profile
    """
    session = boto3.Session(profile_name=aws_profile)
    return session.client("s3")


def parse_video_datetime(s3_key: str) -> Optional[datetime]:
    """Extract datetime from S3 key.

    Expected format: prefix/prefix_YYYY-MM-DD_HH-MM-SS_microseconds.ts

    Args:
        s3_key: S3 object key

    Returns:
        datetime of video start, or None if parsing fails
    """
    filename = Path(s3_key).name
    return parse_video_timestamp(filename)


def _parse_datetime_filter(
    date_input: Optional[Union[str, datetime]], is_end: bool = False
) -> Optional[datetime]:
    """Parse a date/datetime string or datetime object into a datetime object.

    Args:
        date_input: datetime object, or string in YYYY-MM-DD or YYYY-MM-DD HH:MM:SS format
        is_end: If True and only date string provided, set time to 23:59:59

    Returns:
        datetime object or None if date_input is None
    """
    if date_input is None:
        return None

    # If already a datetime, strip timezone info for naive comparison
    # (video timestamps from filenames are naive)
    if isinstance(date_input, datetime):
        return date_input.replace(tzinfo=None)

    # Try datetime format first (YYYY-MM-DD HH:MM:SS)
    try:
        return datetime.strptime(date_input, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        pass

    # Fall back to date-only format
    try:
        dt = datetime.strptime(date_input, "%Y-%m-%d")
        if is_end:
            dt = dt.replace(hour=23, minute=59, second=59)
        return dt
    except ValueError:
        return None


def list_videos_for_camera(
    s3_client,
    bucket: str,
    camera_id: str,
    start_datetime: Optional[Union[str, datetime]] = None,
    end_datetime: Optional[Union[str, datetime]] = None,
) -> List[str]:
    """List all video files for a specific camera.

    Args:
        s3_client: boto3 S3 client
        bucket: S3 bucket name
        camera_id: Camera identifier (used as S3 prefix)
        start_datetime: Optional start filter (datetime object or string YYYY-MM-DD [HH:MM:SS])
        end_datetime: Optional end filter (datetime object or string YYYY-MM-DD [HH:MM:SS])

    Returns:
        List of S3 keys for matching videos
    """
    start_dt = _parse_datetime_filter(start_datetime, is_end=False)
    end_dt = _parse_datetime_filter(end_datetime, is_end=True)

    matching_keys: List[str] = []
    paginator = s3_client.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket, Prefix=camera_id):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".ts"):
                continue

            video_date = parse_video_datetime(key)
            if video_date is None:
                continue

            if start_dt and video_date < start_dt:
                continue
            if end_dt and video_date > end_dt:
                continue

            matching_keys.append(key)

    matching_keys.sort()
    return matching_keys


def get_video_metadata(
    s3_client,
    bucket: str,
    camera_id: str,
    start_datetime: Optional[Union[str, datetime]] = None,
    end_datetime: Optional[Union[str, datetime]] = None,
    frame_rate: float = FRAME_RATE,
) -> List[Tuple[str, datetime, int]]:
    """Get video metadata including estimated frame counts.

    Note: Frame count is estimated based on video duration from filename pattern.
    For precise frame counts, videos need to be downloaded and inspected.

    Args:
        s3_client: boto3 S3 client
        bucket: S3 bucket name
        camera_id: Camera identifier
        start_datetime: Optional start filter (datetime object or string)
        end_datetime: Optional end filter (datetime object or string)
        frame_rate: Frames per second (default 155fps for primary camera)

    Returns:
        List of (s3_key, start_time, estimated_frame_count) tuples
    """
    keys = list_videos_for_camera(s3_client, bucket, camera_id, start_datetime, end_datetime)

    metadata: List[Tuple[str, datetime, int]] = []

    for i, key in enumerate(keys):
        start_time = parse_video_datetime(key)
        if start_time is None:
            continue

        # Estimate frame count: assume ~1 minute per video segment
        # This is a rough estimate; actual count obtained after download
        estimated_frames = int(60 * frame_rate)

        # If we have a next video, calculate actual duration
        if i + 1 < len(keys):
            next_time = parse_video_datetime(keys[i + 1])
            if next_time:
                duration_seconds = (next_time - start_time).total_seconds()
                if 0 < duration_seconds < 300:  # Sanity check: under 5 min
                    estimated_frames = int(duration_seconds * frame_rate)

        metadata.append((key, start_time, estimated_frames))

    return metadata


def download_video(
    s3_client,
    bucket: str,
    s3_key: str,
    temp_dir: Path,
) -> Path:
    """Download a single video from S3.

    Args:
        s3_client: boto3 S3 client
        bucket: S3 bucket name
        s3_key: S3 object key
        temp_dir: Local directory to save the file

    Returns:
        Path to downloaded file
    """
    temp_dir.mkdir(parents=True, exist_ok=True)

    filename = Path(s3_key).name
    local_path = temp_dir / filename

    if local_path.exists():
        print(f"  Already downloaded: {filename}")
        return local_path

    print(f"  Downloading: {filename}")
    try:
        s3_client.download_file(bucket, s3_key, str(local_path))
    except ClientError as e:
        raise RuntimeError(f"Failed to download {s3_key}: {e}") from e

    return local_path


def download_videos_in_date_range(
    s3_client,
    bucket: str,
    camera_id: str,
    start_datetime: Union[str, datetime],
    end_datetime: Union[str, datetime],
    temp_dir: Path,
    max_videos: int = 0,
) -> List[Path]:
    """Download videos from S3 within the specified date/time range.

    Args:
        s3_client: boto3 S3 client
        bucket: S3 bucket name
        camera_id: Camera identifier
        start_datetime: Start datetime (datetime object or string YYYY-MM-DD [HH:MM:SS])
        end_datetime: End datetime (datetime object or string YYYY-MM-DD [HH:MM:SS])
        temp_dir: Local directory for downloads
        max_videos: Maximum videos to download (0 = unlimited)

    Returns:
        List of paths to downloaded video files
    """
    print(f"Listing videos for {camera_id} from {start_datetime} to {end_datetime}...")
    keys = list_videos_for_camera(s3_client, bucket, camera_id, start_datetime, end_datetime)

    if max_videos > 0:
        keys = keys[:max_videos]

    print(f"Found {len(keys)} video(s) to process")

    downloaded: List[Path] = []
    for key in keys:
        local_path = download_video(s3_client, bucket, key, temp_dir)
        downloaded.append(local_path)

    return downloaded
