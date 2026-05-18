"""Timestamp and frame calculation utilities for video synchronization."""
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple
import re


# Frame rates
FRAME_RATE = 155  # fps - primary camera (01a-020-40705465)
SCREEN_CAMERA_FRAME_RATE = 30  # fps - screen camera (01a-020-40705464)


@dataclass
class FlashTimestamp:
    """Represents a flash event with its absolute timestamp."""

    camera_id: str
    video_filename: str
    frame_number: int
    absolute_time: datetime
    roi_name: str  # "left" or "right"

    def __str__(self) -> str:
        return (
            f"Flash at {self.absolute_time.isoformat()} "
            f"(frame {self.frame_number} in {self.video_filename}, {self.roi_name})"
        )


def parse_video_timestamp(filename: str) -> Optional[datetime]:
    """Extract the start timestamp from a video filename.

    Expected format: ``{camera_id}_{YYYY-MM-DD}_{HH-MM-SS}_{microseconds}.ts``

    Trailing annotation tags are tolerated, so filenames like
    ``{...}_870967__RATE_LIMITED__.ts`` still parse to the correct timestamp.
    The regex anchors on the date+time block and stops at the first non-digit
    after the microseconds, so any free-form suffix the recorder appends
    (``__RATE_LIMITED__``, ``__DROPPED__`` etc.) does not break parsing.

    Args:
        filename: Video filename (with or without path)

    Returns:
        datetime of video start, or None if parsing fails
    """
    basename = Path(filename).stem
    # Anchored on the timestamp block; ``(?=$|\D)`` lets us stop after the
    # microseconds at either end-of-string or any non-digit (e.g. an ``__``
    # tag separator) without consuming the tag itself.
    pattern = r"_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})_(\d+)(?=$|\D)"
    match = re.search(pattern, basename)

    if not match:
        return None

    date_str = match.group(1)
    time_str = match.group(2).replace("-", ":")
    microseconds_str = match.group(3)

    try:
        base_time = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
        microseconds = int(microseconds_str)
        return base_time.replace(microsecond=microseconds)
    except ValueError:
        return None


def frame_to_absolute_time(
    video_start_time: datetime,
    frame_number: int,
    frame_rate: float = FRAME_RATE,
) -> datetime:
    """Convert a frame number to an absolute timestamp.

    Args:
        video_start_time: When the video recording started
        frame_number: Zero-indexed frame number within the video
        frame_rate: Frames per second

    Returns:
        Absolute datetime of the frame
    """
    seconds_offset = frame_number / frame_rate
    return video_start_time + timedelta(seconds=seconds_offset)


def absolute_time_to_frame(
    video_start_time: datetime,
    target_time: datetime,
    frame_rate: float = FRAME_RATE,
) -> int:
    """Convert an absolute timestamp to a frame number.

    Args:
        video_start_time: When the video recording started
        target_time: The time to find the frame for
        frame_rate: Frames per second

    Returns:
        Frame number (zero-indexed) closest to the target time
    """
    delta = target_time - video_start_time
    seconds_offset = delta.total_seconds()
    return int(round(seconds_offset * frame_rate))


def find_video_containing_timestamp(
    videos: List[Tuple[str, datetime, int]],
    target_time: datetime,
    frame_rate: float = FRAME_RATE,
) -> Optional[Tuple[str, datetime, int]]:
    """Find the video that contains a given timestamp.

    Args:
        videos: List of (video_key, start_time, frame_count) tuples
        target_time: The timestamp to find
        frame_rate: Frames per second (default 155fps for primary camera)

    Returns:
        Tuple of (video_key, start_time, frame_count) or None if not found
    """
    for video_key, start_time, frame_count in videos:
        duration = timedelta(seconds=frame_count / frame_rate)
        end_time = start_time + duration

        if start_time <= target_time < end_time:
            return (video_key, start_time, frame_count)

    return None


def calculate_frame_in_video(
    video_start_time: datetime,
    target_time: datetime,
    frame_rate: float = FRAME_RATE,
) -> int:
    """Calculate which frame in a video corresponds to a target time.

    Args:
        video_start_time: When the video started
        target_time: The absolute time to find
        frame_rate: Frames per second

    Returns:
        Frame number (zero-indexed) in the video
    """
    return absolute_time_to_frame(video_start_time, target_time, frame_rate)


def create_flash_timestamps(
    video_filename: str,
    flash_frames: List[int],
    roi_name: str,
) -> List[FlashTimestamp]:
    """Create FlashTimestamp objects for detected flashes.

    Args:
        video_filename: Name of the source video file
        flash_frames: List of frame numbers where flashes were detected
        roi_name: Name of the ROI ("left" or "right")

    Returns:
        List of FlashTimestamp objects
    """
    video_start = parse_video_timestamp(video_filename)
    if video_start is None:
        raise ValueError(f"Could not parse timestamp from filename: {video_filename}")

    basename = Path(video_filename).stem
    camera_match = re.match(r"^([^_]+)", basename)
    camera_id = camera_match.group(1) if camera_match else "unknown"

    timestamps = []
    for frame_num in flash_frames:
        absolute_time = frame_to_absolute_time(video_start, frame_num)
        timestamps.append(
            FlashTimestamp(
                camera_id=camera_id,
                video_filename=video_filename,
                frame_number=frame_num,
                absolute_time=absolute_time,
                roi_name=roi_name,
            )
        )

    return timestamps
