"""Shared functions for video analysis pipelines."""

from .timestamp_utils import (
    FRAME_RATE,
    SCREEN_CAMERA_FRAME_RATE,
    FlashTimestamp,
    parse_video_timestamp,
    frame_to_absolute_time,
    absolute_time_to_frame,
    find_video_containing_timestamp,
    calculate_frame_in_video,
    create_flash_timestamps,
)
from .s3_utils import (
    get_s3_client,
    list_videos_for_camera,
    download_video,
    download_videos_in_date_range,
    get_video_metadata,
    parse_video_datetime,
)
