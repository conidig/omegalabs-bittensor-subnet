import tempfile
from typing import Optional, BinaryIO

import ffmpeg
import yt_dlp


def seconds_to_str(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def clip_video(video_path: str, start_time: int, end_time: int) -> Optional[BinaryIO]:
    temp_fileobj = tempfile.NamedTemporaryFile(suffix=".mp4")
    (
        ffmpeg
        .input(video_path, ss=seconds_to_str(start_time), to=seconds_to_str(end_time))
        .output(temp_fileobj.name, c="copy")  # copy flag prevents decoding and re-encoding
        .overwrite_output()
        .run()
    )
    return temp_fileobj


def download_video(
    video_id: str, start: Optional[int]=None, end: Optional[int]=None
) -> Optional[BinaryIO]:
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    temp_fileobj = tempfile.NamedTemporaryFile(suffix=".mp4")
    ydl_opts = {
        "format": "worst",  # Download the worst quality
        "outtmpl": temp_fileobj.name,  # Set the output template to the temporary file"s name
        "overwrites": True,
    }
    if start and end:
        ydl_opts["download_ranges"] = lambda _, __: [{"start_time": start, "end_time": end}]
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        return temp_fileobj
    except Exception as e:
        print(f"Error downloading video: {e}")
        temp_fileobj.close()
        return None


def copy_audio(video_path: str) -> BinaryIO:
    temp_audiofile = tempfile.NamedTemporaryFile(suffix=".aac")
    (
        ffmpeg
        .input(video_path)
        .output(temp_audiofile.name, vn=True, acodec='copy')
        .overwrite_output()
        .run()
    )
    return temp_audiofile
