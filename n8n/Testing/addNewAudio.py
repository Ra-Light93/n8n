import subprocess
import sys


def get_video_duration_seconds(video_path: str) -> float:
    """
    Returns the duration of the input video in seconds (float) using ffprobe.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out)


def replace_audio(video_path: str, audio_path: str, output_path: str) -> None:
    """
    Replaces the original audio of a video with a new audio file.

    Fix: keep full video length even if the new audio is shorter by padding
    the audio with silence (apad) and forcing output duration to video duration.
    """
    dur = get_video_duration_seconds(video_path)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-i", audio_path,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-af", "apad",
        "-t", f"{dur}",
        output_path,
    ]

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python addNewAudio.py <video.mp4> <audio.mp3|wav> <output.mp4>")
        sys.exit(1)

    video = sys.argv[1]
    audio = sys.argv[2]
    output = sys.argv[3]

    replace_audio(video, audio, output)


# Example:
# python addNewAudio.py input.mp4 speech.wav output.mp4