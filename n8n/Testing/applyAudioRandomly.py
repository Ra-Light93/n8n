#!/usr/bin/env python3

import argparse
import os
import random
import subprocess
from pathlib import Path


def ffprobe_duration_seconds(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out)


def video_has_audio(path: Path) -> bool:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        str(path),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return bool(out)


def pick_random_audio_file(folder: Path, exts: set[str]) -> Path:
    files = [
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower().lstrip(".") in exts
    ]
    if not files:
        raise FileNotFoundError(f"No audio files found in: {folder}")
    return random.choice(files)


def add_background_audio(
    category: str,
    target_video: Path,
    out_dir: Path,
    out_name: str,
    library_dir: Path,
    bg_volume: float,
    exts: set[str],
) -> Path:
    if not target_video.exists():
        raise FileNotFoundError(f"Video not found: {target_video}")

    category_folder = library_dir / category
    if not category_folder.exists() or not category_folder.is_dir():
        raise FileNotFoundError(f"Category folder not found: {category_folder}")

    video_len = ffprobe_duration_seconds(target_video)
    audio_file = pick_random_audio_file(category_folder, exts)

    audio_len = ffprobe_duration_seconds(audio_file)
    if audio_len <= 0:
        raise RuntimeError(f"Could not read audio duration: {audio_file}")

    seg = max(0.001, float(video_len))
    if audio_len <= seg:
        start = 0.0
    else:
        start = random.uniform(0.0, audio_len - seg)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{out_name}.mp4"

    has_a = video_has_audio(target_video)

    # Build filter_complex
    # - Trim background audio to video length
    # - Reduce bg volume to ~30-40% (default 0.35)
    if has_a:
        filter_complex = (
            f"[1:a]atrim=start={start:.3f}:duration={seg:.3f},asetpts=PTS-STARTPTS,"
            f"loudnorm=I=-20:TP=-1.5:LRA=11,volume={bg_volume}[bg];"
            f"[0:a]asetpts=PTS-STARTPTS[orig];"
            f"[orig][bg]amix=inputs=2:weights='1 {bg_volume}':normalize=0:dropout_transition=2[mix]"
        )
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(target_video),
            "-i",
            str(audio_file),
            "-filter_complex",
            filter_complex,
            "-map",
            "0:v:0",
            "-map",
            "[mix]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            str(out_path),
        ]
    else:
        filter_complex = (
            f"[1:a]atrim=start={start:.3f}:duration={seg:.3f},asetpts=PTS-STARTPTS,"
            f"loudnorm=I=-20:TP=-1.5:LRA=11,volume={bg_volume}[bg]"
        )
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(target_video),
            "-i",
            str(audio_file),
            "-filter_complex",
            filter_complex,
            "-map",
            "0:v:0",
            "-map",
            "[bg]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            str(out_path),
        ]

    subprocess.run(cmd, check=True)

    print("Category:", category)
    print("Selected audio:", audio_file)
    print("BG volume:", bg_volume)
    print("Saved video:", out_path)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pick random category audio, lower volume, and add it as background audio to a target video."
    )

    # Requested positional args:
    parser.add_argument("categoryname", help="Folder/category name (e.g., Motivation)")
    parser.add_argument("targetvideo", help="Path to input video")
    parser.add_argument("NewVideooutputPath", help="Output directory")
    parser.add_argument("NewVideoFileName", help="Output filename without extension")

    # Optional settings
    parser.add_argument(
        "--library",
        default=os.getenv("AUDIO_LIBRARY_DIR", ""),
        help="Base library path X that contains category folders (or set AUDIO_LIBRARY_DIR).",
    )
    parser.add_argument(
        "--volume",
        type=float,
        default=0.25,
        help="Background audio volume (0.0-1.0). Default 0.25 (~25%).",
    )
    parser.add_argument(
        "--ext",
        default="mp3,wav,m4a,aac,flac,ogg",
        help="Comma-separated allowed audio extensions.",
    )

    args = parser.parse_args()

    library = Path(args.library).expanduser() if args.library else None
    if not library or not str(library).strip():
        print("Error: Missing audio library path X. Provide --library or set AUDIO_LIBRARY_DIR.")
        return 2

    exts = {e.strip().lower() for e in args.ext.split(",") if e.strip()}

    try:
        add_background_audio(
            category=args.categoryname,
            target_video=Path(args.targetvideo).expanduser(),
            out_dir=Path(args.NewVideooutputPath).expanduser(),
            out_name=args.NewVideoFileName,
            library_dir=library,
            bg_volume=float(args.volume),
            exts=exts,
        )
        return 0
    except subprocess.CalledProcessError as e:
        print("ffmpeg failed:", e)
        return 3
    except Exception as e:
        print("Error:", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


# python n8n/Testing/applyAudioRandomly.py \
# Motivation \
# n8n/Testing/videoOuput/last/2ndDone.mp4 \
# n8n/Testing/videoOuput/PROB \
# works \
# --library ./Audios  \
# --volume 0.5