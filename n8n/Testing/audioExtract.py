#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Extract audio using yt-dlp")
    parser.add_argument("input", help="Video URL or file path")
    parser.add_argument("--output", required=True, help="Output directory for audio file")
    parser.add_argument("--format", default="mp3", help="Audio format (mp3, m4a, wav)")
    parser.add_argument("--quality", default="192", help="Audio quality (kbps)")
    args = parser.parse_args()

    in_path = Path(args.input)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # If input is a local file, use ffmpeg (yt-dlp expects a URL and may reject local paths)
    if in_path.exists():
        out_file = out_dir / f"{in_path.stem}.{args.format}"

        # Simple codec mapping
        if args.format.lower() == "mp3":
            codec = "libmp3lame"
        elif args.format.lower() == "m4a":
            codec = "aac"
        elif args.format.lower() == "wav":
            codec = "pcm_s16le"
        else:
            raise SystemExit(f"Unsupported format: {args.format}")

        cmd = ["ffmpeg", "-y", "-i", str(in_path), "-vn", "-acodec", codec]

        # Bitrate only makes sense for lossy formats
        if args.format.lower() in ("mp3", "m4a"):
            cmd += ["-b:a", f"{args.quality}k"]

        cmd += [str(out_file)]

        subprocess.run(cmd, check=True)
        return

    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format", args.format,
        "--audio-quality", args.quality,
        "-o", str(out_dir / "%(title)s.%(ext)s"),
        args.input
    ]

    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()


# ./n8n/Testing/audioExtract.py n8n/Testing/videoOuput/last/lastV1.mp4 \
#   --output n8n/Testing/videoOuput/last/ \
#   --format mp3 \
#   --quality 192

# ./n8n/Testing/audioExtract.py "n8n/Downloads/Sakinah Labs/here.mp4" \
#   --output n8n/Testing/videoOuput/PROB/ \
#   --format mp3 \
#   --quality 192
