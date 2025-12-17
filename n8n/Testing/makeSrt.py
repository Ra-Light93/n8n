#!/usr/bin/env python3
import argparse
from pathlib import Path
from faster_whisper import WhisperModel

def format_ts(t):
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")

def main():
    parser = argparse.ArgumentParser(description="Audio to word-level SRT (Whisper)")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--sprache", "--sp",
        default="auto",
        help="Input language: auto | de | en | fr | ..."
    )
    parser.add_argument("--model", default="medium", help="tiny | base | small | medium | large")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_srt = out_dir / f"{in_path.stem}.srt"

    print(f"[1/3] Loading model: {args.model}", flush=True)
    model = WhisperModel(args.model)

    print("[2/3] Transcribing (this can take a while)...", flush=True)
    segments_gen, _ = model.transcribe(
        str(in_path),
        language=None if args.sprache == "auto" else args.sprache,
        word_timestamps=True
    )

    # Materialize once so we can compute total work and show 0..100%
    segments = list(segments_gen)
    total_words = sum(len(getattr(seg, "words", []) or []) for seg in segments) or 1

    print(f"[3/3] Writing SRT: {out_srt.name}", flush=True)

    idx = 1
    done_words = 0
    last_pct = -1

    with open(out_srt, "w", encoding="utf-8") as f:
        for seg in segments:
            words = getattr(seg, "words", None) or []
            for w in words:  # type: ignore
                f.write(f"{idx}\n")
                f.write(f"{format_ts(w.start)} --> {format_ts(w.end)}\n")
                f.write(f"{w.word.strip()}\n\n")
                idx += 1

                done_words += 1
                pct = int(done_words * 100 / total_words)
                if pct != last_pct:
                    print(f"\rProgress: {pct:3d}% ({done_words}/{total_words} words)", end="", flush=True)
                    last_pct = pct

    print("\nDone.", flush=True)

if __name__ == "__main__":
    main()



# python n8n/Testing/makeSrt.py n8n/Testing/videoOuput/last/lastV1.mp3 \
#   --output n8n/Testing/videoOuput/last/ \
#   --sp en \
#   --model small

