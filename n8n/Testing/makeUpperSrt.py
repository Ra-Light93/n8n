#!/usr/bin/env python3
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Convert SRT subtitles to ALL CAPS")
    parser.add_argument("input", help="Input .srt file")
    parser.add_argument("output", help="Output .srt file")
    parser.add_argument(
        "--remove-extra",
        action="store_true",
        help="Remove ',' and '.' from subtitle text lines"
    )
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output)

    with inp.open("r", encoding="utf-8") as f, out.open("w", encoding="utf-8") as o:
        for line in f:
            if "-->" in line or line.strip().isdigit():
                o.write(line)
            else:
                text = line
                if args.remove_extra:
                    text = text.replace(",", "").replace(".", "")
                o.write(text.upper())

if __name__ == "__main__":
    main()

# python srt_to_caps.py input.srt output_caps.srt
# python ./n8n/Testing/makeUpperSrt.py ./n8n/Testing/videoOuput/last/lastV1.srt ./n8n/Testing/videoOuput/last/lastV1Upper.srt --remove-extra
