# Install required packages:
# pip install moviepy pysrt numpy opencv-python pillow


import os
import pysrt
import numpy as np
from moviepy import VideoFileClip, TextClip, CompositeVideoClip, ImageClip, VideoClip, vfx
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# Fixed subtitle canvas + bottom margin to keep baseline stable
TEXT_CANVAS_HEIGHT = 260  # height (in px) of the subtitle image for all lines
BOTTOM_MARGIN = 140       # distance from bottom of video to the subtitle canvas (adjust as you like)


@dataclass
class SubtitleConfig:
    font_name : str = "SF-Compact-Display-Heavy"
    border_size = 2
    font_size = 110
    opacity = 180

    # old thik
    # border_size = 2
    # font_size = 110
    # opacity = 180

# ---------- 1) WHITE TEXT + BLACK BORDER (ONE FUNCTION) ----------

def create_subtitle_clip(text: str, config: SubtitleConfig) -> ImageClip:
    """Create white subtitle text with thick smooth black border."""

    # debug: show which font is currently used
    print(f"[create_subtitle_clip] Using font: {config.font_name}")

    # try to load font
    try:
        font = ImageFont.truetype(config.font_name, config.font_size)
    except Exception:
        font = ImageFont.load_default()

    # measure text
    tmp = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    draw_tmp = ImageDraw.Draw(tmp)
    bbox = draw_tmp.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    border = config.border_size
    pad_side = border * 2 + 8       # left/right
    pad_top = border * 2 + 8        # top
    pad_bottom = border * 2 + 24    # bottom (extra space to prevent cut-off)

    w = tw + pad_side * 2
    h = th + pad_top + pad_bottom
    x = pad_side
    y = pad_top

    # transparent canvas
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # OUTLINE (thick smooth black border)
    for dx in range(-border, border + 1):
        for dy in range(-border, border + 1):
            if dx == 0 and dy == 0:
                continue
            draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0, 255))

    # MAIN TEXT (pure white)
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))

    # ensure all subtitle images have the same height to avoid vertical shifting
    if img.height < TEXT_CANVAS_HEIGHT:
        canvas_h = TEXT_CANVAS_HEIGHT
        padded = Image.new("RGBA", (img.width, canvas_h), (0, 0, 0, 0))
        # place the text image at the BOTTOM of the fixed canvas to keep baseline stable
        offset_y = canvas_h - img.height
        padded.paste(img, (0, offset_y))
        img = padded

    # convert to clip
    arr = np.array(img)
    return ImageClip(arr, transparent=True)

# ---------- 2) SIMPLE BLUR BAR (ONE FUNCTION) ----------

def create_blur_bar(
    frame_size: Tuple[int, int],
    bar_height: int = 160,
    blur_radius: int = 15,
    opacity: int = 160
) -> ImageClip:
    """
    Creates a blurred dark bar at the bottom of the frame
    to make subtitles easier to read.
    """
    w, h = frame_size

    img = Image.new("RGBA", (w, bar_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # semi-transparent black rectangle
    draw.rectangle(
        [0, 0, w, bar_height],
        fill=(0, 0, 0, opacity)
    )

    # blur the bar
    img = img.filter(ImageFilter.GaussianBlur(blur_radius))

    arr = np.array(img)
    clip = ImageClip(arr, transparent=True)
    return clip


# ---------- SRT PARSER ----------

def parse_srt_file(srt_path: str) -> List[Dict]:
    subs = pysrt.open(srt_path)
    out = []
    for s in subs:
        start = s.start.ordinal / 1000.0
        end = s.end.ordinal / 1000.0
        out.append(
            {
                "text": s.text,
                "start": start,
                "end": end,
                "duration": end - start,
            }
        )
    return out


# ---------- MAIN: ADD SUBTITLES TO VIDEO ----------

def add_subtitles_to_video(
    input_video_path: str,
    srt_path: str,
    output_video_path: str,
    preview: bool = False,
    config: Optional[SubtitleConfig] = None,
):
    print(f"Loading video: {input_video_path}")
    video = VideoFileClip(input_video_path)
    segments = parse_srt_file(srt_path)

    if config is None:
        config = SubtitleConfig()

    clips = [video]

    for i, seg in enumerate(segments):
        print(f"Subtitle {i+1}/{len(segments)}: {seg['text'][:40]!r}")

        # 1) create text clip
        txt_clip = create_subtitle_clip(seg["text"], config)

        # fixed baseline relative to bottom: use constant canvas height + bottom margin
        y = video.h - BOTTOM_MARGIN - TEXT_CANVAS_HEIGHT

        txt_clip = (
            txt_clip
            .with_position(("center", y))
            .with_start(seg["start"])
            .with_duration(seg["duration"])
        )

        # clips.append(blur_clip)
        clips.append(txt_clip)

    print("Compositing...")
    final = CompositeVideoClip(clips)

    print(f"Saving: {output_video_path}")
    final.write_videofile(
        output_video_path,
        codec="libx264",
        fps=video.fps,
        preset="fast" if preview else "medium",
        audio_codec="aac",
        threads=4 if preview else 8,
    )

    video.close()
    final.close()
    print("Done.")
    return output_video_path


# ---------- EXAMPLE USAGE ----------

if __name__ == "__main__":
    input_file = "n8n/Downloads/Sakinah Labs/Test2.mp4"
    srt_file = "transcripts/audio_for_transcription.srt"

    test_fonts = [
        "Impact",
        "SF-Pro-Display-Bold",
        "SF-Pro-Display-Heavy",
        "SF-Pro-Display-Semibold",
        "SF-Pro-Text-Bold",
        "SF-Pro-Text-Heavy",
        "SF-Compact-Display-Heavy",
        "SF-Compact-Display-Bold",
        "Avenir Next Heavy",
        "Avenir Next Condensed Heavy",
        "Helvetica Neue Bold",
        "Helvetica Neue Condensed Bold",
        "Futura Bold",
        "Futura Condensed ExtraBold",
        "SF-Pro-Rounded-Heavy",
        "SF-Pro-Rounded-Bold"
    ]

    for font in test_fonts:
        cfg = SubtitleConfig(font_name=font)
        outfile = f"n8n/Results/fonts/subtest_{font.replace(' ', '_')}.mp4"
        print(f"\n=== Rendering with font: {font} ===")
        add_subtitles_to_video(
            input_video_path=input_file,
            srt_path=srt_file,
            output_video_path=outfile,
            preview=True,
            config=cfg,
        )