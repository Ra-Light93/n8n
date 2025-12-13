import os
import pysrt
import numpy as np
from moviepy import VideoFileClip, TextClip, CompositeVideoClip, ImageClip, VideoClip, vfx
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


# =========================
# CONFIG (ALL VARIABLES HERE)
# =========================

@dataclass(frozen=True)
class Configuration:
    # --- video/subtitle placement ---
    text_canvas_height: int = 260   # fixed subtitle canvas height (px)
    bottom_margin: int = 140        # distance from bottom of video to subtitle canvas (px)

    # --- text style ---
    font_name = "Verdana Bold"      # your chosen font Impact / Verdana Bold
    font_size: int = 70
    border_size: int = 12           # keep small; big values are VERY slow

    # --- extra padding around text (prevents cut-off) ---
    pad_side_extra: int = 60
    pad_top_extra: int = 70
    pad_bottom_extra: int = 70    

    # --- glow / soft shadow behind text (like a dark halo) ---
    glow_enabled: bool = True
    glow_radius: int = 10        # blur radius (bigger = softer / larger glow)
    glow_spread: int = 10         # how far the glow extends before blur
    glow_alpha: int = 255        # 0..255 (opacity of the glow)

    y_position: int = 500


# Backwards-compat alias if you still pass SubtitleConfig somewhere
SubtitleConfig = Configuration


# ---------- 1) WHITE TEXT + BLACK BORDER (+ OPTIONAL SHADOW) ----------

def create_subtitle_clip(text: str, cfg: Configuration) -> ImageClip:
    """Create white subtitle text with smooth black border (+ optional shadow)."""

    # try to load font
    try:
        font = ImageFont.truetype(cfg.font_name, cfg.font_size)
    except Exception as e:
        # IMPORTANT: fail loudly to avoid random metric changes (font fallback causes "jumping")
        raise RuntimeError(
            f"Could not load font '{cfg.font_name}'. Provide a valid font name/path (e.g. '/path/Impact.ttf')."
        ) from e

    # normalize text (avoid bbox surprises)
    text = (text or "").strip()

    # measure text
    tmp = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    draw_tmp = ImageDraw.Draw(tmp)
    bbox = draw_tmp.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    border = int(cfg.border_size)
    pad_side = border * 2 + int(cfg.pad_side_extra)
    pad_top = border * 2 + int(cfg.pad_top_extra)
    pad_bottom = border * 2 + int(cfg.pad_bottom_extra)

    w = tw + pad_side * 2
    h = th + pad_top + pad_bottom
    x = pad_side
    y = pad_top

    # transparent canvas
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))  # type: ignore
    draw = ImageDraw.Draw(img)

    # GLOW (soft dark halo behind text)
    if cfg.glow_enabled:
        glow_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0)) # type: ignore
        glow_draw = ImageDraw.Draw(glow_layer)

        spread = int(cfg.glow_spread)
        for dx in range(-spread, spread + 1):
            for dy in range(-spread, spread + 1):
                glow_draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0, 255))

        glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(int(cfg.glow_radius)))

        # force glow opacity to cfg.glow_alpha
        r, g, b, a = glow_layer.split()
        a = a.point(lambda p: min(255, int(p * (cfg.glow_alpha / 255.0)))) # type:ignore
        glow_layer = Image.merge("RGBA", (r, g, b, a))

        img = Image.alpha_composite(img, glow_layer)
        draw = ImageDraw.Draw(img)  # rebind after composite

    # OUTLINE (round / circular)
    if border > 0:
        r2 = border * border
        for dx in range(-border, border + 1):
            for dy in range(-border, border + 1):
                if dx == 0 and dy == 0:
                    continue
                if (dx * dx + dy * dy) > r2:
                    continue  # outside circle -> skip
                draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0, 255))

    # MAIN TEXT
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))

    # fixed canvas height to avoid vertical shifting
    if img.height < cfg.text_canvas_height:
        canvas_h = cfg.text_canvas_height
        padded = Image.new("RGBA", (img.width, canvas_h), (0, 0, 0, 0))
        offset_y = canvas_h - img.height
        padded.paste(img, (0, offset_y))
        img = padded

    arr = np.array(img)
    return ImageClip(arr, transparent=True)


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
    cfg: Optional[Configuration] = None,
):
    if cfg is None:
        cfg = Configuration()

    # robust path debug
    print("CWD:", os.getcwd())
    print("Input exists:", os.path.exists(input_video_path), input_video_path)
    print("SRT exists:", os.path.exists(srt_path), srt_path)

    if not os.path.exists(input_video_path):
        raise FileNotFoundError(input_video_path)
    if not os.path.exists(srt_path):
        raise FileNotFoundError(srt_path)

    print(f"Loading video: {input_video_path}")
    video = VideoFileClip(input_video_path)
    segments = parse_srt_file(srt_path)

    clips = [video]

    for i, seg in enumerate(segments):
        txt = seg["text"]
        print(f"Subtitle {i+1}/{len(segments)}: {txt[:40]!r}")

        # If you want to avoid multi-line surprises, uncomment:
        # txt = txt.replace("\n", " ")

        txt_clip = create_subtitle_clip(txt, cfg)

        # fixed y position from config
        y = cfg.y_position

        txt_clip = (
            txt_clip
            .with_position(("center", y))
            .with_start(seg["start"])
            .with_duration(seg["duration"])
        )

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
    # IMPORTANT: if you run from inside the `n8n` folder, do NOT prefix with `n8n/`.
    cfg = Configuration()
    input_file = "n8n/Downloads/Sakinah Labs/TestVideo.mp4"
    srt_file = "transcripts/audio_for_transcription.srt"

    outputFileName = cfg.font_name
    out = f"n8n/Testing/videoOuput/{outputFileName}.mp4"



    
    # ensure output dir exists
    out_dir = os.path.dirname(out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    add_subtitles_to_video(
        input_video_path=input_file,
        srt_path=srt_file,
        output_video_path=out,
        preview=True,
        cfg=cfg,
    )
