import os
import pysrt
import numpy as np
from moviepy import VideoFileClip, TextClip, CompositeVideoClip, ImageClip, VideoClip, vfx
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import argparse


# =========================
# CONFIG (ALL VARIABLES HERE)
# =========================

@dataclass(frozen=True)
class Configuration:
    # rectangle to cover old text (ONLY y + size adjustable)
    cover_y: int = 1500
    cover_w: int = 840
    cover_h: int = 260

    # horizontal anchor (fixed): exact center of the video
    # 0.5 means: center of overlay at 50% of video width
    anchor_x_ratio: float = 0.5

    # cover color + opacity
    cover_color: Tuple[int, int, int] = (0, 0, 0)   # RGB
    cover_opacity: int = 230                         # 0..255

    # optional: soften fill slightly (subtle “matte” look)
    blur_enabled: bool = False
    blur_radius: int = 2

    # outside shadow (looks clean + professional)
    shadow_enabled: bool = True
    shadow_color: Tuple[int, int, int] = (0, 0, 0)   # RGB
    shadow_opacity: int = 160                        # 0..255
    shadow_blur: int = 18                            # px
    shadow_spread: int = 6                           # px (bigger shadow)
    shadow_offset: Tuple[int, int] = (0, 8)          # (dx, dy) in px

    # time range (set to full video by default)
    cover_start: float = 0.0
    cover_end: Optional[float] = None   # None => till end


# =========================
# COVER CLIP (RECTANGLE + OUTSIDE SHADOW)
# =========================

def create_shadowed_cover_clip(video_w: int, video_h: int, cfg: Configuration) -> ImageClip:
    """
    Creates an RGBA ImageClip that covers old text with a clean rectangle and an outside shadow.
    X is fixed (perfectly centered). Only y/width/height are configurable.
    """
    w = int(cfg.cover_w)
    h = int(cfg.cover_h)
    y = int(cfg.cover_y)

    # Fixed x: center at anchor_x_ratio of the video width (right-half center)
    cx = int(float(video_w) * float(cfg.anchor_x_ratio))
    x = int(cx - (w / 2))

    # Clamp to video bounds
    x = max(0, min(x, max(0, video_w - 1)))
    y = max(0, min(y, max(0, video_h - 1)))
    w = max(1, min(w, max(1, video_w - x)))
    h = max(1, min(h, max(1, video_h - y)))

    # Shadow padding
    sb = max(0, int(cfg.shadow_blur))
    ss = max(0, int(cfg.shadow_spread))
    dx, dy = (int(cfg.shadow_offset[0]), int(cfg.shadow_offset[1]))

    pad = sb + ss + max(abs(dx), abs(dy)) + 2

    canvas_w = w + pad * 2
    canvas_h = h + pad * 2

    # Base transparent canvas
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    rect_left = pad
    rect_top = pad
    rect_right = pad + w
    rect_bottom = pad + h

    # --- shadow (outside) ---
    if cfg.shadow_enabled and (sb > 0 or ss > 0 or dx != 0 or dy != 0):
        sr, sg, sbc = cfg.shadow_color
        sa = int(max(0, min(255, cfg.shadow_opacity)))

        shadow_layer = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow_layer)

        # spread: draw a slightly larger rect
        spread_left = rect_left - ss + dx
        spread_top = rect_top - ss + dy
        spread_right = rect_right + ss + dx
        spread_bottom = rect_bottom + ss + dy

        shadow_draw.rectangle(
            [spread_left, spread_top, spread_right, spread_bottom],
            fill=(sr, sg, sbc, sa),
        )

        if sb > 0:
            shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(sb))

        canvas = Image.alpha_composite(canvas, shadow_layer)
        draw = ImageDraw.Draw(canvas)  # rebind

    # --- cover rectangle ---
    r, g, b = cfg.cover_color
    a = int(max(0, min(255, cfg.cover_opacity)))
    draw.rectangle([rect_left, rect_top, rect_right, rect_bottom], fill=(r, g, b, a))

    # Optional: subtle blur on the rectangle fill (usually keep very small)
    if cfg.blur_enabled and int(cfg.blur_radius) > 0:
        rect = canvas.crop((rect_left, rect_top, rect_right, rect_bottom))
        rect = rect.filter(ImageFilter.GaussianBlur(int(cfg.blur_radius)))
        canvas.paste(rect, (rect_left, rect_top), rect)

    arr = np.array(canvas)

    # Important: position the whole shadowed canvas so that the RECT (not the shadow) aligns to (x, y)
    clip = ImageClip(arr, transparent=True).with_position((x - pad, y - pad))
    return clip


# =========================
# MAIN (UPDATED)
# =========================

def cover_old_text(
    input_video_path: str,
    output_video_path: str,
    cfg: Optional[Configuration] = None,
):
    if cfg is None:
        cfg = Configuration()

    print("CWD:", os.getcwd())
    print("Input exists:", os.path.exists(input_video_path), input_video_path)
    if not os.path.exists(input_video_path):
        raise FileNotFoundError(input_video_path)

    video = VideoFileClip(input_video_path)

    print("Creating shadowed rectangle cover...")
    cover = create_shadowed_cover_clip(video.w, video.h, cfg)

    # set timing
    start = float(cfg.cover_start)
    end = cfg.cover_end if cfg.cover_end is not None else float(video.duration)
    dur = max(0.0, float(end) - float(start))

    cover = cover.with_duration(dur)
    cover = cover.with_start(start)

    final = CompositeVideoClip([video, cover])

    out_dir = os.path.dirname(output_video_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print("Saving:", output_video_path)
    final.write_videofile(
        output_video_path,
        codec="libx264",
        fps=video.fps,
        preset="medium",
        audio_codec="aac",
        threads=8,
    )

    video.close()
    final.close()
    print("Done.")
    return output_video_path


# =========================
# TEST RUN (UPDATED)
# =========================

if __name__ == "__main__":
    input_file = "n8n/Downloads/Sakinah Labs/TestVideo.mp4"
    output_file = "n8n/Testing/videoOuput/covered_old_text.mp4"

    cfg = Configuration(
        cover_y=1500,
        cover_w=920,
        cover_h=280,
        anchor_x_ratio=0.5,            # fixed exact center

        cover_color=(0, 0, 0),
        cover_opacity=235,

        blur_enabled=False,
        blur_radius=2,

        shadow_enabled=True,
        shadow_color=(0, 0, 0),
        shadow_opacity=170,
        shadow_blur=20,
        shadow_spread=8,
        shadow_offset=(0, 10),

        cover_start=0.0,
        cover_end=None,
    )

    cover_old_text(input_file, output_file, cfg)