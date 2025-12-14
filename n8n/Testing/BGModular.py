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
    
    # rounded corners radius (NEW)
    corner_radius: int = 20

    # anti-aliasing quality for rounded corners (1 = off, 2..4 = smoother)
    cover_aa: int = 2

    # gamma shaping for the cover alpha edge (1.0 = linear)
    cover_alpha_gamma: float = 1.0
    
    # horizontal anchor (fixed): exact center of the video
    # 0.5 means: center of overlay at 50% of video width
    anchor_x_ratio: float = 0.5

    # cover color + opacity
    cover_color: Tuple[int, int, int] = (0, 0, 0)   # RGB
    cover_opacity: int = 230                         # 0..255

    # optional: soften fill slightly (subtle "matte" look)
    blur_enabled: bool = False
    blur_radius: int = 2
    blur_padding: int = 6            # extra padding so blur is not clipped

    # outside shadow (looks clean + professional)
    shadow_enabled: bool = True
    shadow_color: Tuple[int, int, int] = (0, 0, 0)   # RGB
    shadow_opacity: int = 160                        # 0..255
    shadow_blur: int = 18                            # px
    shadow_spread: int = 6                           # px (bigger shadow)
    shadow_offset: Tuple[int, int] = (0, 0)          # (dx, dy) in px (symmetrical shadow)

    # shadow rendering fine controls (NEW)
    shadow_aa: int = 2                 # supersampling for smoother shadow edges (1=off)
    shadow_blur_enabled: bool = True   # allow disabling blur via config
    shadow_radius_add: int = 0         # extra radius added to shadow (in addition to spread)
    shadow_alpha_gamma: float = 1.0    # 1.0=linear, >1 harder edge, <1 softer falloff
    shadow_pad_extra: int = 0          # extra padding to prevent any clipping
    # how much empty margin to reserve for the blur to fully fade out (multiplier of shadow_blur)
    # PIL's GaussianBlur radius acts like sigma; ~3*sigma is a good fade-to-zero margin.
    shadow_blur_margin_mult: float = 3.0

    # time range (set to full video by default)
    cover_start: float = 0.0
    cover_end: Optional[float] = None   # None => till end


# =========================
# COVER CLIP (RECTANGLE WITH ROUNDED CORNERS + OUTSIDE SHADOW)
# =========================

def create_rounded_rectangle(width: int, height: int, radius: int, color: Tuple[int, int, int, int]) -> Image.Image:
    """Create a rounded rectangle image with specified radius."""
    rect = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(rect)
    draw.rounded_rectangle(
        [(0, 0), (width - 1, height - 1)],
        radius=radius,
        fill=color
    )
    return rect

# --- NEW: create an anti-aliased mask for a rounded rectangle (L-mode, 0..255) ---
def create_rounded_mask(width: int, height: int, radius: int, aa: int = 2) -> Image.Image:
    """
    Create an anti-aliased L-mode mask (0..255) for a rounded rectangle.
    Uses supersampling (aa) then downsamples for smoother edges.
    """
    width = max(1, int(width))
    height = max(1, int(height))
    radius = max(0, int(radius))
    aa = max(1, int(aa))

    W, H = width * aa, height * aa
    r = radius * aa

    mask = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle([(0, 0), (W - 1, H - 1)], radius=r, fill=255)

    if aa > 1:
        mask = mask.resize((width, height), resample=Image.LANCZOS) # type:ignore
    return mask

# --- NEW: create an anti-aliased mask for a rounded rectangle, drawn inset from the edges ---
def create_inset_rounded_mask(width: int, height: int, radius: int, inset: int, aa: int = 2) -> Image.Image:
    """
    Create an anti-aliased L-mode mask for a rounded rectangle, drawn inset from the edges.
    This is useful for shadows: blur needs empty margin around the shape so it won't be clipped.
    """
    width = max(1, int(width))
    height = max(1, int(height))
    radius = max(0, int(radius))
    inset = max(0, int(inset))
    aa = max(1, int(aa))

    W, H = width * aa, height * aa
    r = radius * aa
    ins = inset * aa

    mask = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask)

    # Draw the rounded rect with an inset border so blur can expand into the margin.
    left = ins
    top = ins
    right = W - 1 - ins
    bottom = H - 1 - ins
    if right <= left or bottom <= top:
        # fallback: no inset possible
        draw.rounded_rectangle([(0, 0), (W - 1, H - 1)], radius=r, fill=255)
    else:
        draw.rounded_rectangle([(left, top), (right, bottom)], radius=r, fill=255)

    if aa > 1:
        mask = mask.resize((width, height), resample=Image.LANCZOS)  # type:ignore
    return mask


def create_shadowed_cover_clip(video_w: int, video_h: int, cfg: Configuration) -> ImageClip:
    """
    Creates an RGBA ImageClip that covers old text with a clean rectangle with rounded corners 
    and an outside shadow. X is fixed (perfectly centered). Only y/width/height are configurable.
    """
    w = int(cfg.cover_w)
    h = int(cfg.cover_h)
    y = int(cfg.cover_y)
    radius = int(cfg.corner_radius)

    cover_aa = max(1, int(getattr(cfg, "cover_aa", 2)))
    cover_alpha_gamma = float(getattr(cfg, "cover_alpha_gamma", 1.0))

    shadow_aa = max(1, int(getattr(cfg, "shadow_aa", 2)))
    shadow_blur_enabled = bool(getattr(cfg, "shadow_blur_enabled", True))
    shadow_radius_add = int(getattr(cfg, "shadow_radius_add", 0))
    shadow_alpha_gamma = float(getattr(cfg, "shadow_alpha_gamma", 1.0))
    shadow_pad_extra = int(getattr(cfg, "shadow_pad_extra", 0))
    shadow_blur_margin_mult = float(getattr(cfg, "shadow_blur_margin_mult", 3.0))
    
    # Ensure radius is not too large for the rectangle
    max_radius = min(w, h) // 2
    radius = min(radius, max_radius)

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

    # Reserve enough empty space so the blurred shadow alpha can fade to ~0 before the image border.
    blur_margin_est = 0
    if shadow_blur_enabled and sb > 0:
        m = shadow_blur_margin_mult if shadow_blur_margin_mult > 0 else 3.0
        blur_margin_est = max(sb, int(sb * m) + 1)

    pad = ss + blur_margin_est + max(abs(dx), abs(dy)) + int(cfg.blur_padding) + 2 + max(0, shadow_pad_extra)

    canvas_w = w + pad * 2
    canvas_h = h + pad * 2

    # Base transparent canvas
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    
    # --- shadow (outside) with rounded corners (mask-based => no hard square edges) ---
    if cfg.shadow_enabled and int(cfg.shadow_opacity) > 0 and (sb > 0 or ss > 0 or dx != 0 or dy != 0):
        sr, sg, sbc = cfg.shadow_color
        sa = int(max(0, min(255, cfg.shadow_opacity)))

        # Use the same larger margin as the pad estimate, so the shadow can fully fade out.
        blur_margin = blur_margin_est

        shadow_width = w + 2 * (ss + blur_margin)
        shadow_height = h + 2 * (ss + blur_margin)

        # Build a rounded mask drawn inset by blur_margin so blur can fade out naturally
        shadow_mask = create_inset_rounded_mask(
            shadow_width,
            shadow_height,
            radius + ss + shadow_radius_add,
            inset=blur_margin,
            aa=shadow_aa
        )
        if shadow_blur_enabled and sb > 0:
            shadow_mask = shadow_mask.filter(ImageFilter.GaussianBlur(sb))

        # Convert mask to an RGBA shadow image with correct opacity
        def _shadow_alpha(px: int) -> int:
            v = max(0.0, min(1.0, px / 255.0))
            g = shadow_alpha_gamma if shadow_alpha_gamma > 0 else 1.0
            v = v ** g
            return int(max(0, min(255, round(v * sa))))
        shadow = Image.new("RGBA", (shadow_width, shadow_height), (sr, sg, sbc, 0))
        shadow.putalpha(shadow_mask.point(_shadow_alpha))

        shadow_x = pad - (ss + blur_margin) + dx
        shadow_y = pad - (ss + blur_margin) + dy

        # Use alpha_composite for correct blending
        canvas.alpha_composite(shadow, (shadow_x, shadow_y))

    # --- cover rectangle with rounded corners (mask-based => clean corners) ---
    r, g, b = cfg.cover_color
    a = int(max(0, min(255, cfg.cover_opacity)))

    # Create rounded rectangle (mask-based => clean corners)
    cover_mask = create_rounded_mask(w, h, radius, aa=cover_aa)
    cover_rect = Image.new("RGBA", (w, h), (r, g, b, 0))
    def _cover_alpha(px: int) -> int:
        v = max(0.0, min(1.0, px / 255.0))
        g = cover_alpha_gamma if cover_alpha_gamma > 0 else 1.0
        v = v ** g
        return int(max(0, min(255, round(v * a))))
    cover_rect.putalpha(cover_mask.point(_cover_alpha))

    # Optional: subtle blur on the fill (applied to alpha only, avoids ugly edges)
    if cfg.blur_enabled and int(cfg.blur_radius) > 0:
        alpha = cover_rect.split()[-1].filter(ImageFilter.GaussianBlur(int(cfg.blur_radius)))
        cover_rect.putalpha(alpha)

    # Composite cover rectangle onto canvas
    canvas.alpha_composite(cover_rect, (pad, pad))

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

    print("Creating shadowed rectangle cover with rounded corners...")
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
        cover_y=1250,
        cover_w=700,
        cover_h=140,
        corner_radius=20,            # NEW: rounded corners radius
        cover_aa=2,
        cover_alpha_gamma=1.0,
        anchor_x_ratio=0.5,          # fixed exact center

        cover_color=(0, 0, 0),
        cover_opacity=245,

        blur_enabled=False,
        blur_radius=7,
        blur_padding=12,

        shadow_enabled=True,
        shadow_color=(0, 0, 0),
        shadow_opacity=120,
        shadow_blur=28,
        shadow_blur_margin_mult=4.0,
        shadow_spread=14,
        shadow_offset=(0, 0),
        shadow_aa=2,
        shadow_blur_enabled=True,
        shadow_radius_add=0,
        shadow_alpha_gamma=1.0,
        shadow_pad_extra=0,

        cover_start=0.0,
        cover_end=None,
    )

    cover_old_text(input_file, output_file, cfg)