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

    # cover color animation (NEW)
    cover_color_anim_enabled: bool = False
    cover_color_from: Tuple[int, int, int] = (0, 0, 255)   # RGB start
    cover_color_to: Tuple[int, int, int] = (0, 255, 0)     # RGB end
    cover_color_anim_duration: float = 3.0                # seconds for A -> B
    cover_color_anim_pingpong: bool = True                # A->B->A

    # animated gradient cover (NEW)
    cover_gradient_enabled: bool = False

    # gradient colors (left -> right)
    cover_gradient_left: Tuple[int, int, int] = (0, 120, 255)
    cover_gradient_right: Tuple[int, int, int] = (0, 255, 160)

    # multi-color gradient stops (optional)
    # If `cover_gradient_colors_enabled=True`, these colors are used from left->right as stops.
    # Examples:
    #   3 colors: (blue, green, pink)
    #   4 colors: (blue, cyan, green, yellow)
    cover_gradient_colors_enabled: bool = False
    cover_gradient_colors: Tuple[Tuple[int, int, int], ...] = (
        (0, 120, 255),   # blue
        (0, 255, 160),   # green
        (255, 0, 120),   # pink
    )

    # gradient animation
    cover_gradient_speed: float = 0.25   # cycles per second
    cover_gradient_width: float = 1.0    # 1.0 = full width, >1 wider & softer

    # gradient smoothing / motion function (NEW)
    # Options:
    #   "cosine"   -> seamless periodic blend (no seam edge) ✅ recommended
    #   "triangle" -> seamless periodic blend (sharper mid-transition)
    #   "wrap"     -> simple wrap (can show a seam edge)
    cover_gradient_mode: str = "cosine"
    cover_gradient_blur_radius: int = 0   # 0=off; >0 makes colors blend like ink

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


def lerp_color(c1: Tuple[int, int, int], c2: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    """Linear interpolation between two RGB colors."""
    t = max(0.0, min(1.0, t))
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


# --- Multi-stop color sampling for gradients ---
def sample_color_stops(colors: Tuple[Tuple[int, int, int], ...], t: float) -> Tuple[int, int, int]:
    """
    Sample a multi-stop gradient.
    colors: (c0, c1, ..., cn) at evenly spaced stops across 0..1
    t: 0..1
    """
    if not colors:
        return (0, 0, 0)
    if len(colors) == 1:
        return colors[0]

    t = max(0.0, min(1.0, float(t)))
    n = len(colors) - 1
    pos = t * n
    i = int(pos)
    if i >= n:
        return colors[-1]
    local_t = pos - i
    return lerp_color(colors[i], colors[i + 1], local_t)



def create_horizontal_gradient(
    width: int,
    height: int,
    left: Tuple[int, int, int],
    right: Tuple[int, int, int],
    offset: float,
    span: float = 1.0,
    mode: str = "cosine",
    blur_radius: int = 0,
    colors: Optional[Tuple[Tuple[int, int, int], ...]] = None,
) -> Image.Image:
    """
    Creates a horizontally moving RGB gradient.

    mode:
      - "cosine": seamless periodic blend (no seam)
      - "triangle": seamless periodic blend (different feel)
      - "wrap": simple wrap (may show seam)

    colors:
      - None => use left/right
      - Tuple of 3+ RGB colors => multi-stop gradient across 0..1
    """
    width = max(1, int(width))
    height = max(1, int(height))
    span = max(0.01, float(span))

    # positions across the width (0..1)
    xs = np.linspace(0.0, 1.0, width, endpoint=True, dtype=np.float32)
    p = xs * span + float(offset)

    mode_l = str(mode).lower().strip()

    # Base parameter in [0..1]
    if mode_l == "wrap":
        u = np.mod(p, 1.0)
    else:
        # seamless periodic blend 0..1 with no seam
        # cosine: u = 0.5 - 0.5*cos(2*pi*p)
        # triangle: u = 1 - |2*fract(p) - 1|
        if mode_l == "triangle":
            frac = np.mod(p, 1.0)
            u = 1.0 - np.abs(2.0 * frac - 1.0)
        else:
            u = 0.5 - 0.5 * np.cos((2.0 * np.pi) * p)

    # Build columns
    arr = np.zeros((height, width, 3), dtype=np.uint8)

    if colors is not None and len(colors) >= 2:
        # Multi-stop: sample per x
        for x in range(width):
            c = sample_color_stops(colors, float(u[x]))
            arr[:, x, :] = c
    else:
        # Two-color: vectorized lerp
        left_arr = np.array(left, dtype=np.float32)
        right_arr = np.array(right, dtype=np.float32)
        cols = left_arr + (right_arr - left_arr) * u[:, None]   # (W,3)
        cols = np.clip(cols, 0, 255).astype(np.uint8)
        arr = np.repeat(cols[None, :, :], height, axis=0)

    img = Image.fromarray(arr, mode="RGB")

    br = int(max(0, blur_radius))
    if br > 0:
        img = img.filter(ImageFilter.GaussianBlur(br))

    return img


def create_shadowed_cover_clip(video_w: int, video_h: int, cfg: Configuration) -> VideoClip:
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

    r, g, b = cfg.cover_color
    a = int(max(0, min(255, cfg.cover_opacity)))

    def make_frame(t: float):
        # --- animated cover fill (gradient or flat) ---
        if cfg.cover_gradient_enabled:
            speed = float(cfg.cover_gradient_speed)
            offset = (t * speed) % 1.0

            grad = create_horizontal_gradient(
                w,
                h,
                cfg.cover_gradient_left,
                cfg.cover_gradient_right,
                offset=offset,
                span=cfg.cover_gradient_width,
                mode=getattr(cfg, "cover_gradient_mode", "cosine"),
                blur_radius=int(getattr(cfg, "cover_gradient_blur_radius", 0)),
                colors=(cfg.cover_gradient_colors if getattr(cfg, "cover_gradient_colors_enabled", False) else None),
            )

            cover_rect = grad.convert("RGBA")
            cover_rect.putalpha(0)  # alpha added via mask below

        else:
            if cfg.cover_color_anim_enabled:
                dur = max(0.001, float(cfg.cover_color_anim_duration))
                phase = (t % dur) / dur

                if cfg.cover_color_anim_pingpong:
                    phase = phase * 2 if phase <= 0.5 else 2 - phase * 2

                r0, g0, b0 = lerp_color(cfg.cover_color_from, cfg.cover_color_to, phase)
            else:
                r0, g0, b0 = cfg.cover_color

            cover_rect = Image.new("RGBA", (w, h), (r0, g0, b0, 0))

        # reuse shadow (static, already computed)
        frame = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        frame.alpha_composite(canvas)

        cover_mask = create_rounded_mask(w, h, radius, aa=cover_aa)

        def _cover_alpha(px: int) -> int:
            v = max(0.0, min(1.0, px / 255.0))
            g = cover_alpha_gamma if cover_alpha_gamma > 0 else 1.0
            v = v ** g
            return int(max(0, min(255, round(v * a))))

        cover_rect.putalpha(cover_mask.point(_cover_alpha))
        frame.alpha_composite(cover_rect, (pad, pad))

        return np.array(frame)

    clip = VideoClip(make_frame, is_mask=False).with_position((x - pad, y - pad))
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
    output_file = "n8n/Testing/videoOuput/covered_old_textColor.mp4"

    cfg = Configuration(
        cover_y=1265,
        cover_w=700,
        cover_h=140,
        corner_radius=20,            # NEW: rounded corners radius
        cover_aa=2,
        cover_alpha_gamma=1.0,
        anchor_x_ratio=0.5,          # fixed exact center

        cover_color=(0, 0, 0),
        cover_opacity=300,

        cover_color_anim_enabled=True,
        cover_color_from=(0, 120, 255),   # blue
        cover_color_to=(0, 255, 160),     # green
        cover_color_anim_duration=3.5,
        cover_color_anim_pingpong=True,

        cover_gradient_enabled=True,
        cover_gradient_left=(0, 120, 255),   # blue
        cover_gradient_right=(0, 255, 160),  # green
        cover_gradient_colors_enabled=True,
        cover_gradient_colors=(
            (90, 180, 255),  # ice blue
            (120, 255, 220), # mint
            (190, 140, 255), # lavender
        ),
        cover_gradient_mode="cosine",  # try: "triangle" or "wrap" or "cosine"
        cover_gradient_speed=0.5,            # slow drift
        cover_gradient_width=1.4,            # softer blend
        cover_gradient_blur_radius=6,

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


    # •	cover_gradient_mode
        # •	"cosine": very smooth, no hard “seam”. Best default.
        # •	"triangle": sharper transitions (more “banding/edge” feel), still seamless.
        # •	"wrap": simple wrap-around; can show a visible seam line.
	# •	cover_gradient_speed (cycles per second)
        # •	Higher: gradient moves faster (more motion, can look distracting).
        # •	Lower: slower drift (calmer, subtle).
	# •	cover_gradient_width (how “stretched” the gradient is)
        # •	Higher (>1): wider/softer blend, slower-looking change across the rectangle.
        # •	Lower (<1): tighter gradient, faster color change across the width (can look harsh).
	# •	cover_gradient_blur_radius (Gaussian blur in pixels)
        # •	Higher: smoother “ink-like” blending, less banding; too high = muddy/washed.
        # •	Lower/0: crisp gradient; may show banding on some videos.