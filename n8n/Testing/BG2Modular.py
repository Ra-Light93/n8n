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


# =========================
# COLOR MAPPING (TEXT -> BACKGROUND PALETTE)
# =========================

def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _srgb_to_linear(c: float) -> float:
    # c in [0..1]
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _relative_luminance(rgb: Tuple[int, int, int]) -> float:
    r, g, b = rgb
    R = _srgb_to_linear(r / 255.0)
    G = _srgb_to_linear(g / 255.0)
    B = _srgb_to_linear(b / 255.0)
    return 0.2126 * R + 0.7152 * G + 0.0722 * B


def _contrast_ratio(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> float:
    La = _relative_luminance(a)
    Lb = _relative_luminance(b)
    L1, L2 = (La, Lb) if La >= Lb else (Lb, La)
    return (L1 + 0.05) / (L2 + 0.05)


def _parse_hex_color(hex_color: str) -> Tuple[int, int, int]:
    s = hex_color.strip().lstrip("#")
    if len(s) == 3:  # #RGB
        s = "".join([ch * 2 for ch in s])
    if len(s) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}")
    r = int(s[0:2], 16)
    g = int(s[2:4], 16)
    b = int(s[4:6], 16)
    return (r, g, b)


def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    r, g, b = rgb
    return f"#{r:02X}{g:02X}{b:02X}"


def _rgb_to_hsl(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    # returns (h in [0..360), s in [0..1], l in [0..1])
    r, g, b = [v / 255.0 for v in rgb]
    mx, mn = max(r, g, b), min(r, g, b)
    d = mx - mn
    l = (mx + mn) / 2.0
    if d == 0:
        return (0.0, 0.0, l)
    s = d / (2.0 - mx - mn) if l > 0.5 else d / (mx + mn)
    if mx == r:
        h = ((g - b) / d) % 6.0
    elif mx == g:
        h = ((b - r) / d) + 2.0
    else:
        h = ((r - g) / d) + 4.0
    h *= 60.0
    return (h % 360.0, _clamp01(s), _clamp01(l))


def _hsl_to_rgb(h: float, s: float, l: float) -> Tuple[int, int, int]:
    # h in degrees, s/l in [0..1]
    h = (h % 360.0) / 360.0
    s = _clamp01(s)
    l = _clamp01(l)

    def hue2rgb(p: float, q: float, t: float) -> float:
        if t < 0:
            t += 1
        if t > 1:
            t -= 1
        if t < 1 / 6:
            return p + (q - p) * 6 * t
        if t < 1 / 2:
            return q
        if t < 2 / 3:
            return p + (q - p) * (2 / 3 - t) * 6
        return p

    if s == 0:
        v = int(round(l * 255))
        return (v, v, v)

    q = l * (1 + s) if l < 0.5 else l + s - l * s
    p = 2 * l - q
    r = hue2rgb(p, q, h + 1 / 3)
    g = hue2rgb(p, q, h)
    b = hue2rgb(p, q, h - 1 / 3)
    return (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))


def _ensure_contrast(bg: Tuple[int, int, int], text: Tuple[int, int, int], min_ratio: float) -> Tuple[int, int, int]:
    """
    Adjust background lightness in HSL until contrast >= min_ratio (or max iterations).
    Keeps hue/saturation, only changes lightness.

    Important: Some (bg,text) pairs can NEVER reach strict ratios (e.g. bright red text vs white bg with min_ratio=4.5).
    In that case we return the best-contrast candidate instead of drifting to pure white/black.
    """
    target = float(min_ratio)
    current = _contrast_ratio(bg, text)
    if current >= target:
        return bg

    h, s, l0 = _rgb_to_hsl(bg)

    def search(direction: float) -> Tuple[Tuple[int, int, int], float]:
        l = l0
        best_rgb = bg
        best_cr = current
        for _ in range(40):
            l = _clamp01(l + direction * 0.03)
            cand = _hsl_to_rgb(h, s, l)
            cr = _contrast_ratio(cand, text)
            if cr > best_cr:
                best_cr = cr
                best_rgb = cand
            if cr >= target:
                return cand, cr
        return best_rgb, best_cr

    # Try both directions and pick whichever reaches target; otherwise pick best contrast.
    cand_dark, cr_dark = search(-1.0)
    cand_light, cr_light = search(1.0)

    # Prefer the one that meets the target; otherwise, the higher contrast one.
    if cr_dark >= target and cr_light >= target:
        return cand_dark if cr_dark >= cr_light else cand_light
    if cr_dark >= target:
        return cand_dark
    if cr_light >= target:
        return cand_light
    return cand_dark if cr_dark >= cr_light else cand_light


def MapColors(
    text_color: str | Tuple[int, int, int],
    n: int = 3,
    scheme: str = "complementary",
    *,
    min_contrast: float = 3.0,
    as_hex: bool = False,
    background: str = "auto",  # "auto" | "dark" | "light"
) -> List[str] | List[Tuple[int, int, int]]:
    """
    Map a TEXT color -> a BACKGROUND palette (2/3/5 colors) using math in HSL space.

    Parameters:
      text_color: "#RRGGBB" / "#RGB" or (r,g,b)
      n: number of background colors (2, 3, or 5 recommended)
      scheme:
        - "complementary": opposite hue + slight variants ✅ good default
        - "analogous": close hues around complement
        - "triadic": 3-way split hues
        - "monochrome": same hue, different lightness
      min_contrast: minimum WCAG-like contrast ratio vs text_color
      as_hex: True => return ["#..."], False => return [(r,g,b), ...]
      background:
        - "auto": detect whether input color is dark/light (luminance) and choose a matching dark/light background ✅
        - "dark": force darker backgrounds
        - "light": force lighter backgrounds

    Notes:
      - This is designed for backgrounds behind text, so it prioritizes contrast.
      - It keeps saturation moderate so the result looks “clean” in video overlays.
    """
    if isinstance(text_color, str):
        text_rgb = _parse_hex_color(text_color)
    else:
        text_rgb = (int(text_color[0]), int(text_color[1]), int(text_color[2]))

    n = int(n)
    if n <= 0:
        raise ValueError("n must be > 0")

    th, ts, tl = _rgb_to_hsl(text_rgb)

    # Choose a base hue opposite the text for background readability.
    base_h = (th + 180.0) % 360.0

    # Background saturation/lightness baseline
    # Requested behavior:
    #   - If the input (text) color is LIGHT -> choose DARK backgrounds
    #   - If the input (text) color is DARK  -> choose LIGHT backgrounds
    bg_mode = str(background).lower().strip()
    text_lum = _relative_luminance(text_rgb)  # 0..1

    dark_l = 0.16   # avoid pure black, looks nicer in video overlays
    light_l = 0.78  # avoid near-white, keeps colors visible

    if bg_mode == "dark":
        base_l = dark_l
    elif bg_mode == "light":
        base_l = light_l
    else:
        # auto: detect light/dark by luminance (simple + stable)
        # threshold ~0.50 works well for most colors
        base_l = dark_l if text_lum >= 0.50 else light_l

        # If the chosen side still gives weak contrast, flip it (rare edge cases)
        # This keeps "auto" robust for very saturated mid-luminance colors.
        test_bg = _hsl_to_rgb(base_h, 0.55, base_l)
        if _contrast_ratio(test_bg, text_rgb) < float(min_contrast):
            base_l = light_l if base_l == dark_l else dark_l

    # pleasant saturation range for backgrounds
    # - keep saturated enough to look “colorful”
    # - but not so saturated that it looks noisy behind text
    base_s = max(0.35, min(0.70, ts if ts > 0.18 else 0.55))

    scheme_l = scheme.lower().strip()

    def hues_for_scheme() -> List[float]:
        if scheme_l == "monochrome":
            return [base_h] * n
        if scheme_l == "triadic":
            # spread around the wheel
            return [(base_h + 0.0) % 360.0, (base_h + 120.0) % 360.0, (base_h + 240.0) % 360.0][:n] + [base_h] * max(0, n - 3)
        if scheme_l == "analogous":
            # around base_h
            steps = [-25.0, 0.0, 25.0, -50.0, 50.0]
            return [((base_h + steps[i % len(steps)]) % 360.0) for i in range(n)]
        # default: complementary with subtle variants
        steps = [0.0, 18.0, -18.0, 36.0, -36.0]
        return [((base_h + steps[i % len(steps)]) % 360.0) for i in range(n)]

    hues = hues_for_scheme()

    # Lightness pattern for 2/3/5 backgrounds to get variety but keep readability
    if n == 2:
        ls = [base_l, _clamp01(base_l + (0.10 if base_l < 0.5 else -0.10))]
    elif n == 3:
        ls = [base_l, _clamp01(base_l + (0.08 if base_l < 0.5 else -0.08)), _clamp01(base_l + (0.16 if base_l < 0.5 else -0.16))]
    elif n == 5:
        deltas = [0.0, 0.07, -0.07, 0.14, -0.14]
        ls = [_clamp01(base_l + (d if base_l < 0.5 else -d)) for d in deltas]
    else:
        # general fallback
        ls = [_clamp01(base_l + ((i - (n - 1) / 2) * 0.06) * (1 if base_l < 0.5 else -1)) for i in range(n)]

    # Build palette, then enforce contrast
    palette_rgb: List[Tuple[int, int, int]] = []
    for i in range(n):
        h = hues[i]
        l = ls[i]
        rgb = _hsl_to_rgb(h, base_s, l)
        rgb = _ensure_contrast(rgb, text_rgb, float(min_contrast))
        palette_rgb.append(rgb)

    if as_hex:
        return [_rgb_to_hex(c) for c in palette_rgb]
    return palette_rgb


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

#
# Example:
#   MapColors("#FFFFFF", n=3, scheme="complementary") -> 3 dark, readable backgrounds
#   MapColors("#111111", n=5, scheme="analogous") -> 5 light backgrounds with harmony

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
        cover_opacity=280,

        cover_color_anim_enabled=True,
        cover_color_from=(0, 120, 255),   # blue
        cover_color_to=(0, 255, 160),     # green
        cover_color_anim_duration=3.5,
        cover_color_anim_pingpong=True,

        cover_gradient_enabled=True,
        cover_gradient_left=(0, 120, 255),   # blue
        cover_gradient_right=(0, 255, 160),  # green
        cover_gradient_colors_enabled=True,
        cover_gradient_colors = tuple(MapColors("#05F3A0", 4, background="auto")), # type:ignore

        cover_gradient_mode="triangle",         # Verlauf-Bewegung: "cosine"=seamless weich, "triangle"=härter, "wrap"=kann Naht zeigen, “seam” ?
        cover_gradient_speed=0.3,             # Geschwindigkeit: höher = schnellerer Drift, niedriger = ruhiger (0.3)     
        cover_gradient_width=1.4,             # Breite/Stretch: höher = weicher & langsamerer Wechsel, niedriger = stärkerer Wechsel (1.4)
        cover_gradient_blur_radius=6,         # Weichzeichnung (px): höher = mehr "ink"-Blend, niedriger/0 = schärfer (6)

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


    # (
    #     (90, 180, 255),  # ice blue
    #     (120, 255, 220), # mint
    #     (190, 140, 255), # lavender
    # ),