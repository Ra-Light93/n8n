import os
import pysrt
import numpy as np
from moviepy import VideoFileClip, TextClip, CompositeVideoClip, ImageClip, VideoClip, vfx
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import argparse
import cv2



# =========================
# CONFIG (ALL VARIABLES HERE)
# =========================

@dataclass(frozen=True)
class Configuration:
    # Draw a border/frame around the whole video
    frame_enabled: bool = True
    frame_thickness: int = 12                 # px
    frame_opacity: int = 255                  # 0..255

    # Misty / shadow-like frame edge (NEW)
    frame_blur_enabled: bool = False
    frame_blur_radius: int = 8               # px | höher = weicher/mistiger Rand
    frame_blur_opacity_mult: float = 1.0     # <1.0 = weniger sichtbar nach Blur
    frame_blur_alpha_gamma: float = 1.0      # >1 härter, <1 weicher (nur Alpha)

    # Moving mist (travels WITH the moving colors) (NEW)
    # Only applied when frame_mode == "moving_palette"
    frame_moving_mist_enabled: bool = False
    frame_moving_mist_blur_radius: int = 14        # px | höher = mehr "Nebel"
    frame_moving_mist_alpha_mult: float = 0.9      # 0..2 | höher = sichtbarer Nebel
    frame_moving_mist_threshold: float = 0.06      # 0..1 | höher = Nebel nur bei starken Farbübergängen
    frame_moving_mist_edge_exclude_px: int = 4      # px | blendet statische Frame-Kanten aus (höher = weniger Nebel an den Rändern)
    frame_moving_mist_expand_alpha: bool = True    # True = Nebel darf leicht nach innen/außen laufen

    # Neon / Glow effect (NEW)
    frame_neon_enabled: bool = False
    frame_neon_glow_radius: int = 18              # px | höher = größerer Glow
    frame_neon_glow_intensity: float = 1.2        # 0..5 | höher = stärker/heller
    frame_neon_saturation_mult: float = 1.2       # 0..3 | höher = sattere Farben
    frame_neon_brightness_add: int = 0            # -255..255 | + = heller
    frame_neon_tint_rgb: Optional[Tuple[int,int,int]] = None  # None = Glow nutzt Frame-Farben; sonst Glow einfärben
    frame_neon_alpha_mult: float = 1.0            # 0..3 | verstärkt/abschwaecht Alpha des Glows
    frame_neon_glow_luma_threshold: float = 0.12   # 0..1 | Glow nur aus helleren Bereichen (höher = weniger dunkler Rand)
    frame_neon_glow_luma_softness: float = 0.10    # 0..1 | weicher Übergang der Threshold-Maske
    frame_neon_base_recolor_enabled: bool = False  # True = Basis-Frame (nicht nur Glow) wird eingefärbt (verhindert schwarzen Rand)
    frame_neon_base_tint_rgb: Optional[Tuple[int,int,int]] = None  # z.B. (0,255,255)
    frame_neon_base_tint_strength: float = 0.35    # 0..1 | höher = mehr Tint auf dem Basis-Frame
    frame_neon_base_dark_lift: int = 0             # 0..255 | hebt dunkle Kanten im Basis-Frame leicht an (0 = aus)
    # Edge color bleed (removes dark rim by bleeding local color into edge) (NEW)
    frame_neon_edge_bleed_enabled: bool = True
    frame_neon_edge_bleed_blur: int = 6            # px | blur used to sample local color
    frame_neon_edge_bleed_strength: float = 0.55   # 0..1 | higher = stronger rim removal
    frame_neon_edge_bleed_luma_cut: float = 0.35   # 0..1 | which pixels count as "too dark"

    # Frame mode
    # - "solid": single color (from frame_color_rgb/hex OR from palette/index)
    # - "moving_palette": a list of colors moves around the frame perimeter
    frame_mode: str = "solid"

    # Moving palette (used when frame_mode == "moving_palette")
    # Provide either hex list or rgb list.
    frame_colors_hex: List[str] = field(default_factory=list)      # e.g. ["#FF0000", "#00FF00", "#0000FF"]
    frame_colors_rgb: List[Tuple[int, int, int]] = field(default_factory=list)
    frame_palette_speed: float = 0.25   # cycles per second around the full perimeter
    frame_palette_direction: int = 1    # 1 = clockwise, -1 = counter-clockwise


    # Solid color selection (used when frame_mode == "solid")
    # 1) Direct override (highest priority)
    frame_color_rgb: Optional[Tuple[int, int, int]] = None
    frame_color_hex: Optional[str] = None

    # 2) Otherwise pick from the existing arrays (palette/index)
    # - "dark" uses BEST_DARK_COLOR_COMBINATIONS
    # - "light" uses BEST_LIGHT_COLOR_COMBINATIONS
    frame_palette: str = "dark"
    frame_palette_index: int = 0              # 0..len-1
    frame_color_index_in_combo: int = 0       # 0 or 1 (each combo has 2 colors)

    # optional inset so the frame is drawn slightly inside the edges
    frame_inset: int = 0                      # px

    # time range (default: full duration)
    start: float = 0.0
    end: Optional[float] = None               # None => till end


# =========================
# FRAME / BORDER CLIP
# =========================

def _parse_hex_color(s: str) -> Tuple[int, int, int]:
    s = s.strip()
    if s.startswith("#"):
        s = s[1:]
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    if len(s) != 6:
        raise ValueError(f"Invalid hex color: {s}")
    r = int(s[0:2], 16)
    g = int(s[2:4], 16)
    b = int(s[4:6], 16)
    return (r, g, b)

def _colors_from_config(cfg: Configuration) -> np.ndarray:
    """Return an (N,3) uint8 palette from config for moving palette mode."""
    colors: List[Tuple[int, int, int]] = []

    if cfg.frame_colors_rgb:
        for r, g, b in cfg.frame_colors_rgb:
            colors.append((int(r), int(g), int(b)))

    if cfg.frame_colors_hex:
        for s in cfg.frame_colors_hex:
            colors.append(_parse_hex_color(s))

    # fallback: use the selected palette-combo (2 colors) so moving mode always works
    if not colors:
        pal = str(cfg.frame_palette).lower().strip()
        combos = BEST_LIGHT_COLOR_COMBINATIONS if pal == "light" else BEST_DARK_COLOR_COMBINATIONS
        if combos:
            i = int(cfg.frame_palette_index) % len(combos)
            combo = combos[i]
            colors = [tuple(combo[0]), tuple(combo[1])]  # type: ignore

    if not colors:
        colors = [(255, 255, 255)]

    arr = np.array(colors, dtype=np.uint8)
    # if user gave a single color, duplicate it so interpolation math stays valid
    if arr.shape[0] == 1:
        arr = np.vstack([arr, arr])
    return arr


def _sample_palette(colors: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Vectorized sampling of multi-stop palette. colors: (N,3) uint8, u: any shape float in [0,1]."""
    n = int(colors.shape[0])
    u = np.mod(u, 1.0).astype(np.float32)

    # Cyclic sampling: last color blends back into the first color
    pos = u * n  # 0..n
    i0_raw = np.floor(pos).astype(np.int32)        # 0..n
    f = (pos - i0_raw.astype(np.float32))[..., None]

    i0 = np.mod(i0_raw, n)                         # 0..n-1
    i1 = np.mod(i0 + 1, n)                         # next index, wraps to 0

    c0 = colors[i0].astype(np.float32)
    c1 = colors[i1].astype(np.float32)

    out = c0 + (c1 - c0) * f
    return np.clip(out, 0, 255).astype(np.uint8)


def _pick_frame_color(cfg: Configuration) -> Tuple[int, int, int]:
    # Solid mode only
    if cfg.frame_color_rgb is not None:
        r, g, b = cfg.frame_color_rgb
        return (int(r), int(g), int(b))

    if cfg.frame_color_hex is not None:
        return _parse_hex_color(cfg.frame_color_hex)

    pal = str(cfg.frame_palette).lower().strip()
    combos = BEST_LIGHT_COLOR_COMBINATIONS if pal == "light" else BEST_DARK_COLOR_COMBINATIONS

    if not combos:
        return (255, 255, 255)

    i = int(cfg.frame_palette_index) % len(combos)
    combo = combos[i]
    j = int(cfg.frame_color_index_in_combo) % len(combo)
    return tuple(combo[j])  # type: ignore


# ===== helper: RGBA blur with premultiplied alpha (OpenCV) =====
def _blur_rgba_premult(img_rgba: np.ndarray, radius: int) -> np.ndarray:
    """
    Blur RGBA using premultiplied alpha to avoid dark/black fringes.
    Uses OpenCV GaussianBlur for speed (fallbacks to no-op if radius <= 0).
    """
    r = int(max(0, radius))
    if r <= 0:
        return img_rgba

    x = img_rgba.astype(np.float32)
    a = x[:, :, 3:4] / 255.0  # (H,W,1)

    # premultiply
    rgb_pm = x[:, :, 0:3] * a
    pm = np.concatenate([rgb_pm, x[:, :, 3:4]], axis=2)  # keep alpha in 0..255 space

    # OpenCV blur (fast). sigma=r, kernel auto.
    b = cv2.GaussianBlur(pm, ksize=(0, 0), sigmaX=float(r), sigmaY=float(r), borderType=cv2.BORDER_REPLICATE)

    a_b = b[:, :, 3:4] / 255.0
    rgb_b_pm = b[:, :, 0:3]

    # unpremultiply (avoid division by zero)
    denom = np.maximum(a_b, 1e-6)
    rgb_b = rgb_b_pm / denom
    rgb_b = np.clip(rgb_b, 0.0, 255.0)

    out = np.concatenate([rgb_b, b[:, :, 3:4]], axis=2)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


# ===== helper: perimeter position to (x, y) on frame centerline =====
def _perim_pos_to_xy(s: float, x0: int, y0: int, x1: int, y1: int, thickness: int) -> Tuple[float, float]:
    """Map perimeter distance s (0..perim) to a point on the frame centerline."""
    top_len = float(max(1, x1 - x0 + 1))
    side_len = float(max(1, y1 - y0 + 1))
    perim = 2.0 * (top_len + side_len)
    s = float(s % perim)

    half = thickness / 2.0
    # TOP: left->right
    if s < top_len:
        return (x0 + s, y0 + half)
    s -= top_len
    # RIGHT: top->bottom
    if s < side_len:
        return (x1 - half, y0 + s)
    s -= side_len
    # BOTTOM: right->left
    if s < top_len:
        return (x1 - s, y1 - half)
    s -= top_len
    # LEFT: bottom->top
    return (x0 + half, y1 - s)


# ===== fast inner mask helper (OpenCV erosion) =====
def _compute_inner_mask(alpha01: np.ndarray, exclude: int) -> np.ndarray:
    """
    Fast erosion-based inner mask to exclude static frame edges.
    alpha01: float32 alpha in [0,1]
    returns float32 mask in {0,1}
    """
    ex = int(max(0, exclude))
    if ex <= 0:
        return (alpha01 > 0.5).astype(np.float32)

    m = (alpha01 > 0.5).astype(np.uint8) * 255
    kernel = np.ones((3, 3), dtype=np.uint8)
    er = cv2.erode(m, kernel, iterations=ex)
    return (er > 127).astype(np.float32)

# ===== moving mist helper (travels with the moving palette transitions) =====
def _apply_moving_mist(img_rgba: np.ndarray, cfg: Configuration, inner_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Adds a mist layer that follows moving palette transitions.
    Works by blurring the frame and using a mask derived from color-change magnitude,
    so mist appears mainly where colors transition (and therefore moves over time).
    """
    if not bool(getattr(cfg, "frame_moving_mist_enabled", False)):
        return img_rgba

    br = int(max(0, getattr(cfg, "frame_moving_mist_blur_radius", 0)))
    if br <= 0:
        return img_rgba

    # Work masks
    alpha = img_rgba[:, :, 3].astype(np.float32) / 255.0

    # Inner mask: cacheable (depends only on alpha-shape + exclude)
    if inner_mask is None:
        exclude = int(max(0, getattr(cfg, "frame_moving_mist_edge_exclude_px", 0)))
        inner_mask = _compute_inner_mask(alpha, exclude)

    # Gradient magnitude of RGB (high at palette color transitions, low on flat areas)
    rgb = img_rgba[:, :, 0:3].astype(np.float32) / 255.0

    # x gradient
    gx = np.abs(rgb[:, 1:, :] - rgb[:, :-1, :])
    gx = np.pad(gx, ((0, 0), (0, 1), (0, 0)), mode="edge")

    # y gradient
    gy = np.abs(rgb[1:, :, :] - rgb[:-1, :, :])
    gy = np.pad(gy, ((0, 1), (0, 0), (0, 0)), mode="edge")

    grad = (gx + gy).mean(axis=2)  # 0..~1

    # Only consider gradients well inside the frame (exclude edges)
    grad = grad * inner_mask

    thr = float(getattr(cfg, "frame_moving_mist_threshold", 0.06))
    thr = max(0.0, min(0.95, thr))
    mask = (grad - thr) / max(1e-6, (1.0 - thr))
    mask = np.clip(mask, 0.0, 1.0)

    # Blur the mask a bit by using the blurred RGBA as a carrier (cheap, no scipy)
    blur = cv2.GaussianBlur(img_rgba, ksize=(0, 0), sigmaX=float(br), sigmaY=float(br), borderType=cv2.BORDER_REPLICATE)
    rgb_blur = blur[:, :, 0:3].astype(np.float32)

    # Base alpha: optionally allow slight expansion using blurred alpha, but still respect inner_mask
    base_alpha = alpha
    if bool(getattr(cfg, "frame_moving_mist_expand_alpha", True)):
        base_alpha = np.maximum(base_alpha, blur[:, :, 3].astype(np.float32) / 255.0)
    base_alpha = base_alpha * inner_mask

    alpha_mult = float(getattr(cfg, "frame_moving_mist_alpha_mult", 0.9))
    alpha_mult = max(0.0, alpha_mult)

    mist_alpha = np.clip(base_alpha * mask * alpha_mult, 0.0, 1.0)

    # Composite: overlay blurred RGB only where mist_alpha>0
    out = img_rgba.copy().astype(np.float32)
    out[:, :, 0:3] = out[:, :, 0:3] * (1.0 - mist_alpha[:, :, None]) + rgb_blur * (mist_alpha[:, :, None])

    # Alpha: keep original alpha, add a little where mist exists
    out_a = np.maximum(out[:, :, 3] / 255.0, mist_alpha)
    out[:, :, 3] = np.clip(out_a * 255.0, 0.0, 255.0)

    return out.astype(np.uint8)


# ===== neon glow helper (optional, post-process) =====
def _apply_neon_glow(img_rgba: np.ndarray, cfg: Configuration) -> np.ndarray:
    """
    Neon glow post-process for the frame.
    - Builds a blurred glow layer from the frame (or a tint color)
    - Additively blends glow into RGB
    - Optionally boosts saturation and brightness
    Keeps the original alpha while allowing a soft outer glow.
    """
    if not bool(getattr(cfg, "frame_neon_enabled", False)):
        return img_rgba

    br = int(max(0, getattr(cfg, "frame_neon_glow_radius", 0)))
    if br <= 0:
        return img_rgba

    intensity = float(getattr(cfg, "frame_neon_glow_intensity", 1.2))
    intensity = max(0.0, intensity)

    sat_mult = float(getattr(cfg, "frame_neon_saturation_mult", 1.2))
    sat_mult = max(0.0, sat_mult)

    bright_add = int(getattr(cfg, "frame_neon_brightness_add", 0))

    alpha_mult = float(getattr(cfg, "frame_neon_alpha_mult", 1.0))
    alpha_mult = max(0.0, alpha_mult)

    tint = getattr(cfg, "frame_neon_tint_rgb", None)

    # Build glow carrier: premultiplied-alpha blur (prevents dark rim)
    glow = _blur_rgba_premult(img_rgba, br).astype(np.float32)

    base = img_rgba.astype(np.float32)

    # ----- edge color bleed: remove dark rim by blending in LOCAL (original) color -----
    if bool(getattr(cfg, "frame_neon_edge_bleed_enabled", False)):
        ebb = int(max(0, getattr(cfg, "frame_neon_edge_bleed_blur", 0)))
        ebs = float(getattr(cfg, "frame_neon_edge_bleed_strength", 0.55))
        ebs = max(0.0, min(1.0, ebs))
        cut = float(getattr(cfg, "frame_neon_edge_bleed_luma_cut", 0.35))
        cut = max(0.0, min(1.0, cut))

        if ebb > 0 and ebs > 0.0:
            # local color carrier (small blur) – keeps hue identical to original moving palette
            base_blur = _blur_rgba_premult(img_rgba, ebb).astype(np.float32)

            rgb = base[:, :, 0:3]
            alpha = base[:, :, 3] / 255.0

            luma = (0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]) / 255.0

            # weight: only for dark pixels inside the frame (alpha>0)
            w = np.clip((cut - luma) / max(1e-6, cut), 0.0, 1.0)
            w = w * alpha  # only where frame exists
            w = w * ebs

            base[:, :, 0:3] = rgb * (1.0 - w[:, :, None]) + base_blur[:, :, 0:3] * (w[:, :, None])

    # ----- optional: recolor / lift the BASE frame to avoid black outline -----
    if bool(getattr(cfg, "frame_neon_base_recolor_enabled", False)):
        bt = getattr(cfg, "frame_neon_base_tint_rgb", None)
        if bt is None:
            bt = getattr(cfg, "frame_neon_tint_rgb", None)
        if bt is not None:
            tr, tg, tb = bt
            strength = float(getattr(cfg, "frame_neon_base_tint_strength", 0.35))
            strength = max(0.0, min(1.0, strength))
            tint_rgb = np.array([float(tr), float(tg), float(tb)], dtype=np.float32)[None, None, :]
            base[:, :, 0:3] = base[:, :, 0:3] * (1.0 - strength) + tint_rgb * strength

    lift = int(getattr(cfg, "frame_neon_base_dark_lift", 0))
    if lift != 0:
        # lift only darker pixels a bit (prevents dark/black edge)
        rgb = base[:, :, 0:3]
        luma = (0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2])
        w = np.clip((80.0 - luma) / 80.0, 0.0, 1.0)  # 1 for very dark, 0 for bright
        base[:, :, 0:3] = np.clip(rgb + (w[:, :, None] * float(lift)), 0.0, 255.0)

    # Glow alpha (soft outer glow) + mask to suppress dark/black outlines
    g_a = (glow[:, :, 3] / 255.0) * alpha_mult
    g_a = np.clip(g_a, 0.0, 1.0)

    # Build a "glow source" mask from luminance: dark pixels contribute less glow
    thr = float(getattr(cfg, "frame_neon_glow_luma_threshold", 0.12))
    thr = max(0.0, min(0.95, thr))
    soft = float(getattr(cfg, "frame_neon_glow_luma_softness", 0.10))
    soft = max(1e-4, min(1.0, soft))

    src_rgb = base[:, :, 0:3]
    src_luma = (0.2126 * src_rgb[:, :, 0] + 0.7152 * src_rgb[:, :, 1] + 0.0722 * src_rgb[:, :, 2]) / 255.0
    # smoothstep-like
    m = np.clip((src_luma - thr) / soft, 0.0, 1.0)
    m = m * m * (3.0 - 2.0 * m)

    g_a = g_a * m

    # Choose glow color: either tinted or from blurred RGB
    if tint is not None:
        tr, tg, tb = tint
        g_rgb = np.stack(
            [
                np.full_like(g_a, float(tr)),
                np.full_like(g_a, float(tg)),
                np.full_like(g_a, float(tb)),
            ],
            axis=2,
        )
    else:
        g_rgb = glow[:, :, 0:3]

    # Additive blend scaled by alpha and intensity
    add = g_rgb * (g_a[:, :, None] * intensity)

    out_rgb = base[:, :, 0:3] + add

    # Brightness adjust
    if bright_add != 0:
        out_rgb = out_rgb + float(bright_add)

    # Saturation boost (simple, stable)
    if sat_mult != 1.0:
        gray = out_rgb.mean(axis=2, keepdims=True)
        out_rgb = gray + (out_rgb - gray) * sat_mult

    out_rgb = np.clip(out_rgb, 0.0, 255.0)

    # Alpha: keep original, but allow slight extension where glow exists
    out_a = np.maximum(base[:, :, 3] / 255.0, g_a)
    out_a = np.clip(out_a, 0.0, 1.0)

    out = base.copy()
    out[:, :, 0:3] = out_rgb
    out[:, :, 3] = out_a * 255.0

    return out.astype(np.uint8)


def create_border_frame_clip(video_w: int, video_h: int, cfg: Configuration) -> VideoClip:
    """Transparent RGBA clip that draws a border (frame) on the video edges."""
    W = int(max(1, video_w))
    H = int(max(1, video_h))

    _mist_inner_mask_cache: Dict[Tuple[int, int, int], np.ndarray] = {}

    def make_frame(t: float):
        img = np.zeros((H, W, 4), dtype=np.uint8)

        if not cfg.frame_enabled:
            return img

        thickness = int(max(1, cfg.frame_thickness))
        inset = int(max(0, cfg.frame_inset))
        a = int(max(0, min(255, cfg.frame_opacity)))

        x0 = inset
        y0 = inset
        x1 = max(inset, W - 1 - inset)
        y1 = max(inset, H - 1 - inset)

        inner_w = max(1, x1 - x0 + 1)
        inner_h = max(1, y1 - y0 + 1)

        # clamp thickness so bands stay inside
        thickness = int(min(thickness, (inner_w + 1) // 2, (inner_h + 1) // 2))

        mode = str(getattr(cfg, "frame_mode", "solid")).lower().strip()

        if mode != "moving_palette":
            r, g, b = _pick_frame_color(cfg)
            # top
            img[y0:y0 + thickness, x0:x1 + 1, 0:3] = (r, g, b)
            img[y0:y0 + thickness, x0:x1 + 1, 3] = a
            # bottom
            img[y1 - thickness + 1:y1 + 1, x0:x1 + 1, 0:3] = (r, g, b)
            img[y1 - thickness + 1:y1 + 1, x0:x1 + 1, 3] = a
            # left
            img[y0:y1 + 1, x0:x0 + thickness, 0:3] = (r, g, b)
            img[y0:y1 + 1, x0:x0 + thickness, 3] = a
            # right
            img[y0:y1 + 1, x1 - thickness + 1:x1 + 1, 0:3] = (r, g, b)
            img[y0:y1 + 1, x1 - thickness + 1:x1 + 1, 3] = a
            # fall through to optional blur post-process
        else:
            # ===== moving palette mode =====
            colors = _colors_from_config(cfg)  # (N,3)
            speed = float(getattr(cfg, "frame_palette_speed", 0.25))
            direction = 1 if int(getattr(cfg, "frame_palette_direction", 1)) >= 0 else -1

            top_len = inner_w
            side_len = inner_h
            perim = float(2 * (top_len + side_len))
            if perim <= 1.0:
                return img

            # Global offset in [0..1)
            offset = (t * speed * direction) % 1.0

            # TOP (left->right)
            xs = np.linspace(0.0, (top_len - 1) / perim, top_len, dtype=np.float32)
            u_top = (offset + xs) % 1.0
            rgb_top = _sample_palette(colors, u_top)  # (top_len,3)
            band = np.repeat(rgb_top[None, :, :], thickness, axis=0)
            img[y0:y0 + thickness, x0:x1 + 1, 0:3] = band
            img[y0:y0 + thickness, x0:x1 + 1, 3] = a

            # RIGHT (top->bottom)
            ys = np.linspace(0.0, (side_len - 1) / perim, side_len, dtype=np.float32)
            u_right = (offset + (top_len / perim) + ys) % 1.0
            rgb_right = _sample_palette(colors, u_right)  # (side_len,3)
            band_r = np.repeat(rgb_right[:, None, :], thickness, axis=1)
            img[y0:y1 + 1, x1 - thickness + 1:x1 + 1, 0:3] = band_r
            img[y0:y1 + 1, x1 - thickness + 1:x1 + 1, 3] = a

            # BOTTOM (right->left)
            xs2 = np.linspace(0.0, (top_len - 1) / perim, top_len, dtype=np.float32)
            u_bottom = (offset + ((top_len + side_len) / perim) + xs2) % 1.0
            rgb_bottom = _sample_palette(colors, u_bottom)[::-1]  # reverse direction along edge
            band_b = np.repeat(rgb_bottom[None, :, :], thickness, axis=0)
            img[y1 - thickness + 1:y1 + 1, x0:x1 + 1, 0:3] = band_b
            img[y1 - thickness + 1:y1 + 1, x0:x1 + 1, 3] = a

            # LEFT (bottom->top)
            ys2 = np.linspace(0.0, (side_len - 1) / perim, side_len, dtype=np.float32)
            u_left = (offset + ((2 * top_len + side_len) / perim) + ys2) % 1.0
            rgb_left = _sample_palette(colors, u_left)[::-1]
            band_l = np.repeat(rgb_left[:, None, :], thickness, axis=1)
            img[y0:y1 + 1, x0:x0 + thickness, 0:3] = band_l
            img[y0:y1 + 1, x0:x0 + thickness, 3] = a


        # ===== moving mist that follows the moving palette transitions =====
        if mode == "moving_palette":
            ex = int(max(0, getattr(cfg, "frame_moving_mist_edge_exclude_px", 0)))
            key = (H, W, ex)
            inner_mask = _mist_inner_mask_cache.get(key)
            if inner_mask is None:
                alpha01 = img[:, :, 3].astype(np.float32) / 255.0
                inner_mask = _compute_inner_mask(alpha01, ex)
                _mist_inner_mask_cache[key] = inner_mask
            img = _apply_moving_mist(img, cfg, inner_mask=inner_mask)

        # ===== neon glow post-process (optional) =====
        img = _apply_neon_glow(img, cfg)

        # ===== optional misty/shadow-like blur post-process =====
        if bool(getattr(cfg, "frame_blur_enabled", False)):
            br = int(max(0, getattr(cfg, "frame_blur_radius", 0)))
            if br > 0:
                # blur RGBA so edges get misty/soft (OpenCV for speed)
                img = cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=float(br), sigmaY=float(br), borderType=cv2.BORDER_REPLICATE).astype(np.uint8)

            # adjust alpha softness (gamma) and intensity (mult)
            a_gamma = float(getattr(cfg, "frame_blur_alpha_gamma", 1.0))
            a_mult = float(getattr(cfg, "frame_blur_opacity_mult", 1.0))
            if a_gamma != 1.0 or a_mult != 1.0:
                alpha = img[:, :, 3].astype(np.float32) / 255.0
                alpha = np.clip(alpha, 0.0, 1.0)
                # gamma: <1 softer, >1 harder
                if a_gamma != 1.0:
                    alpha = np.power(alpha, max(0.05, a_gamma))
                if a_mult != 1.0:
                    alpha = np.clip(alpha * max(0.0, a_mult), 0.0, 1.0)
                img[:, :, 3] = (alpha * 255.0).astype(np.uint8)

        return img

    return VideoClip(make_frame, is_mask=False)


# =========================
# REELS / SHORTS PREVIEW RENDERER
# =========================

def render_reel_preview(
    output_video_path: str,
    cfg: Configuration,
    *,
    duration: float = 10.0,
    fps: int = 30,
    size: Tuple[int, int] = (1080, 1920),  # Shorts / Reels (9:16)
    bg_color: Tuple[int, int, int] = (255, 255, 255),
):
    """Preview-only vertical renderer (white background) + border frame."""
    W, H = size

    bg = Image.new("RGB", (W, H), bg_color)
    bg_clip = ImageClip(np.array(bg)).with_duration(duration)

    frame = create_border_frame_clip(W, H, cfg).with_duration(duration)

    final = CompositeVideoClip([bg_clip, frame], size=(W, H))

    out_dir = os.path.dirname(output_video_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    final.write_videofile(
        output_video_path,
        codec="libx264",
        fps=fps,
        preset="ultrafast",
        audio=False,
        threads=4,
    )

    final.close()
    return output_video_path


# =========================
# NORMAL VIDEO: ADD FRAME
# =========================

def add_frame_to_video(
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

    # duration range
    start = float(cfg.start)
    end = float(cfg.end) if cfg.end is not None else float(video.duration)
    dur = max(0.0, end - start)

    frame = create_border_frame_clip(video.w, video.h, cfg).with_duration(dur).with_start(start)

    final = CompositeVideoClip([video, frame])

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

def mid_rgb(c1: tuple[int, int, int], c2: tuple[int, int, int]) -> tuple[int, int, int]:
    """Return the midpoint RGB color between two RGB colors."""
    r = (int(c1[0]) + int(c2[0])) // 2
    g = (int(c1[1]) + int(c2[1])) // 2
    b = (int(c1[2]) + int(c2[2])) // 2
    return (r, g, b)



# =========================
# TEST RUN (UPDATED)
# =========================

#
# Example:
#   MapColors("#FFFFFF", n=3, scheme="complementary") -> 3 dark, readable backgrounds
#   MapColors("#111111", n=5, scheme="analogous") -> 5 light backgrounds with harmony



BEST_LIGHT_COLOR_COMBINATIONS = [
    [(236, 245, 255), (210, 230, 255)],            # Text: dunkelgrau / schwarz
    [(240, 255, 250), (200, 240, 230)],            # Text: dunkelgrün
    [(255, 245, 235), (255, 220, 190)],            # Text: braun / dunkelgrau
    [(245, 240, 255), (220, 210, 255)],            # Text: dunkelviolett
    [(255, 240, 245), (255, 210, 225)],            # Text: dunkelrot / anthrazit

    [(235, 250, 255), (190, 225, 240)],            # Text: navy
    [(245, 255, 240), (210, 240, 200)],            # Text: dunkelgrün
    [(255, 250, 230), (255, 230, 180)],            # Text: dunkelbraun
    [(240, 240, 240), (210, 210, 210)],            # Text: schwarz
    [(250, 240, 255), (225, 210, 240)],            # Text: dunkelviolett

    [(240, 255, 255), (200, 235, 235)],            # Text: petrol
    [(255, 248, 240), (235, 215, 200)],            # Text: espresso
    [(245, 255, 245), (215, 235, 215)],            # Text: dunkelgrün
    [(255, 235, 240), (240, 200, 210)],            # Text: dunkelrot
    [(245, 245, 255), (215, 215, 240)],            # Text: navy

    [(255, 255, 240), (235, 235, 200)],            # Text: dunkeloliv
    [(240, 250, 245), (200, 225, 215)],            # Text: petrol
    [(255, 245, 250), (240, 210, 225)],            # Text: dunkelrosa
    [(245, 240, 235), (220, 205, 195)],            # Text: dunkelbraun
    [(235, 240, 255), (205, 215, 240)],            # Text: navy

    [(250, 255, 245), (220, 235, 215)],            # Text: dunkelgrün
    [(255, 250, 240), (235, 225, 205)],            # Text: dunkelbraun
    [(245, 255, 250), (210, 240, 225)],            # Text: petrol
    [(255, 245, 230), (235, 215, 185)],            # Text: espresso
    [(240, 245, 255), (215, 220, 240)],            # Text: navy

    [(255, 240, 240), (235, 210, 210)],            # Text: dunkelrot
    [(245, 255, 240), (220, 240, 210)],            # Text: dunkelgrün
    [(240, 255, 250), (210, 235, 230)],            # Text: petrol
    [(255, 250, 245), (235, 220, 215)],            # Text: dunkelgrau
]

BEST_DARK_COLOR_COMBINATIONS = [
    [(20, 24, 36), (40, 48, 72)],                   # Text: weiß
    [(18, 30, 25), (35, 70, 55)],                   # Text: weiß
    [(40, 20, 20), (90, 40, 40)],                   # Text: creme / weiß
    [(30, 20, 45), (80, 60, 120)],                  # Text: weiß
    [(25, 25, 25), (60, 60, 60)],                   # Text: weiß

    [(10, 35, 60), (20, 90, 130)],                  # Text: weiß
    [(20, 45, 30), (50, 110, 70)],                  # Text: weiß
    [(60, 35, 15), (130, 80, 40)],                  # Text: creme
    [(30, 30, 30), (90, 90, 90)],                   # Text: weiß
    [(45, 20, 60), (110, 70, 140)],                 # Text: weiß

    [(15, 50, 55), (40, 110, 120)],                 # Text: weiß
    [(55, 40, 25), (120, 90, 60)],                  # Text: creme
    [(20, 55, 25), (60, 120, 70)],                  # Text: weiß
    [(60, 25, 35), (130, 60, 80)],                  # Text: weiß
    [(20, 25, 55), (60, 70, 130)],                  # Text: weiß

    [(55, 55, 20), (120, 120, 60)],                 # Text: weiß
    [(15, 40, 35), (50, 100, 90)],                  # Text: weiß
    [(60, 30, 45), (130, 70, 100)],                 # Text: weiß
    [(45, 35, 25), (100, 80, 60)],                  # Text: creme
    [(25, 30, 60), (60, 70, 130)],                  # Text: weiß

    [(20, 45, 25), (55, 110, 60)],                  # Text: weiß
    [(60, 45, 30), (120, 90, 60)],                  # Text: creme
    [(20, 50, 45), (55, 110, 100)],                 # Text: weiß
    [(60, 45, 20), (130, 100, 60)],                 # Text: creme
    [(25, 35, 55), (60, 80, 120)],                  # Text: weiß

    [(60, 20, 20), (130, 50, 50)],                  # Text: weiß
    [(25, 55, 25), (70, 120, 70)],                  # Text: weiß
    [(20, 55, 50), (60, 120, 110)],                 # Text: weiß
    [(55, 45, 40), (110, 90, 80)],                  # Text: creme
]


if __name__ == "__main__":
    input_file = "n8n/Testing/videoOuput/last/last.mp4"
    output_file = "n8n/Testing/videoOuput/last/lastV1.mp4"

    cfg = Configuration(
            frame_enabled=True,
            frame_thickness=16,
            frame_opacity=255,
            frame_mode="moving_palette",
            frame_colors_rgb= [(30, 20, 45), (80, 60, 120)],
            frame_palette_speed=0.20,
            frame_palette_direction=1,
            frame_color_hex=None,
            frame_palette="dark",               # uses BEST_DARK_COLOR_COMBINATIONS
            frame_palette_index=0,              # arbitrary: first combo
            frame_color_index_in_combo=0,       # arbitrary: first color in that combo
            frame_inset=0,
            start=0.0,
            end=None,

            frame_blur_enabled=False,               # Aktiviert weichen Rand (Blur)
            frame_blur_radius=5,                    # Blur-Radius; höher = weicherer Rand
            frame_blur_opacity_mult=0.9,            # Multipliziert Sichtbarkeit nach Blur; niedriger = weniger sichtbar
            frame_blur_alpha_gamma=0.85,            # Alpha-Gamma; <1 = weicher, >1 = härter
            frame_moving_mist_enabled=True,         # Aktiviert bewegten Nebel-Effekt
            frame_moving_mist_blur_radius=4,        # Blur für Nebel; höher = mehr Ausbreitung
            frame_moving_mist_alpha_mult=0.3,       # Sichtbarkeit des Nebels; höher = stärker sichtbar
            frame_moving_mist_threshold=0.10,       # Schwelle; höher = Nebel nur bei starken Übergängen
            frame_moving_mist_edge_exclude_px=1,    # blendet statische Frame-Kanten aus (höher = weniger Rand-Nebel)
            frame_moving_mist_expand_alpha=False,    # Nebel darf Alpha nach innen/außen erweitern

            frame_neon_enabled=True,                # Neon/Glow aktivieren
            frame_neon_glow_radius=4,               # größer = weiterer Glow
            frame_neon_glow_intensity=1.25,         # stärker = heller
            frame_neon_saturation_mult=1.05,        #1.1,    # sattere Farben
            frame_neon_brightness_add=0,            # + macht heller
            frame_neon_tint_rgb=None,               # None = Frame-Farben nutzen; z.B. (0,255,255) für Cyan-Neon
            frame_neon_alpha_mult=1.1,   #1.0       # Glow-Alpha verstärken/abschwaechen
            frame_neon_glow_luma_threshold=0.0,     # 0.12,  höher = weniger Glow aus dunklen Bereichen (weniger schwarzer Rand)
            frame_neon_glow_luma_softness= 0.0,     # 0.10 weicher Übergang der Maske
            frame_neon_base_recolor_enabled=False,   # Basis einfärben -> reduziert schwarzen Rand stark
            frame_neon_base_tint_strength=0.35,     # Stärke der Basis-Einfärbung
            frame_neon_base_dark_lift=12,           # hebt dunkle Kanten leicht an
            frame_neon_edge_bleed_enabled=False,    # entfernt dunklen Rand (bleed lokale Farbe in die Kante)
            frame_neon_edge_bleed_blur=2,           # kleiner Blur = Farbe bleibt “original”
            frame_neon_edge_bleed_strength=0.55,  #0.55  # höher = weniger dunkle Kante
            frame_neon_edge_bleed_luma_cut=0.35, #0.35   # höher = mehr Pixel gelten als „zu dunkel“
        )

        # Run immediately (vertical Shorts / Reels preview)
    try: 
        v = VideoFileClip(input_file) #type:ignore
        dur = float(v.duration)
        v.close()
        if not (dur > 0.0):
            dur = 2.0
    except Exception:
        dur = 2.0

    render_reel_preview(
        output_video_path=output_file,
        cfg=cfg,
        duration=dur,
        fps=20,
        size=(720, 1280),
    )