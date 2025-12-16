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





# =========================
# CONFIG (ALL VARIABLES HERE)
# =========================

@dataclass(frozen=True)
class Configuration:
    # Draw a border/frame around the whole video
    frame_enabled: bool = True
    frame_thickness: int = 12                 # px
    frame_opacity: int = 255                  # 0..255

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
    pos = u * (n - 1)
    i0 = np.floor(pos).astype(np.int32)
    i1 = np.clip(i0 + 1, 0, n - 1)
    f = (pos - i0).astype(np.float32)[..., None]

    c0 = colors[i0]
    c1 = colors[i1]
    out = c0.astype(np.float32) + (c1.astype(np.float32) - c0.astype(np.float32)) * f
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


def create_border_frame_clip(video_w: int, video_h: int, cfg: Configuration) -> VideoClip:
    """Transparent RGBA clip that draws a border (frame) on the video edges."""
    W = int(max(1, video_w))
    H = int(max(1, video_h))

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
            return img

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
    input_file = "n8n/Downloads/Sakinah Labs/TestVideo.mp4"
    output_file = "n8n/Testing/videoOuput/last anpassung ArrayFarbe.mp4"

    cfg = Configuration(
        frame_enabled=True,
        frame_thickness=14,
        frame_opacity=255,
        frame_mode="moving_palette",
         frame_colors_rgb=  [(148, 0, 211),(0,0,0),(255,255,255)],
        frame_palette_speed=0.35,
        frame_palette_direction=1,
        frame_color_hex=None,
        frame_palette="dark",              # uses BEST_DARK_COLOR_COMBINATIONS
        frame_palette_index=0,              # arbitrary: first combo
        frame_color_index_in_combo=0,       # arbitrary: first color in that combo
        frame_inset=0,
        start=0.0,
        end=None,
    )

    # Run immediately (vertical Shorts / Reels preview)
    try:
        v = VideoFileClip(input_file)
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
    )