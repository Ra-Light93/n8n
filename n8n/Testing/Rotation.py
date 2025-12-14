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
    bottom_margin: int = 140        # (not used in this version, kept for compatibility)

    # --- text style ---
    font_name: str = "Verdana Bold"      # can be a font name or a path to .ttf/.otf
    font_size: int = 73
    border_size: int = 12

    # --- extra padding around text (prevents cut-off) ---
    pad_side_extra: int = 300
    pad_top_extra: int = 300
    pad_bottom_extra: int = 300

    # --- glow / soft shadow behind text (like a dark halo) ---
    glow_enabled: bool = True
    glow_radius: int = 12
    glow_spread: int = 12
    glow_alpha: int = 255

    # --- absolute Y position (top-left y of the subtitle image) ---
    y_position: int = 500

    # --- pop / bubble-in animation (optional) ---
    pop_min_scale: float = 1.0
    pop_dur: float = 0.60


SubtitleConfig = Configuration

# =========================
# ANIMATIONS (reusable)
# =========================

def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

def _ease_out_back(u: float, overshoot: float = 1.70158) -> float:
    c1 = float(overshoot)
    c3 = c1 + 1.0
    return 1.0 + c3 * (u - 1.0) ** 3 + c1 * (u - 1.0) ** 2

def _in_place_position(
    video_w: int,
    y_top: float,
    base_w: float,
    base_h: float,
    start_t: float,
    scale_at_local_t,
):
    anchor_cx = float(video_w) / 2.0
    anchor_cy = float(y_top) + (float(base_h) / 2.0)

    def pos(t):
        lt = max(0.0, float(t) - float(start_t))
        s = float(scale_at_local_t(lt))
        w = float(base_w) * s
        h = float(base_h) * s
        return (anchor_cx - (w / 2.0), anchor_cy - (h / 2.0))

    return pos

def anim_punch_rotate(
    clip: VideoClip,
    *,
    video_w: int,
    start_t: float,
    y_top: float,
    dur: float = 0.14,
    min_scale: float = 0.85,
    max_scale: float = 1.00,
    start_deg: float = 0.0,   # 0..359
    end_deg: float = 0.0,     # 0..359
    in_place: bool = True,
) -> VideoClip:
    """
    PUNCH ROTATE: quick scale punch + rotation from start_deg -> end_deg.
    """
    base_w, base_h = float(clip.w), float(clip.h)

    def scale_fn(local_t: float) -> float:
        if local_t <= 0.0:
            return max(0.01, float(min_scale))
        if local_t >= float(dur):
            return float(max_scale)
        u = _clamp01(local_t / float(dur))
        s = float(min_scale) + (float(max_scale) - float(min_scale)) * _ease_out_back(u, overshoot=1.2)
        return max(0.01, float(s))

    def rot_fn(local_t: float) -> float:
        if local_t <= 0.0:
            return float(start_deg % 360)
        if local_t >= float(dur):
            return float(end_deg % 360)
        u = _clamp01(local_t / float(dur))
        return float(start_deg + (end_deg - start_deg) * u)

    clip2 = clip.with_effects([vfx.Resize(scale_fn)])

    if hasattr(vfx, "Rotate"):
        try:
            clip2 = clip2.with_effects([vfx.Rotate(lambda t: rot_fn(max(0.0, float(t))), expand=False)])  # type: ignore
        except TypeError:
            clip2 = clip2.with_effects([vfx.Rotate(lambda t: rot_fn(max(0.0, float(t))))])  # type: ignore
    else:
        if hasattr(clip2, "fx") and hasattr(vfx, "rotate"):
            clip2 = clip2.fx(vfx.rotate, lambda t: rot_fn(max(0.0, float(t))))  # type: ignore

    if in_place:
        pos = _in_place_position(video_w, y_top, base_w, base_h, start_t, scale_fn)
        clip2 = clip2.with_position(pos)  # type: ignore

    return clip2  # type: ignore

# ---------- TEXT RENDER ----------

def create_subtitle_clip(text: str, cfg: Configuration) -> ImageClip:
    """Create white subtitle text with smooth black border + optional glow."""
    try:
        font = ImageFont.truetype(cfg.font_name, cfg.font_size)
    except Exception as e:
        raise RuntimeError(
            f"Could not load font '{cfg.font_name}'. Provide a valid font name/path (e.g. '/path/Impact.ttf')."
        ) from e

    text = (text or "").strip()

    tmp = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    draw_tmp = ImageDraw.Draw(tmp)
    bbox = draw_tmp.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    border = int(cfg.border_size)
    pad_side = border * 2 + int(cfg.pad_side_extra)
    pad_top = border * 2 + int(cfg.pad_top_extra)
    pad_bottom = border * 2 + int(cfg.pad_bottom_extra)

    w = int(tw + pad_side * 2)
    h = int(th + pad_top + pad_bottom)
    x = pad_side
    y = pad_top

    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))  # type: ignore
    draw = ImageDraw.Draw(img)

    if cfg.glow_enabled:
        glow_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))  # type: ignore
        glow_draw = ImageDraw.Draw(glow_layer)

        spread = int(cfg.glow_spread)
        for dx in range(-spread, spread + 1):
            for dy in range(-spread, spread + 1):
                glow_draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0, 255))

        glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(int(cfg.glow_radius)))

        r, g, b, a = glow_layer.split()
        a = a.point(lambda p: min(255, int(p * (cfg.glow_alpha / 255.0))))  # type: ignore
        glow_layer = Image.merge("RGBA", (r, g, b, a))

        img = Image.alpha_composite(img, glow_layer)
        draw = ImageDraw.Draw(img)

    if border > 0:
        r2 = border * border
        for dx in range(-border, border + 1):
            for dy in range(-border, border + 1):
                if dx == 0 and dy == 0:
                    continue
                if (dx * dx + dy * dy) > r2:
                    continue
                draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0, 255))

    draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))

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
    out: List[Dict] = []
    for s in subs:
        start = s.start.ordinal / 1000.0
        end = s.end.ordinal / 1000.0
        out.append(
            {
                "text": (s.text or "").strip(),
                "start": start,
                "end": end,
                "duration": max(0.0, end - start),
            }
        )
    return out


# ---------- MAIN ----------

def add_subtitles_to_video(
    input_video_path: str,
    srt_path: str,
    output_video_path: str,
    preview: bool = False,
    cfg: Optional[Configuration] = None,
):
    if cfg is None:
        cfg = Configuration()

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

        txt_clip = create_subtitle_clip(txt, cfg)

        # rotation steps: 10->20, 20->30, ... 360->10 ...
        start_deg = float(((i * 10) % 120) + 10)
        end_deg = float(start_deg + 10)

        txt_clip = anim_punch_rotate(
            txt_clip,
            video_w=video.w,
            start_t=seg["start"],
            y_top=cfg.y_position,
            dur=0.35,
            min_scale=0.90,
            max_scale=1.00,
            start_deg=start_deg,
            end_deg=end_deg,
            in_place=True,
        )

        # IMPORTANT: do NOT override position here; anim_punch_rotate(in_place=True) already anchors it.
        txt_clip = (
            txt_clip
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


if __name__ == "__main__":
    cfg = Configuration()
    input_file = "n8n/Downloads/Sakinah Labs/TestVideo.mp4"
    srt_file = "transcripts/audio_for_transcription.srt"

    outputFileName = "animation_rotation_steps"
    out = f"n8n/Testing/videoOuput/{outputFileName}.mp4"

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
