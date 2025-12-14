import os
import pysrt
import numpy as np
from moviepy import VideoFileClip, TextClip, CompositeVideoClip, ImageClip, VideoClip, vfx
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import argparse
import math


# =========================
# CONFIG (ALL VARIABLES HERE)
# =========================

@dataclass(frozen=True)
class Configuration:
    # rectangle to cover old text (ONLY y + size adjustable)
    cover_y: int = 1500
    cover_w: int = 840
    cover_h: int = 260
    
    # corner styling (NEW ENHANCED OPTIONS)
    corner_radius: int = 25                     # base corner radius
    corner_style: str = "rounded"               # "rounded", "soft", "chamfered", "puffy"
    
    # visual enhancements
    gradient_enabled: bool = True               # subtle gradient effect
    gradient_direction: str = "vertical"        # "vertical", "horizontal", "diagonal"
    gradient_start_opacity: int = 245           # top/left opacity
    gradient_end_opacity: int = 230             # bottom/right opacity
    
    # border styling (subtle)
    border_enabled: bool = True
    border_width: int = 2
    border_color: Tuple[int, int, int] = (30, 30, 30)  # very dark gray
    border_opacity: int = 100
    
    # inner glow / highlight
    inner_glow_enabled: bool = True
    inner_glow_color: Tuple[int, int, int] = (60, 60, 60)  # soft inner highlight
    inner_glow_size: int = 2
    inner_glow_opacity: int = 80
    
    # horizontal anchor (fixed): exact center of the video
    # 0.5 means: center of overlay at 50% of video width
    anchor_x_ratio: float = 0.5

    # cover color + opacity
    cover_color: Tuple[int, int, int] = (15, 15, 15)   # softer black
    cover_opacity: int = 240                         # 0..255

    # optional: soften fill slightly (subtle "matte" look)
    blur_enabled: bool = True
    blur_radius: int = 1
    blur_padding: int = 6            # extra padding so blur is not clipped

    # outside shadow (looks clean + professional)
    shadow_enabled: bool = True
    shadow_color: Tuple[int, int, int] = (0, 0, 0)   # RGB
    shadow_opacity: int = 180                        # 0..255
    shadow_blur: int = 25                            # px (softer shadow)
    shadow_spread: int = 4                           # px (subtler spread)
    shadow_offset: Tuple[int, int] = (0, 6)          # (dx, dy) in px
    
    # reflection effect (subtle)
    reflection_enabled: bool = True
    reflection_height: int = 10
    reflection_opacity: int = 40

    # time range (set to full video by default)
    cover_start: float = 0.0
    cover_end: Optional[float] = None   # None => till end


# =========================
# ENHANCED VISUAL FUNCTIONS
# =========================

def create_gradient_mask(width: int, height: int, direction: str = "vertical") -> Image.Image:
    """Create a gradient mask for smooth opacity transition."""
    mask = Image.new("L", (width, height), 255)
    
    for y in range(height):
        for x in range(width):
            if direction == "vertical":
                # Vertical gradient (top to bottom)
                value = int(255 * (height - y) / height)
            elif direction == "horizontal":
                # Horizontal gradient (left to right)
                value = int(255 * (width - x) / width)
            elif direction == "diagonal":
                # Diagonal gradient (top-left to bottom-right)
                value = int(255 * (width - x + height - y) / (width + height))
            else:
                value = 255
            
            mask.putpixel((x, y), value)
    
    return mask


def create_enhanced_rounded_rectangle(width: int, height: int, radius: int, 
                                     style: str = "rounded") -> Image.Image:
    """Create an enhanced rectangle with different corner styles."""
    if style == "soft":
        # Super soft corners (larger radius)
        rect = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(rect)
        radius = min(radius * 1.5, min(width, height) // 2)          # type: ignore
        draw.rounded_rectangle([(0, 0), (width - 1, height - 1)], radius=radius, fill=(255, 255, 255, 255))
        
    elif style == "chamfered":
        # Chamfered corners (angled)
        rect = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(rect)
        
        # Draw main rectangle
        draw.rectangle([(radius, 0), (width - radius - 1, height - 1)], fill=(255, 255, 255, 255))
        draw.rectangle([(0, radius), (width - 1, height - radius - 1)], fill=(255, 255, 255, 255))
        
        # Draw chamfered corners
        for corner in [(0, 0), (width - radius, 0), (0, height - radius), (width - radius, height - radius)]:
            x, y = corner
            draw.polygon([
                (x, y + radius),
                (x + radius, y),
                (x + radius, y + radius)
            ], fill=(255, 255, 255, 255))
        
    elif style == "puffy":
        # Puffy/cloud-like corners
        rect = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(rect)
        
        # Create a more organic shape
        points = []
        steps = 8
        
        # Top edge with slight wave
        for i in range(steps + 1):
            x = i * width / steps
            y = math.sin(i * math.pi / steps) * 3  # subtle wave
            points.append((x, y))
        
        # Right edge
        for i in range(steps + 1):
            x = width - 1
            y = i * height / steps
            points.append((x, y))
        
        # Bottom edge with slight wave
        for i in range(steps, -1, -1):
            x = i * width / steps
            y = height - 1 - math.sin(i * math.pi / steps) * 3
            points.append((x, y))
        
        # Left edge
        for i in range(steps, -1, -1):
            x = 0
            y = i * height / steps
            points.append((x, y))
        
        draw.polygon(points, fill=(255, 255, 255, 255))
        
    else:  # "rounded" - default
        rect = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(rect)
        draw.rounded_rectangle([(0, 0), (width - 1, height - 1)], radius=radius, fill=(255, 255, 255, 255))
    
    return rect


def add_inner_glow(base_image: Image.Image, glow_color: Tuple[int, int, int], 
                  size: int = 2, opacity: int = 80) -> Image.Image:
    """Add a subtle inner glow effect."""
    if size <= 0:
        return base_image.copy()
    
    # Create glow layer
    width, height = base_image.size
    glow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(glow)
    
    # Draw slightly smaller rectangle for inner glow
    draw.rectangle([(size, size), (width - size - 1, height - size - 1)], 
                   fill=(*glow_color, opacity))
    
    # Composite with base image using alpha
    result = base_image.copy()
    result.alpha_composite(glow)
    
    return result


def add_reflection(base_image: Image.Image, height: int = 10, opacity: int = 40) -> Image.Image:
    """Add a subtle reflection effect at the bottom."""
    if height <= 0:
        return base_image.copy()
    
    width, img_height = base_image.size
    
    # Create reflection mask
    reflection = Image.new("RGBA", (width, img_height + height), (0, 0, 0, 0))
    
    # Paste original image
    reflection.paste(base_image, (0, 0))
    
    # Create gradient for reflection
    for y in range(height):
        alpha = int(opacity * (height - y) / height)
        if alpha > 0:
            # Get bottom row of original image
            row = base_image.crop((0, img_height - 1, width, img_height))
            # Apply fading alpha
            row_alpha = row.getchannel('A')
            new_alpha = Image.new('L', row.size, alpha)
            row.putalpha(ImageChops.multiply(row_alpha, new_alpha))          # type: ignore
            reflection.paste(row, (0, img_height + y), row)
    
    return reflection


# =========================
# ENHANCED COVER CLIP
# =========================

def create_shadowed_cover_clip(video_w: int, video_h: int, cfg: Configuration) -> ImageClip:
    """
    Creates an enhanced RGBA ImageClip with beautiful styling.
    """
    w = int(cfg.cover_w)
    h = int(cfg.cover_h)
    y = int(cfg.cover_y)
    radius = int(cfg.corner_radius)
    
    # Ensure radius is appropriate
    max_radius = min(w, h) // 2
    radius = min(radius, max_radius)

    # Fixed x: center at anchor_x_ratio of the video width
    cx = int(float(video_w) * float(cfg.anchor_x_ratio))
    x = int(cx - (w / 2))

    # Clamp to video bounds
    x = max(0, min(x, max(0, video_w - 1)))
    y = max(0, min(y, max(0, video_h - 1)))
    w = max(1, min(w, max(1, video_w - x)))
    h = max(1, min(h, max(1, video_h - y)))

    # Calculate padding
    sb = max(0, int(cfg.shadow_blur))
    ss = max(0, int(cfg.shadow_spread))
    dx, dy = (int(cfg.shadow_offset[0]), int(cfg.shadow_offset[1]))
    reflection_pad = cfg.reflection_height if cfg.reflection_enabled else 0
    
    pad = sb + ss + max(abs(dx), abs(dy)) + int(cfg.blur_padding) + 2 + reflection_pad

    canvas_w = w + pad * 2
    canvas_h = h + pad * 2 + reflection_pad

    # Base transparent canvas
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    
    # --- Enhanced shadow with rounded corners ---
    if cfg.shadow_enabled:
        sr, sg, sbc = cfg.shadow_color
        sa = int(max(0, min(255, cfg.shadow_opacity)))
        
        # Create shadow shape
        shadow_width = w + (ss * 2)
        shadow_height = h + (ss * 2)
        shadow_shape = create_enhanced_rounded_rectangle(
            shadow_width, 
            shadow_height, 
            radius + ss, 
            cfg.corner_style
        )
        
        # Colorize shadow
        shadow_colored = Image.new("RGBA", (shadow_width, shadow_height), (sr, sg, sbc, sa))
        shadow_mask = shadow_shape.getchannel('A')
        shadow_colored.putalpha(shadow_mask)
        
        # Apply blur for soft shadow
        if sb > 0:
            shadow_colored = shadow_colored.filter(ImageFilter.GaussianBlur(sb))
        
        # Position shadow with offset
        shadow_x = pad - ss + dx
        shadow_y = pad - ss + dy
        
        canvas.alpha_composite(shadow_colored, (shadow_x, shadow_y))

    # --- Create main cover rectangle with enhanced styling ---
    r, g, b = cfg.cover_color
    
    # Create shape mask
    shape_mask = create_enhanced_rounded_rectangle(w, h, radius, cfg.corner_style)
    
    # Create base color layer
    if cfg.gradient_enabled:
        # Create gradient fill
        base_layer = Image.new("RGBA", (w, h), (r, g, b, 255))
        
        # Create gradient mask
        gradient_mask = create_gradient_mask(w, h, cfg.gradient_direction)
        
        # Adjust opacity based on gradient
        for y_pos in range(h):
            for x_pos in range(w):
                pixel = base_layer.getpixel((x_pos, y_pos))
                mask_val = gradient_mask.getpixel((x_pos, y_pos))
                
                # Calculate opacity based on gradient mask
                opacity_range = cfg.gradient_start_opacity - cfg.gradient_end_opacity
                target_opacity = cfg.gradient_end_opacity + (mask_val * opacity_range / 255)     # type: ignore
                target_opacity = max(0, min(255, int(target_opacity)))
                
                base_layer.putpixel((x_pos, y_pos), (*pixel[:3], target_opacity))                # type: ignore
    else:
        # Solid fill
        base_layer = Image.new("RGBA", (w, h), (r, g, b, cfg.cover_opacity))
    
    # Apply shape mask
    base_alpha = base_layer.getchannel('A')
    shape_alpha = shape_mask.getchannel('A')
    combined_alpha = ImageChops.multiply(base_alpha, shape_alpha)                                # type: ignore
    base_layer.putalpha(combined_alpha)
    
    # Apply inner glow
    if cfg.inner_glow_enabled:
        base_layer = add_inner_glow(
            base_layer, 
            cfg.inner_glow_color, 
            cfg.inner_glow_size, 
            cfg.inner_glow_opacity
        )
    
    # Apply blur if enabled
    if cfg.blur_enabled and cfg.blur_radius > 0:
        base_layer = base_layer.filter(ImageFilter.GaussianBlur(cfg.blur_radius))
    
    # Add border
    if cfg.border_enabled and cfg.border_width > 0:
        border_draw = ImageDraw.Draw(base_layer)
        br, bg, bb = cfg.border_color
        border_alpha = cfg.border_opacity
        
        # Draw border inside the rectangle
        border_rect = [
            (cfg.border_width // 2, cfg.border_width // 2),
            (w - cfg.border_width // 2 - 1, h - cfg.border_width // 2 - 1)
        ]
        
        # Create border with rounded corners
        border_draw.rounded_rectangle(
            border_rect,
            radius=max(0, radius - cfg.border_width // 2),
            outline=(br, bg, bb, border_alpha),
            width=cfg.border_width
        )
    
    # Add reflection if enabled
    if cfg.reflection_enabled and cfg.reflection_height > 0:
        base_layer = add_reflection(base_layer, cfg.reflection_height, cfg.reflection_opacity)
        paste_y = pad
    else:
        paste_y = pad
    
    # Paste onto canvas
    canvas.alpha_composite(base_layer, (pad, paste_y))

    arr = np.array(canvas)

    # Position the clip
    clip = ImageClip(arr, transparent=True).with_position((x - pad, y - pad))
    return clip


# =========================
# MAIN FUNCTION
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

    print("Creating enhanced shadowed rectangle cover...")
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
# TEST RUN WITH ENHANCED STYLING
# =========================

if __name__ == "__main__":
    input_file = "n8n/Downloads/Sakinah Labs/TestVideo.mp4"
    output_file = "n8n/Testing/videoOuput/covered_old_text2.mp4"

    cfg = Configuration(
        cover_y=1000,
        cover_w=920,
        cover_h=180,
        corner_radius=25,
        corner_style="soft",  # try: "rounded", "soft", "chamfered", "puffy"
        
        # Visual enhancements
        gradient_enabled=True,
        gradient_direction="vertical",
        gradient_start_opacity=250,
        gradient_end_opacity=235,
        
        border_enabled=True,
        border_width=1,
        border_color=(40, 40, 40),
        border_opacity=120,
        
        inner_glow_enabled=True,
        inner_glow_color=(50, 50, 50),
        inner_glow_size=1,
        inner_glow_opacity=60,
        
        anchor_x_ratio=0.5,

        cover_color=(18, 18, 18),  # softer black
        cover_opacity=245,

        blur_enabled=True,
        blur_radius=0.8,    # type: ignore
        blur_padding=10,

        shadow_enabled=True,
        shadow_color=(0, 0, 0),
        shadow_opacity=200,  # slightly stronger for better depth
        shadow_blur=28,
        shadow_spread=3,     # subtler spread
        shadow_offset=(0, 8),
        
        reflection_enabled=True,
        reflection_height=8,
        reflection_opacity=30,

        cover_start=0.0,
        cover_end=None,
    )

    cover_old_text(input_file, output_file, cfg)