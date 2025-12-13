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


class GlowStyle(Enum):
    NEON_CYAN = "neon_cyan"
    NEON_PURPLE = "neon_purple"


class AnimationStyle(Enum):
    FADE_IN = "fade_in"


@dataclass
class SubtitleConfig:
    font_name: str = "Arial"
    font_size: int = 80
    primary_color: str = "#FFFFFF"
    glow_color: str = "#00FFFF"
    glow_size: int = 20
    glow_intensity: float = 0.8
    position: str = "center"
    margin: int = 50
    background_color: Optional[str] = "#000000"
    background_opacity: float = 0.4
    animation: AnimationStyle = AnimationStyle.FADE_IN
    duration: float = 3.0


class MoviePySubtitleEngine:
    def __init__(self):
        self.styles = {
            GlowStyle.NEON_CYAN: SubtitleConfig(
                font_name="Impact",
                glow_color="#00FFFF",
                background_color="#000033",
            ),
            GlowStyle.NEON_PURPLE: SubtitleConfig(
                font_name="Arial Black",
                glow_color="#FF00FF",
                background_color="#220022",
            ),
        }

    # ---------- TEXT + GLOW ----------

    def _load_font(self, name: str, size: int) -> ImageFont.FreeTypeFont:
        paths = [
            name,
            f"/System/Library/Fonts/{name}.ttf",
            f"/System/Library/Fonts/{name}.ttc",
            f"/Library/Fonts/{name}.ttf",
        ]
        for p in paths:
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
        return ImageFont.load_default()

    def create_text_with_glow(
        self, text: str, config: SubtitleConfig, frame_size: Tuple[int, int]
    ) -> ImageClip:
        from PIL import ImageColor

        scale = 2
        font = self._load_font(config.font_name, config.font_size * scale)
        pad = config.glow_size * 2 + 20

        # measure
        tmp = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        draw_tmp = ImageDraw.Draw(tmp)
        bbox = draw_tmp.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

        w = tw + pad * 2
        h = th + pad * 2
        x = (w - tw) // 2
        y = (h - th) // 2

        # base image (transparent)
        base = Image.new("RGBA", (w, h), (0, 0, 0, 0))

        # shadow (black)
        shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        sdraw = ImageDraw.Draw(shadow)
        sdraw.text((x + 3, y + 3), text, font=font, fill=(0, 0, 0, 200))
        shadow = shadow.filter(ImageFilter.GaussianBlur(3))
        base = Image.alpha_composite(base, shadow)

        # glow (white)
        glow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        gdraw = ImageDraw.Draw(glow)
        gc = (255, 255, 255)
        for i in range(config.glow_size, 0, -1):
            alpha = int(255 * config.glow_intensity * (i / config.glow_size))
            col = (*gc, alpha)
            for dx in [-i, 0, i]:
                for dy in [-i, 0, i]:
                    if dx or dy:
                        gdraw.text((x + dx, y + dy), text, font=font, fill=col)
        glow = glow.filter(ImageFilter.GaussianBlur(radius=config.glow_size))
        base = Image.alpha_composite(base, glow)

        # main text (pure white)
        tdraw = ImageDraw.Draw(base)
        tdraw.text((x, y), text, font=font, fill=(255, 255, 255, 255))

        # downscale
        base = base.resize((w // scale, h // scale), Image.Resampling.LANCZOS)
        arr = np.array(base)

        # ImageClip with alpha + fade-in
        clip = ImageClip(arr, transparent=True).with_duration(config.duration)
        clip = clip.with_effects([vfx.FadeIn(duration=0.3)])

        return clip

    # ---------- SRT ----------

    def parse_srt_file(self, srt_path: str) -> List[Dict]:
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

    # ---------- BACKGROUND BAR ----------

    def _make_background_bar(
        self, text_clip: ImageClip, config: SubtitleConfig
    ) -> ImageClip:
        from PIL import ImageColor

        pad_x = 40
        pad_y = 25
        bw = text_clip.w + pad_x
        bh = text_clip.h + pad_y

        bg = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
        draw = ImageDraw.Draw(bg)
        rgb = ImageColor.getrgb(config.background_color)
        alpha = int(255 * config.background_opacity)
        draw.rounded_rectangle(
            [0, 0, bw, bh], radius=20, fill=(*rgb, alpha)
        )

        arr = np.array(bg)
        return ImageClip(arr, transparent=True).with_duration(text_clip.duration)

    # ---------- MAIN FUNCTION ----------

    def add_subtitles_to_video(
        self,
        input_video_path: str,
        srt_path: str,
        output_video_path: str,
        glow_style: GlowStyle = GlowStyle.NEON_CYAN,
        position: str = "bottom",
        preview: bool = False,
    ) -> str:
        print(f"Loading video: {input_video_path}")
        video = VideoFileClip(input_video_path)

        print(f"Parsing SRT: {srt_path}")
        segments = self.parse_srt_file(srt_path)

        config = self.styles.get(glow_style, self.styles[GlowStyle.NEON_CYAN])
        config.position = position

        clips = [video]

        for i, seg in enumerate(segments):
            print(f"Subtitle {i+1}/{len(segments)}: {seg['text'][:40]!r}")
            txt = self.create_text_with_glow(seg["text"], config, video.size)

            # vertical position
            if position == "top":
                y = config.margin
            elif position == "center":
                y = 1260
            else:  # bottom
                y = video.h - config.margin - txt.h

            # position + timing
            txt = txt.with_position(("center", y))
            txt = txt.with_start(seg["start"])
            txt = txt.with_duration(seg["duration"])

            # background behind text
            if config.background_color and config.background_opacity > 0:
                bg = self._make_background_bar(txt, config)
                bg = bg.with_position(("center", y))
                bg = bg.with_start(seg["start"])
                bg = bg.with_duration(seg["duration"])
                clips.append(bg)

            clips.append(txt)

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


# Example usage
if __name__ == "__main__":
    engine = MoviePySubtitleEngine()
    input_file = "n8n/Downloads/Sakinah Labs/Test2.mp4"
    srt_file = "transcripts/audio_for_transcription.srt"
    output_file = "n8n/Results/test_moviepy_neon.mp4"

    engine.add_subtitles_to_video(
        input_video_path=input_file,
        srt_path=srt_file,
        output_video_path=output_file,
        glow_style=GlowStyle.NEON_CYAN,
        position="center",
        preview=True,
    )

