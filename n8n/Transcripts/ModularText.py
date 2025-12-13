# ---------- IMPORTS ----------
import numpy as np
from typing import Tuple
from PIL import Image, ImageDraw, ImageFont
from moviepy import ImageClip

# ---------- GLOBAL CONFIG (IMMER HIER) ----------

# Text
FONT_NAME = "Impact"
FONT_SIZE = 110

TEXT_COLOR = (255, 255, 255, 255)     # RGBA
OUTLINE_COLOR = (0, 0, 0, 255)
OUTLINE_PX = 2

# Layout
TEXT_PADDING_X = 16
TEXT_PADDING_Y = 16

# ---------- TEXT DRAW FUNCTION ----------

def ZeichnerText(
    text: str,
    position: Tuple[int, int],
    start_s: float,
    duration_s: float,
) -> ImageClip:
    """
    Zeichnet Text ins Video.
    Keine Logik, keine Effekte – nur Zeichnen + Platzieren.
    """

    # font
    try:
        font = ImageFont.truetype(FONT_NAME, FONT_SIZE)
    except Exception:
        font = ImageFont.load_default()

    # measure text
    tmp = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    dtmp = ImageDraw.Draw(tmp)
    l, t, r, b = dtmp.textbbox((0, 0), text, font=font)
    tw, th = r - l, b - t

    w = tw + TEXT_PADDING_X * 2
    h = th + TEXT_PADDING_Y * 2

    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    x = TEXT_PADDING_X
    y = TEXT_PADDING_Y

    # outline
    for dx in range(-OUTLINE_PX, OUTLINE_PX + 1):
        for dy in range(-OUTLINE_PX, OUTLINE_PX + 1):
            if dx == 0 and dy == 0:
                continue
            draw.text((x + dx, y + dy), text, font=font, fill=OUTLINE_COLOR)

    # main text
    draw.text((x, y), text, font=font, fill=TEXT_COLOR)

    clip = ImageClip(np.array(img), transparent=True)
    return (
        clip
        .with_position(position)
        .with_start(start_s)
        .with_duration(duration_s)
    )