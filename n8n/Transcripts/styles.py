def build_force_style(style: dict) -> str:
    """
    Build an ASS force_style string from a dict.

    Supported keys (dict):
      font_name, font_size,
      primary_color, secondary_color, outline_color, back_color,
      bold, italic, underline, strike_out,
      scale_x, scale_y, spacing, angle,
      border_style, outline, shadow,
      alignment, margin_l, margin_r, margin_v,
      encoding
    """

    # All supported mappings: dict_key -> ASS style key
    key_map = {
        "font_name":      "FontName",
        "font_size":      "Fontsize",
        "primary_color":  "PrimaryColour",
        "secondary_color":"SecondaryColour",
        "outline_color":  "OutlineColour",
        "back_color":     "BackColour",
        "bold":           "Bold",
        "italic":         "Italic",
        "underline":      "Underline",
        "strike_out":     "StrikeOut",
        "scale_x":        "ScaleX",
        "scale_y":        "ScaleY",
        "spacing":        "Spacing",     # < 0 = letters closer, > 0 = farther
        "angle":          "Angle",
        "border_style":   "BorderStyle",
        "outline":        "Outline",
        "shadow":         "Shadow",
        "alignment":      "Alignment",
        "margin_l":       "MarginL",
        "margin_r":       "MarginR",
        "margin_v":       "MarginV",
        "encoding":       "Encoding",
    }

    # Default values (you can change them)
    defaults = {
        "font_name":     "Arial",
        "font_size":     "28",
        "primary_color": "FFFFFF",
        "outline_color": "000000",
        "back_color":    "000000",
        "secondary_color": "FFFFFF",
        "bold":          "0",
        "italic":        "0",
        "underline":     "0",
        "strike_out":    "0",
        "scale_x":       "100",
        "scale_y":       "100",
        "spacing":       "0",     # set to "-1" or "-2" for tighter letters
        "angle":         "0",
        "border_style":  "1",
        "outline":       "2",
        "shadow":        "1",
        "alignment":     "2",
        "margin_l":      "20",
        "margin_r":      "20",
        "margin_v":      "40",
        "encoding":      "1",
    }

    merged = {**defaults, **style}
    parts = []

    for k, ass_k in key_map.items():
        val = merged.get(k)
        if val is None:
            continue

        # Color fields: ensure &H prefix
        if "color" in k:
            v = str(val)
            if not v.upper().startswith("&H"):
                v = "&H" + v
            parts.append(f"{ass_k}={v}")
        else:
            parts.append(f"{ass_k}={val}")

    return ",".join(parts)


# SUPER COOL NEON STYLE WITH OVERLAPPING LETTERS
def get_neon_style() -> dict:
    """
    Returns an awesome neon style with:
    - Electric blue/purple glow effect
    - Letters that overlap slightly for a compact look
    - Multiple shadow layers for depth
    - Special spacing for cool text effects
    """
    return {
        # Font choice - use a bold, modern font if available
        "font_name": "Impact",  # Great for bold neon look, or try "Arial Black", "Bauhaus 93"
        "font_size": "36",
        
        # NEON GLOW COLORS - Electric blue with purple outline
        "primary_color": "00FFFF",      # Cyan/blue neon center
        "secondary_color": "FF00FF",    # Magenta/purple secondary glow
        "outline_color": "4B0082",      # Deep indigo outer glow
        "back_color": "000000",         # Black background (makes neon pop)
        
        # Text styling
        "bold": "1",                    # Bold for thicker neon tubes
        "italic": "0",
        
        # LETTER OVERLAP EFFECT - negative spacing makes letters overlap
        "spacing": "-3.5",             # Negative value = letters overlap
        
        # Scale for slightly stretched neon look
        "scale_x": "105",              # Slightly wider
        "scale_y": "95",               # Slightly shorter
        
        # NEON GLOW EFFECTS - multiple layers of outline/shadow
        "outline": "8",                # Thick outline for glow effect
        "shadow": "10",                # Large shadow for depth
        
        # Border style 1 = outline + shadow
        "border_style": "1",
        
        # Text angle for dynamic look (slight tilt)
        "angle": "1.5",
        
        # Positioning - bottom center but with cool margins
        "alignment": "2",              # Bottom center
        "margin_l": "10",
        "margin_r": "10",
        "margin_v": "30",              # Closer to bottom for more space
    }


# ALTERNATIVE: CYBERPUNK RED NEON
def get_cyberpunk_style() -> dict:
    """
    Cyberpunk red/blue neon with intense glow
    """
    return {
        "font_name": "Courier New",    # Monospace for tech look
        "font_size": "32",
        "primary_color": "FF0033",     # Bright red
        "secondary_color": "0066FF",   # Electric blue
        "outline_color": "3300FF",     # Purple-blue
        "back_color": "000011",        # Very dark blue background
        
        "bold": "1",
        "italic": "0",
        "spacing": "-2.8",             # Overlapping letters
        "scale_x": "102",
        "scale_y": "98",
        
        # EXTREME GLOW
        "outline": "12",               # Very thick outline
        "shadow": "15",                # Huge shadow
        
        "border_style": "1",
        "angle": "0",                  # No angle for cyberpunk
        "alignment": "2",
        "margin_l": "15",
        "margin_r": "15",
        "margin_v": "35",
    }


# ALTERNATIVE: RAINBOW NEON WITH MULTI-LAYER
def get_rainbow_neon_style() -> dict:
    """
    Rainbow neon with multiple color layers
    """
    return {
        "font_name": "Comic Sans MS",  # Fun, rounded font
        "font_size": "34",
        "primary_color": "FF0000",     # Red
        "secondary_color": "00FF00",   # Green  
        "outline_color": "0000FF",     # Blue
        "back_color": "FFFFFF",        # White for maximum contrast
        
        "bold": "1",
        "italic": "0",
        "spacing": "-4.0",             # Strong overlap
        "scale_x": "108",
        "scale_y": "92",
        
        "outline": "6",
        "shadow": "8",
        
        "border_style": "1",
        "angle": "-2.0",               # Slight opposite tilt
        "alignment": "2",
        "margin_l": "5",
        "margin_r": "5",
        "margin_v": "25",
    }


# EXAMPLE USAGE in your main function:
if __name__ == "__main__":
    # Choose your style:
    subtitle_style = get_neon_style()           # Blue/Purple neon
    # subtitle_style = get_cyberpunk_style()    # Red/Blue cyberpunk
    # subtitle_style = get_rainbow_neon_style() # Rainbow colors
    
    # For even more extreme effects, you can override specific values:
    subtitle_style.update({
        "spacing": "-5.0",  # Even more overlap
        "shadow": "20",     # Mega shadow
        "outline": "10",    # Super thick outline
    })