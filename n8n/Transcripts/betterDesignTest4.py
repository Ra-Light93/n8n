import os
import subprocess


def get_video_resolution(path: str) -> tuple[int, int]:
    """
    Return (width, height) of the first video stream using ffprobe.
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0", path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"ffprobe failed for {path}")
    w_str, h_str = result.stdout.strip().split("x")
    return int(w_str), int(h_str)


def get_perfect_style(style_name: str = "cyberpunk_clean") -> dict:
    """
    PERFECT text styles that look amazing
    """
    styles = {
        "cyberpunk_clean": {
            "font_name": "Helvetica Neue Bold",
            "font_size": "20",
            "primary_color": "FFFFFF",    # White text
            "secondary_color": "00FFFF",  # Cyan accent
            "outline_color": "0066FF",    # Blue outline
            "back_color": "000000",       # Black shadow
            
            "bold": "1",
            "italic": "0",
            "spacing": "0",              # Clean spacing
            "scale_x": "100",
            "scale_y": "100",
            
            "outline": "3",              # Good outline for readability
            "shadow": "4",               # Shadow for depth
            "shadow_color": "0066FF",    # Blue shadow for neon effect
            
            "border_style": "1",
            "angle": "0",
            "alignment": "2",
            "margin_l": "25",
            "margin_r": "25",
            "margin_v": "90",            # Perfect bottom positioning
        },
        "neon_glow": {
            "font_name": "Arial Black",
            "font_size": "22",
            "primary_color": "00FFFF",   # Cyan text
            "secondary_color": "FF00FF", # Magenta accent
            "outline_color": "FFFFFF",   # White outline
            "back_color": "000000",
            
            "bold": "1",
            "italic": "0",
            "spacing": "-0.5",
            "scale_x": "102",
            "scale_y": "100",
            
            "outline": "5",              # Thick outline for glow
            "shadow": "8",               # Big shadow for glow effect
            "shadow_color": "0066FF",
            
            "border_style": "1",
            "angle": "0",
            "alignment": "2",
            "margin_l": "30",
            "margin_r": "30",
            "margin_v": "95",
        },
        "minimal_white": {
            "font_name": "Helvetica Neue",
            "font_size": "18",
            "primary_color": "FFFFFF",   # Pure white
            "secondary_color": "CCCCCC",
            "outline_color": "000000",   # Black outline for contrast
            "back_color": "000000",
            
            "bold": "0",                 # Not bold for minimal look
            "italic": "0",
            "spacing": "0",
            "scale_x": "100",
            "scale_y": "100",
            
            "outline": "1.5",            # Thin outline
            "shadow": "3",               # Small shadow
            "shadow_color": "000000",
            
            "border_style": "1",
            "angle": "0",
            "alignment": "2",
            "margin_l": "20",
            "margin_r": "20",
            "margin_v": "85",
        },
        "cyberpunk_red": {
            "font_name": "Impact",
            "font_size": "21",
            "primary_color": "FF3366",   # Pink-red
            "secondary_color": "FF9900", # Orange
            "outline_color": "660033",   # Dark red outline
            "back_color": "000000",
            
            "bold": "1",
            "italic": "0",
            "spacing": "-1",
            "scale_x": "101",
            "scale_y": "99",
            
            "outline": "4",
            "shadow": "6",
            "shadow_color": "660033",
            
            "border_style": "1",
            "angle": "0",
            "alignment": "2",
            "margin_l": "25",
            "margin_r": "25",
            "margin_v": "88",
        }
    }
    return styles.get(style_name, styles["cyberpunk_clean"])


def build_force_style(font_style: dict) -> str:
    font_name = font_style.get("font_name", "Arial")
    font_size = font_style.get("font_size", "28")
    font_color = font_style.get("font_color") or font_style.get("primary_color", "FFFFFF")
    outline_color = font_style.get("outline_color", "000000")
    shadow_color = font_style.get("shadow_color", "000000")
    outline = font_style.get("outline", "2")
    shadow = font_style.get("shadow", "1")
    alignment = font_style.get("alignment", "2")
    margin_l = font_style.get("margin_l", "20")
    margin_r = font_style.get("margin_r", "20")
    margin_v = font_style.get("margin_v", "40")

    return (
        f"FontName={font_name},"
        f"Fontsize={font_size},"
        f"PrimaryColour=&H{font_color},"
        f"OutlineColour=&H{outline_color},"
        f"BackColour=&H{shadow_color},"
        f"BorderStyle=1,"
        f"Outline={outline},"
        f"Shadow={shadow},"
        f"Alignment={alignment},"
        f"MarginL={margin_l},"
        f"MarginR={margin_r},"
        f"MarginV={margin_v}"
    )


def escape_path_for_filter(path: str) -> str:
    """
    Escape path for use inside ffmpeg filter (single-quoted).
    Very simple: escape single quotes by '\''.
    """
    return path.replace("'", r"'\''")


def burn_srt_with_enhanced_glow(
    input_video_path: str,
    srt_file_path: str,
    output_video_path: str,
    blur_area: dict,
    font_style: dict,
    blur_strength: int = 30,
    glow_color: str = "cyan",
    background_style: str = "gradient",
    log_file_path: str = "ffmpeg_output.log",
):
    """
    Enhanced version with beautiful text and background
    """
    x = blur_area.get("x", 0)
    y = blur_area.get("y", 0)
    w = blur_area.get("w", 0)
    h = blur_area.get("h", 0)

    force_style = build_force_style(font_style)
    srt_path_escaped = escape_path_for_filter(os.path.abspath(srt_file_path))
    
    # Define different background styles
    background_filters = {
        "cyan_glow": (
            f"colorchannelmixer="
            f"rr=0.4:rg=0.2:rb=0.4:"    # More cyan tint
            f"gr=0.2:gg=0.5:gb=0.3:"    
            f"br=0.4:bg=0.3:bb=0.7,"    # Strong blue component
            f"eq=brightness=0.15:contrast=1.3,"  # More contrast
            f"curves=r='0/0 0.3/0.4 1/1':g='0/0 0.3/0.5 1/1':b='0/0 0.3/0.8 1/1'"
        ),
        "purple_gradient": (
            f"colorchannelmixer="
            f"rr=0.5:rg=0.3:rb=0.2:"
            f"gr=0.2:gg=0.4:gb=0.4:"
            f"br=0.3:bg=0.3:bb=0.8,"
            f"eq=brightness=0.1:saturation=1.2,"
            f"geq="
            f"r='lerp(p(X,Y),200,0.3)':"
            f"g='lerp(p(X,Y),100,0.3)':"
            f"b='lerp(p(X,Y),255,0.7)'"
        ),
        "dark_fog": (
            f"noise=alls=20:allf=t,"     # Subtle animated noise
            f"colorbalance=rs=-0.3:gs=-0.2:bs=0.5,"  # Blue tint
            f"boxblur=15,"               # Fog effect
            f"eq=brightness=-0.05:contrast=1.2,"  # Darker
            f"colorchannelmixer=aa=0.9"  # Keep it opaque
        ),
        "simple_blur": (
            f"eq=brightness=0.05:contrast=1.1,"  # Slight brightness boost
            f"colorbalance=rs=-0.1:gs=-0.05:bs=0.15"  # Subtle blue tint
        )
    }
    
    bg_filter = background_filters.get(background_style, background_filters["cyan_glow"])
    
    # Build the filter
    filter_complex = (
        f"[0:v]crop={w}:{h}:{x}:{y},"
        f"boxblur={blur_strength}:{blur_strength},"
        f"{bg_filter}[final_bg];"
        
        f"[0:v][final_bg]overlay={x}:{y}[background];"
        f"[background]subtitles='{srt_path_escaped}':"
        f"force_style='{force_style}'[final_v]"
    )

    cmd = [
        "ffmpeg",
        "-i", input_video_path,
        "-filter_complex", filter_complex,
        "-map", "[final_v]",
        "-map", "0:a?",
        "-c:a", "copy",
        "-y",
        output_video_path,
    ]

    print(f"Running enhanced {glow_color} glow with {background_style} background...")
    print(" ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
        print(f"✓ Enhanced effect applied: {output_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed: {e.returncode}")


# ULTIMATE VERSION: Multiple effects combined
def burn_srt_with_ultimate_effect(
    input_video_path: str,
    srt_file_path: str,
    output_video_path: str,
    blur_area: dict,
    font_style: dict,
    blur_strength: int = 25,
    effect_preset: str = "cyberpunk_ultimate",
    log_file_path: str = "ffmpeg_output.log",
):
    """
    Ultimate effect combining multiple techniques
    """
    x = blur_area.get("x", 0)
    y = blur_area.get("y", 0)
    w = blur_area.get("w", 0)
    h = blur_area.get("h", 0)

    force_style = build_force_style(font_style)
    srt_path_escaped = escape_path_for_filter(os.path.abspath(srt_file_path))
    
    # Preset configurations
    presets = {
        "cyberpunk_ultimate": {
            "blur": 25,
            "color_tint": "colorbalance=rs=-0.2:gs=-0.1:bs=0.3,",
            "brightness": "eq=brightness=0.1:contrast=1.2:saturation=1.1,",
            "curves": "curves=r='0/0 0.4/0.5 1/1':g='0/0 0.4/0.6 1/1':b='0/0 0.3/0.7 1/1',",
            "scanlines": "split=2[blurred][scan];[scan]geq=r='255*mod(floor(Y/3),2)':g='255*mod(floor(Y/3),2)':b='255*mod(floor(Y/3),2)',colorchannelmixer=aa=0.08[scanlines];[blurred][scanlines]overlay[bg_with_scan];"
        },
        "matrix_style": {
            "blur": 30,
            "color_tint": "colorbalance=rs=-0.4:gs=0.3:bs=-0.4,",
            "brightness": "eq=brightness=-0.05:contrast=1.3:saturation=0.8,",
            "curves": "curves=r='0/0 0.2/0.3 1/1':g='0/0 0.3/0.8 1/1':b='0/0 0.2/0.3 1/1',",
            "scanlines": "split=2[blurred][scan];[scan]geq=r='255*mod(floor(Y/2),2)':g='255*mod(floor(Y/2),2)':b='255*mod(floor(Y/2),2)',colorchannelmixer=aa=0.12[scanlines];[blurred][scanlines]overlay[bg_with_scan];"
        },
        "synthwave_vibe": {
            "blur": 22,
            "color_tint": "colorbalance=rs=0.3:gs=-0.2:bs=0.3,",
            "brightness": "eq=brightness=0.15:contrast=1.15:saturation=1.3,",
            "curves": "curves=r='0/0 0.5/0.8 1/1':g='0/0 0.3/0.4 1/1':b='0/0 0.5/0.9 1/1',",
            "scanlines": ""
        }
    }
    
    preset = presets.get(effect_preset, presets["cyberpunk_ultimate"])
    
    # Build filter with preset
    filter_parts = [
        f"[0:v]crop={w}:{h}:{x}:{y},"
        f"boxblur={preset['blur']}:{preset['blur']},"
        f"{preset['color_tint']}"
        f"{preset['brightness']}"
        f"{preset['curves']}"
    ]
    
    if preset['scanlines']:
        filter_parts.append(preset['scanlines'])
        bg_stream = "bg_with_scan"
    else:
        filter_parts.append("[blurred]")
        bg_stream = "blurred"
    
    filter_parts.append(f"{bg_stream}[final_bg];")
    filter_parts.append(f"[0:v][final_bg]overlay={x}:{y}[background];")
    filter_parts.append(f"[background]subtitles='{srt_path_escaped}':")
    filter_parts.append(f"force_style='{force_style}'[final_v]")
    
    filter_complex = "".join(filter_parts)

    cmd = [
        "ffmpeg",
        "-i", input_video_path,
        "-filter_complex", filter_complex,
        "-map", "[final_v]",
        "-map", "0:a?",
        "-c:a", "copy",
        "-y",
        output_video_path,
    ]

    print(f"Running ULTIMATE {effect_preset} effect...")
    print(" ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
        print(f"✓ Ultimate effect applied: {output_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed: {e.returncode}")


if __name__ == "__main__":
    # Configuration
    input_file = "n8n/Downloads/Sakinah Labs/Test2.mp4"
    srt_file = "transcripts/audio_for_transcription.srt"
    output_file = "n8n/Results/test_perfect.mp4"
    log_file = "ffmpeg_perfect.log"

    width, height = get_video_resolution(input_file)
    print(f"📺 Video resolution: {width}x{height}")

    area_to_blur = {"x": 0, "y": 1255, "w": width, "h": 150}
    
    # Choose the PERFECT style for you:
    subtitle_style = get_perfect_style("cyberpunk_clean")  # Clean white text
    # subtitle_style = get_perfect_style("neon_glow")        # Glowing cyan text
    # subtitle_style = get_perfect_style("minimal_white")    # Minimal white
    # subtitle_style = get_perfect_style("cyberpunk_red")    # Red cyberpunk
    
    # Option 1: Enhanced glow (recommended)
    burn_srt_with_enhanced_glow(
        input_video_path=input_file,
        srt_file_path=srt_file,
        output_video_path=output_file,
        blur_area=area_to_blur,
        font_style=subtitle_style,
        blur_strength=28,
        glow_color="cyan",
        background_style="cyan_glow",  # Try: "cyan_glow", "purple_gradient", "dark_fog", "simple_blur"
        log_file_path=log_file,
    )
    
    # Option 2: Ultimate effect (more complex)
    # burn_srt_with_ultimate_effect(
    #     input_video_path=input_file,
    #     srt_file_path=srt_file,
    #     output_video_path=output_file.replace(".mp4", "_ultimate.mp4"),
    #     blur_area=area_to_blur,
    #     font_style=subtitle_style,
    #     blur_strength=25,
    #     effect_preset="cyberpunk_ultimate",  # Try: "cyberpunk_ultimate", "matrix_style", "synthwave_vibe"
    #     log_file_path=log_file,
    # )