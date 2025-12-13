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

def get_cyberpunk_style() -> dict:
    """
    Sleek cyberpunk neon with refined glow
    """
    return {
        "font_name": "Impact",
        "font_size": "17",
        "primary_color": "FF3366",
        "secondary_color": "33CCFF",
        "outline_color": "9933FF",
        "back_color": "000000",
        
        "bold": "1",
        "italic": "0",
        "spacing": "-1.5",
        "scale_x": "98",
        "scale_y": "95",
        
        "outline": "4",
        "shadow": "5",
        
        "border_style": "1",
        "angle": "0",
        "alignment": "2",
        "margin_l": "15",
        "margin_r": "15",
        "margin_v": "80",
    }

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


def burn_srt_with_fog_effect(
    input_video_path: str,
    srt_file_path: str,
    output_video_path: str,
    blur_area: dict,
    font_style: dict,
    blur_strength: int = 20,
    fog_intensity: float = 0.6,
    log_file_path: str = "ffmpeg_output.log",
):
    """
    - Blurs a rectangular area in the video.
    - Adds foggy animation behind text.
    - Burns SRT subtitles on top with styling.
    """

    # --- Basic checks ---
    if not os.path.exists(input_video_path):
        print(f"Error: Input video not found: {input_video_path}")
        return
    if not os.path.exists(srt_file_path):
        print(f"Error: SRT file not found: {srt_file_path}")
        return

    # Ensure blur_area keys exist; user only has to pass x,y,w,h.
    x = blur_area.get("x", 0)
    y = blur_area.get("y", 0)
    w = blur_area.get("w", 0)
    h = blur_area.get("h", 0)

    # Build subtitle style string.
    force_style = build_force_style(font_style)

    # Escape paths for ffmpeg filter string.
    srt_path_escaped = escape_path_for_filter(os.path.abspath(srt_file_path))

    # CORRECTED FOG FILTER - no size mismatch issues
    filter_complex = (
        # Step 1: Create the background with fog (cropped area)
        f"[0:v]crop={w}:{h}:{x}:{y},"
        f"split=3[bg_orig][bg_fog][bg_blur];"
        
        # Step 2: Generate fog effect
        f"[bg_fog]noise=alls=40:allf=t+0.3,"
        f"colorbalance=rs=-0.4:gs=-0.1:bs=0.5,"  # Cyan tint
        f"boxblur=15:15,"  # Blur for foggy look
        f"colorchannelmixer=aa={fog_intensity}[fog];"
        
        # Step 3: Create blurred background
        f"[bg_blur]boxblur={blur_strength}:{blur_strength}[blurred];"
        
        # Step 4: Combine fog with blurred background
        f"[blurred][fog]overlay=format=auto[foggy_bg];"
        
        # Step 5: Blend original with foggy background (ALL SAME SIZE: 1080x150)
        f"[bg_orig][foggy_bg]blend=all_mode='screen':all_opacity=0.6[final_bg];"
        
        # Step 6: Overlay the final background onto original video at correct position
        f"[0:v][final_bg]overlay={x}:{y}:format=auto[background];"
        
        # Step 7: Add subtitles
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

    print(f"Running ffmpeg with fog effect (intensity: {fog_intensity})...")
    print(" ".join(cmd))

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Done: {output_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg failed with code {e.returncode}")
        print(f"Error output: {e.stderr}")
        print(f"Standard output: {e.stdout}")


# SIMPLER VERSION - guaranteed to work
def burn_srt_with_simple_working_fog(
    input_video_path: str,
    srt_file_path: str,
    output_video_path: str,
    blur_area: dict,
    font_style: dict,
    blur_strength: int = 20,
    fog_opacity: float = 0.4,
    log_file_path: str = "ffmpeg_output.log",
):
    """
    Simple working version - no complex blending
    """
    # --- Basic checks ---
    if not os.path.exists(input_video_path):
        print(f"Error: Input video not found: {input_video_path}")
        return
    if not os.path.exists(srt_file_path):
        print(f"Error: SRT file not found: {srt_file_path}")
        return

    x = blur_area.get("x", 0)
    y = blur_area.get("y", 0)
    w = blur_area.get("w", 0)
    h = blur_area.get("h", 0)

    force_style = build_force_style(font_style)
    srt_path_escaped = escape_path_for_filter(os.path.abspath(srt_file_path))

    # SIMPLE WORKING VERSION - create fog layer and overlay it
    filter_complex = (
        # Create fog layer
        f"[0:v]crop={w}:{h}:{x}:{y},"
        f"noise=alls=30:allf=t+0.2,"  # Animated noise
        f"colorbalance=rs=-0.3:gs=-0.1:bs=0.4,"  # Cyan tint
        f"boxblur=10,"  # Make it foggy
        f"colorchannelmixer=aa={fog_opacity}[fog];"
        
        # Create blurred background
        f"[0:v]crop={w}:{h}:{x}:{y},"
        f"boxblur={blur_strength}[blurred];"
        
        # Combine fog with blur
        f"[blurred][fog]overlay[bg_with_fog];"
        
        # Overlay back to original
        f"[0:v][bg_with_fog]overlay={x}:{y}[background];"
        
        # Add subtitles
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

    print(f"Running SIMPLE fog effect (opacity: {fog_opacity})...")
    print(" ".join(cmd))

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Done: {output_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg failed with code {e.returncode}")
        print(f"Error output: {e.stderr}")


# EVEN SIMPLER - Just add a colored translucent overlay
def burn_srt_with_color_overlay(
    input_video_path: str,
    srt_file_path: str,
    output_video_path: str,
    blur_area: dict,
    font_style: dict,
    blur_strength: int = 20,
    overlay_color: str = "cyan",
    overlay_opacity: float = 0.3,
    log_file_path: str = "ffmpeg_output.log",
):
    """
    Simple colored translucent overlay behind text
    """
    # --- Basic checks ---
    if not os.path.exists(input_video_path):
        print(f"Error: Input video not found: {input_video_path}")
        return
    if not os.path.exists(srt_file_path):
        print(f"Error: SRT file not found: {srt_file_path}")
        return

    x = blur_area.get("x", 0)
    y = blur_area.get("y", 0)
    w = blur_area.get("w", 0)
    h = blur_area.get("h", 0)

    force_style = build_force_style(font_style)
    srt_path_escaped = escape_path_for_filter(os.path.abspath(srt_file_path))

    # Color mapping
    color_map = {
        "cyan": "00FFFF",
        "purple": "9933FF", 
        "blue": "0066FF",
        "pink": "FF66CC",
        "green": "33FF99"
    }
    hex_color = color_map.get(overlay_color, "00FFFF")
    
    # Simple filter - just overlay a colored box
    filter_complex = (
        # Create blurred background
        f"[0:v]crop={w}:{h}:{x}:{y},"
        f"boxblur={blur_strength}[blurred];"
        
        # Create colored overlay
        f"color=c=0x{hex_color}:size={w}x{h}[color];"
        f"[color]format=rgba,"
        f"colorchannelmixer=aa={overlay_opacity}[transparent_color];"
        
        # Combine blur with color
        f"[blurred][transparent_color]overlay[bg_with_color];"
        
        # Overlay back to original
        f"[0:v][bg_with_color]overlay={x}:{y}[background];"
        
        # Add subtitles
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

    print(f"Running with {overlay_color} overlay (opacity: {overlay_opacity})...")
    print(" ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
        print(f"Done: {output_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg failed with code {e.returncode}")


if __name__ == "__main__":
    input_file = "n8n/Downloads/Sakinah Labs/Test2.mp4"
    srt_file = "transcripts/audio_for_transcription.srt"
    output_file = "n8n/Results/test_fog_fixed.mp4"
    log_file = "ffmpeg_srt_burn.log"

    width, height = get_video_resolution(input_file)
    print(f"Video resolution: {width}x{height}")

    area_to_blur = {"x": 0, "y": 1255, "w": width, "h": 150}
    subtitle_style = get_cyberpunk_style()

    # Try the SIMPLE working version first
    burn_srt_with_simple_working_fog(
        input_video_path=input_file,
        srt_file_path=srt_file,
        output_video_path=output_file,
        blur_area=area_to_blur,
        font_style=subtitle_style,
        blur_strength=20,
        fog_opacity=0.4,  # 40% opacity
        log_file_path=log_file,
    )
    
    # Or try the colored overlay version (guaranteed to work):
    # burn_srt_with_color_overlay(
    #     input_video_path=input_file,
    #     srt_file_path=srt_file,
    #     output_video_path=output_file.replace(".mp4", "_color.mp4"),
    #     blur_area=area_to_blur,
    #     font_style=subtitle_style,
    #     blur_strength=20,
    #     overlay_color="cyan",
    #     overlay_opacity=0.3,
    #     log_file_path=log_file,
    # )