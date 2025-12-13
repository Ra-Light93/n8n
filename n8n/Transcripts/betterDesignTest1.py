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
    Cyberpunk red/blue neon with intense glow
    """
    return {
        "font_name": "Courier New",    # Monospace for tech look
        "font_size": "18",
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


def build_force_style(font_style: dict) -> str:
    font_name = font_style.get("font_name", "Arial")
    font_size = font_style.get("font_size", "28")
    # accept both font_color and primary_color
    font_color = font_style.get("font_color") or font_style.get("primary_color", "FFFFFF")
    outline_color = font_style.get("outline_color", "000000")
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


def burn_srt_with_blur(
    input_video_path: str,
    srt_file_path: str,
    output_video_path: str,
    blur_area: dict,
    font_style: dict,
    blur_strength: int = 20,
    log_file_path: str = "ffmpeg_output.log",
):
    """
    - Blurs a rectangular area in the video.
    - Burns SRT subtitles on top with simple styling.
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

    # Filter graph:
    # 1) Crop + blur region from original video => [blurred_box]
    # 2) Overlay blurred_box back onto original => [blurred_vid]
    # 3) Burn subtitles on blurred_vid          => [final_v]
    filter_complex = (
        f"[0:v]crop={w}:{h}:{x}:{y},"
        f"boxblur={blur_strength}[blurred_box];"
        f"[0:v][blurred_box]overlay={x}:{y}[blurred_vid];"
        f"[blurred_vid]subtitles='{srt_path_escaped}':"
        f"force_style='{force_style}'[final_v]"
    )

    cmd = [
        "ffmpeg",
        "-i", input_video_path,
        "-filter_complex", filter_complex,
        "-map", "[final_v]",  # use processed video
        "-map", "0:a?",       # copy audio if present
        "-c:a", "copy",
        "-y",                 # overwrite output
        output_video_path,
    ]

    print("Running ffmpeg...")
    print(" ".join(cmd))

    try:
        # Run ffmpeg and show progress directly in the terminal.
        # If something is wrong, you will see the error immediately.
        subprocess.run(cmd, check=True)
        print(f"Done: {output_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg failed with code {e.returncode}")


if __name__ == "__main__":
    # --- Simple example configuration ---

    input_file = "n8n/Downloads/Sakinah Labs/Test2.mp4"
    srt_file = "transcripts/audio_for_transcription.srt"
    output_file = "n8n/Results/test1.mp4"
    log_file = "ffmpeg_srt_burn.log"

    # Get video size only to define blur area easily.
    width, height = get_video_resolution(input_file)
    print(f"Video resolution: {width}x{height}")

    # Blur a strip at the bottom (full width, 130px high).
    area_to_blur = {"x": 0, "y": 1255, "w": width, "h": 150}

# use the new cyberpunk style
    subtitle_style = get_cyberpunk_style()

    burn_srt_with_blur(
        input_video_path=input_file,
        srt_file_path=srt_file,
        output_video_path=output_file,
        blur_area=area_to_blur,
        font_style=subtitle_style,
        blur_strength=20,
        log_file_path=log_file,
    )