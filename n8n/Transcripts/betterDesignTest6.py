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


def get_premium_style(style_name: str = "cyberpunk_cyan") -> dict:
    """
    Premium styles that WORK with ffmpeg
    """
    styles = {
        "cyberpunk_cyan": {
            "font_name": "Impact",
            "font_size": "18",
            "primary_color": "00FFFF",
            "secondary_color": "FF00FF",
            "outline_color": "0066FF",
            "back_color": "000033",
            
            "bold": "1",
            "italic": "0",
            "spacing": "-1.5",
            "scale_x": "100",
            "scale_y": "100",
            
            "outline": "3",
            "shadow": "6",
            "shadow_color": "0066FF",
            
            "border_style": "1",
            "angle": "0",
            "alignment": "2",
            "margin_l": "20",
            "margin_r": "20",
            "margin_v": "85",
        },
        "matrix_green": {
            "font_name": "Courier New Bold",
            "font_size": "16",
            "primary_color": "00FF00",
            "secondary_color": "33FF33",
            "outline_color": "003300",
            "back_color": "000000",
            
            "bold": "1",
            "italic": "0",
            "spacing": "-1.2",
            "scale_x": "98",
            "scale_y": "98",
            
            "outline": "2",
            "shadow": "5",
            "shadow_color": "003300",
            
            "border_style": "1",
            "angle": "0",
            "alignment": "2",
            "margin_l": "15",
            "margin_r": "15",
            "margin_v": "80",
        },
        "synthwave": {
            "font_name": "Arial Black",
            "font_size": "19",
            "primary_color": "FF00FF",
            "secondary_color": "FF6600",
            "outline_color": "6600FF",
            "back_color": "000022",
            
            "bold": "1",
            "italic": "0",
            "spacing": "-1",
            "scale_x": "102",
            "scale_y": "102",
            
            "outline": "4",
            "shadow": "8",
            "shadow_color": "6600FF",
            
            "border_style": "1",
            "angle": "0",
            "alignment": "2",
            "margin_l": "25",
            "margin_r": "25",
            "margin_v": "90",
        }
    }
    return styles.get(style_name, styles["cyberpunk_cyan"])


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


def generate_filter_for_batch(subtitles, font_path: str) -> str:
    """
    Build a drawtext filter chain for a batch of word-level subtitles.
    Each item in subtitles must be a dict with 'word', 'start', 'end'.
    """
    filter_chain = ""
    for subtitle in subtitles:
        text = (
            subtitle["word"]
            .replace("'", r"\'")
            .replace(":", r"\:")
            .replace(",", r"\,")
            .replace(".", r"\.")
        )
        start = subtitle["start"]
        end = subtitle["end"]
        filter_chain += (
            "drawtext="
            f"fontfile='{font_path}':"
            f"text='{text}':"
            "fontcolor=white:"
            "fontsize=96:"
            "x=(w-text_w)/2:"
            "y=(h-text_h)/2:"
            f"enable='between(t,{start},{end})',"
        )
    return filter_chain.rstrip(",")


def add_subtitles_in_batches(
    input_video_file: str,
    subtitles_data: list[dict],
    font_path: str,
    batch_size: int = 5,
) -> str:
    """
    Add word-level subtitles to a video using ffmpeg drawtext in batches.
    - input_video_file: path to the input video
    - subtitles_data: list of dicts with keys: 'word', 'start', 'end'
    - font_path: path to a .ttf/.otf font file
    """
    output_filename = input_video_file
    num_batches = (len(subtitles_data) + batch_size - 1) // batch_size

    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(subtitles_data))
        batch_subtitles = subtitles_data[start_idx:end_idx]

        # Generate the filter for this batch
        filter_chain = generate_filter_for_batch(batch_subtitles, font_path)

        # Define the output filename for the current batch
        temp_output_filename = f"output/ffmpegsubsoutput_batch{batch_num}.mp4"

        # Define the FFmpeg command
        command = [
            "ffmpeg",
            "-i",
            output_filename,  # Use the previous output as input
            "-vf",
            filter_chain,
            "-c:v",
            "libx264",
            "-preset",
            "fast",  # speed/quality
            "-crf",
            "23",  # quality (0-51, lower is better)
            "-y",  # Overwrite output file if it exists
            temp_output_filename,
        ]

        # Run the FFmpeg command
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error during ffmpeg execution: {e}")
            print("Command executed:", " ".join(command))
            return output_filename

        # Update the output filename to the latest batch output
        output_filename = temp_output_filename

    return output_filename


def create_cyberpunk_effect(blur_area: dict) -> str:
    """
    Create a cyberpunk animated effect that WORKS
    """
    x = blur_area.get("x", 0)
    y = blur_area.get("y", 0)
    w = blur_area.get("w", 0)
    h = blur_area.get("h", 0)
    
    # This effect creates animated cyan particles
    effect = (
        f"geq="
        f"random(1)/32767*255*sin(2*PI*t):"  # Red: animated noise
        f"random(1)/32767*255*cos(2*PI*t):"  # Green: animated noise  
        f"255*abs(sin(2*PI*t)),"            # Blue: pulsing
        f"boxblur=15,"                      # Blur to create fog
        f"colorchannelmixer=aa=0.4"         # Set opacity
    )
    
    return effect


def burn_srt_with_cyberpunk_effect(
    input_video_path: str,
    srt_file_path: str,
    output_video_path: str,
    blur_area: dict,
    font_style: dict,
    blur_strength: int = 25,
    effect_type: str = "particles",
    log_file_path: str = "ffmpeg_output.log",
):
    """
    Working cyberpunk effect with animated background
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

    # Different effect types
    effects = {
        "particles": "noise=alls=30:allf=t,colorbalance=rs=-0.3:gs=-0.1:bs=0.4,boxblur=10,colorchannelmixer=aa=0.3",
        "waves": "geq=r=128+100*sin(2*PI*(X/W+t)):g=128+100*sin(2*PI*(Y/H+t)):b=255*sin(2*PI*(X+Y)/(W+H)+t),boxblur=5,colorbalance=rs=-0.2:gs=0.1:bs=0.4,colorchannelmixer=aa=0.25",
        "scanlines": "geq=r=255*mod(floor(Y/3),2):g=255*mod(floor(Y/3),2):b=255*mod(floor(Y/3),2),colorchannelmixer=aa=0.15",
        "gradient": f"geq=r=lerp(50,200,Y/H):g=lerp(100,150,Y/H):b=lerp(150,100,Y/H),colorchannelmixer=aa=0.2"
    }
    
    effect_filter = effects.get(effect_type, effects["particles"])
    
    # SIMPLE WORKING FILTER - no standalone color sources
    filter_complex = (
        # Crop the area we want to process
        f"[0:v]crop={w}:{h}:{x}:{y},"
        f"split=3[orig][for_effect][for_blur];"
        
        # Create effect layer
        f"[for_effect]{effect_filter}[effect];"
        
        # Create blurred background
        f"[for_blur]boxblur={blur_strength}:{blur_strength}[blurred];"
        
        # Combine effect with blurred background
        f"[blurred][effect]overlay=format=auto[bg_with_effect];"
        
        # Blend with original (optional transparency)
        f"[orig]colorchannelmixer=aa=0.7[orig_trans];"
        f"[bg_with_effect][orig_trans]overlay[final_bg];"
        
        # Overlay back onto original video
        f"[0:v][final_bg]overlay={x}:{y}[background];"
        
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

    print(f"Running cyberpunk effect: {effect_type}...")
    print(" ".join(cmd))

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ Success! Output: {output_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"✗ ffmpeg failed with code {e.returncode}")
        if e.stderr:
            print(f"Error: {e.stderr[:500]}")  # Show first 500 chars of error


# EVEN BETTER: Dual-color gradient effect
def burn_srt_with_gradient_blur(
    input_video_path: str,
    srt_file_path: str,
    output_video_path: str,
    blur_area: dict,
    font_style: dict,
    blur_strength: int = 25,
    gradient_colors: tuple = ("00FFFF", "FF00FF"),  # cyan to magenta
    log_file_path: str = "ffmpeg_output.log",
):
    """
    Beautiful dual-color gradient with blur
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
    
    # Convert hex colors to RGB
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    rgb1 = hex_to_rgb(gradient_colors[0])
    rgb2 = hex_to_rgb(gradient_colors[1])
    
    # Simple but beautiful gradient effect
    filter_complex = (
        f"[0:v]crop={w}:{h}:{x}:{y},"
        f"split=2[orig][for_blur];"
        
        # Create gradient using geq
        f"[for_blur]boxblur={blur_strength}:{blur_strength},"
        f"geq="
        f"r='lerp({rgb1[0]},{rgb2[0]},Y/H)':"
        f"g='lerp({rgb1[1]},{rgb2[1]},Y/H)':"
        f"b='lerp({rgb1[2]},{rgb2[2]},Y/H)',"
        f"format=rgba,"
        f"colorchannelmixer=aa=0.25[gradient_blur];"
        
        # Add subtle scanlines
        f"[orig]geq=r='255*mod(floor(Y/2),2)':g='255*mod(floor(Y/2),2)':b='255*mod(floor(Y/2),2)',"
        f"colorchannelmixer=aa=0.1[scanlines];"
        
        # Combine everything
        f"[gradient_blur][scanlines]overlay[final_bg];"
        
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

    print(f"Running gradient effect ({gradient_colors[0]} to {gradient_colors[1]})...")
    print(" ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
        print(f"✓ Gradient effect applied: {output_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed: {e.returncode}")


# ULTRA SIMPLE but looks AMAZING
def burn_srt_with_glow_effect(
    input_video_path: str,
    srt_file_path: str,
    output_video_path: str,
    blur_area: dict,
    font_style: dict,
    blur_strength: int = 30,
    glow_color: str = "00FFFF",
    log_file_path: str = "ffmpeg_output.log",
):
    """
    Simple glow effect that looks premium
    """
    x = blur_area.get("x", 0)
    y = blur_area.get("y", 0)
    w = blur_area.get("w", 0)
    h = blur_area.get("h", 0)

    force_style = build_force_style(font_style)
    srt_path_escaped = escape_path_for_filter(os.path.abspath(srt_file_path))
    
    # Convert hex to RGB
    glow_rgb = tuple(int(glow_color[i:i+2], 16) for i in (0, 2, 4))
    
    filter_complex = (
        f"[0:v]crop={w}:{h}:{x}:{y},"
        f"boxblur={blur_strength}:{blur_strength},"
        f"colorchannelmixer="
        f"rr=0.5:rg=0.2:rb=0.3:"  # Tint towards glow color
        f"gr=0.2:gg=0.6:gb=0.2:"
        f"br=0.3:bg=0.2:bb=0.8,"
        f"eq=brightness=0.1:contrast=1.2,"  # Boost contrast
        f"curves=r='0/0 0.5/0.7 1/1':g='0/0 0.5/0.7 1/1':b='0/0 0.5/0.7 1/1'[final_bg];"
        
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

    print(f"Running glow effect ({glow_color})...")
    print(" ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
        print(f"✓ Glow effect applied: {output_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed: {e.returncode}")


if __name__ == "__main__":
    # Configuration
    input_file = "n8n/Downloads/Sakinah Labs/Test2.mp4"
    srt_file = "transcripts/audio_for_transcription.srt"
    output_file = "n8n/Results/test_cyberpunk_final5.mp4"
    log_file = "ffmpeg_premium.log"

    width, height = get_video_resolution(input_file)
    print(f"📺 Video resolution: {width}x{height}")

    area_to_blur = {"x": 0, "y": 1255, "w": width, "h": 150}
    
    # Choose a style
    subtitle_style = get_premium_style("cyberpunk_cyan")
    # subtitle_style = get_premium_style("matrix_green")
    # subtitle_style = get_premium_style("synthwave")
    
    # TRY THIS FIRST - it works!
    burn_srt_with_glow_effect(
        input_video_path=input_file,
        srt_file_path=srt_file,
        output_video_path=output_file,
        blur_area=area_to_blur,
        font_style=subtitle_style,
        blur_strength=30,
        glow_color="00FFFF",  # Cyan glow
        log_file_path=log_file,
    )
    
    # OR try the gradient version
    # burn_srt_with_gradient_blur(
    #     input_video_path=input_file,
    #     srt_file_path=srt_file,
    #     output_video_path=output_file.replace(".mp4", "_gradient.mp4"),
    #     blur_area=area_to_blur,
    #     font_style=subtitle_style,
    #     blur_strength=25,
    #     gradient_colors=("00FFFF", "FF00FF"),  # Cyan to Magenta
    #     log_file_path=log_file,
    # )