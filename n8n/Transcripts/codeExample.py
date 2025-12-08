import subprocess
import os
import json
import tempfile


def burn_srt_with_blur(
    input_video_path: str,
    srt_file_path: str,
    output_video_path: str,
    blur_area: dict,
    font_style: dict,
    blur_strength: int = 20,
    log_file_path: str = "ffmpeg_output.log"
):
    
    """
    Applies a blur effect to a region and burns an SRT subtitle file onto the video.
    Requires ffmpeg to be installed with libass support.

    Args:
        input_video_path: Path to the source video file.
        srt_file_path: Path to the .srt subtitle file.
        output_video_path: Path to save the modified video file.
        blur_area: A dictionary {'x': int, 'y': int, 'w': int, 'h': int} 
                   defining the rectangle to blur.
        font_style: A dictionary defining the subtitle style.
        blur_strength: How strong the blur should be.
        log_file_path: Path to the file where ffmpeg's output will be written.
    """
    if not os.path.exists(input_video_path):
        print(f"Error: Input video not found at {input_video_path}")
        return
    if not os.path.exists(srt_file_path):
        print(f"Error: SRT file not found at {srt_file_path}")
        return
    if not os.path.exists(font_style['font_file']):
        print(f"Error: Font file not found at {font_style['font_file']}.")
        return

    # Get video resolution for proper ASS scaling
    width, height = get_video_resolution(input_video_path)
    
    # Create a temporary ASS file with proper styling
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ass', delete=False, encoding='utf-8') as temp_ass:
        temp_ass_file = temp_ass.name
        
        # Read the SRT file
        with open(srt_file_path, 'r', encoding='utf-8') as srt_file:
            srt_content = srt_file.read()
        
        # Get style parameters with defaults
        font_name = os.path.basename(font_style['font_file']).split('.')[0]
        font_size = font_style.get('font_size', '24')
        font_color = font_style.get('font_color', 'FFFFFF')
        outline_color = font_style.get('outline_color', '000000')
        outline = font_style.get('outline', '2.0')
        shadow = font_style.get('shadow', '1.0')
        alignment = font_style.get('alignment', '2')  # 2 = bottom-center
        margin_l = font_style.get('margin_l', '20')
        margin_r = font_style.get('margin_r', '20')
        margin_v = font_style.get('margin_v', str(height - 100))  # Position near bottom
        
        # Create ASS header with proper styling - USE ACTUAL VIDEO RESOLUTION
        ass_header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},&H{font_color},&H000000FF,&H{outline_color},&H00000000,0,0,0,0,100,100,0,0,1,{outline},{shadow},{alignment},{margin_l},{margin_r},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        
        # Convert SRT to ASS events
        ass_events = ""
        srt_lines = srt_content.split('\n')
        i = 0
        while i < len(srt_lines):
            line = srt_lines[i].strip()
            if '-->' in line:
                timestamp = line.replace(',', '.')
                start_time, end_time = timestamp.split(' --> ')
                text_lines = []
                i += 1
                while i < len(srt_lines) and srt_lines[i].strip() and not srt_lines[i].strip().isdigit() and '-->' not in srt_lines[i]:
                    text_lines.append(srt_lines[i].strip())
                    i += 1
                
                if text_lines:
                    text = '\\N'.join(text_lines)  # \N for line breaks in ASS
                    ass_events += f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{text}\n"
            else:
                i += 1
        
        temp_ass.write(ass_header + ass_events)
    
    print(f"Created temporary ASS file: {temp_ass_file}")
    print(f"ASS PlayRes: {width}x{height}")
    print(f"Text position - MarginV: {margin_v}, MarginR: {margin_r}")
    
    # OPTION 1: Text ON TOP of blur (recommended - text appears over blurred area)
    filter_complex = (
        f"[0:v]crop={blur_area['w']}:{blur_area['h']}:{blur_area['x']}:{blur_area['y']},"
        f"boxblur={blur_strength}[blurred_box];"
        f"[0:v][blurred_box]overlay={blur_area['x']}:{blur_area['y']}[blurred_vid];"
        f"[blurred_vid]ass='{temp_ass_file}'[final_v]"
    )
    
    # OPTION 2: Text BELOW blur (if you want text outside blurred area)
    # filter_complex = (
    #     f"[0:v]crop={blur_area['w']}:{blur_area['h']}:{blur_area['x']}:{blur_area['y']},"
    #     f"boxblur={blur_strength}[blurred_box];"
    #     f"[0:v]ass='{temp_ass_file}'[subtitled];"
    #     f"[subtitled][blurred_box]overlay={blur_area['x']}:{blur_area['y']}[final_v]"
    # )
    
    # The full ffmpeg command
    command = [
        'ffmpeg',
        '-i', input_video_path,
        '-filter_complex', filter_complex,
        '-map', '[final_v]',
        '-map', '0:a?',
        '-c:a', 'copy',
        '-y',
        output_video_path
    ]

    print("Running ffmpeg command to burn subtitles...")
    print(f"Command: {' '.join(command)}")
    print(f"Output will be logged to: {log_file_path}")

    try:
        with open(log_file_path, 'w') as log_file:
            result = subprocess.run(command, check=True, stdout=log_file, stderr=log_file)
        print(f"\nSuccessfully created video with burned subtitles: {output_video_path}")
        
    except subprocess.CalledProcessError as e:
        print(f"\nError during ffmpeg execution (code: {e.returncode}). Check the log file for details:")
        print(f"Log file: {log_file_path}")
        
        # Try alternative approach if first one fails
        print("\nTrying alternative approach...")
        alt_filter_complex = (
            f"[0:v]ass='{temp_ass_file}'[subtitled];"
            f"[subtitled]crop={blur_area['w']}:{blur_area['h']}:{blur_area['x']}:{blur_area['y']},"
            f"boxblur={blur_strength}[blurred_box];"
            f"[subtitled][blurred_box]overlay={blur_area['x']}:{blur_area['y']}[final_v]"
        )
        
        alt_command = [
            'ffmpeg',
            '-i', input_video_path,
            '-filter_complex', alt_filter_complex,
            '-map', '[final_v]',
            '-map', '0:a?',
            '-c:a', 'copy',
            '-y',
            output_video_path
        ]
        
        try:
            with open(log_file_path, 'a') as log_file:
                log_file.write("\n\n=== TRYING ALTERNATIVE APPROACH ===\n")
                subprocess.run(alt_command, check=True, stdout=log_file, stderr=log_file)
            print(f"Alternative approach succeeded! Created: {output_video_path}")
        except subprocess.CalledProcessError as e2:
            print(f"Alternative approach also failed (code: {e2.returncode})")
            
    finally:
        # Clean up temporary file
        if os.path.exists(temp_ass_file):
            os.remove(temp_ass_file)
            print(f"Cleaned up temporary file: {temp_ass_file}")


def get_video_resolution(path: str):
    """Get video width and height using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json", path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    data = json.loads(result.stdout)
    w = data["streams"][0]["width"]
    h = data["streams"][0]["height"]
    return w, h


if __name__ == '__main__':
    # --- CONFIGURATION ---
    input_file = "n8n/Downloads/Sakinah Labs/Test2.mp4"
    srt_file = "transcripts/audio_for_transcription.srt"
    output_file = "n8n/Downloads/Sakinah Labs/video_with_subtitles.mp4"
    log_file = "ffmpeg_srt_burn.log"

    # Get video resolution
    width, height = get_video_resolution(input_file)
    print(f"Video resolution: {width}x{height}")

    # Define blur area (adjust as needed)
    area_to_blur = {'x': 0, 'y': 1270, 'w': width, 'h': 130}
    
    # --- SUBTITLE STYLE CONFIGURATION ---
    # IMPORTANT: Adjust these values based on your video resolution
    # MarginV: Distance from top of screen (in pixels)
    # MarginR: Distance from right edge (for alignment 3, 6, 9)
    # MarginL: Distance from left edge (for alignment 1, 4, 7)
    # Alignment: 1=bottom-left, 2=bottom-center, 3=bottom-right,
    #            4=middle-left, 5=middle-center, 6=middle-right,
    #            7=top-left, 8=top-center, 9=top-right
    
    # Option 1: Text at bottom-center (most common for subtitles)
    subtitle_style = {
        'font_file': "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        'font_size': '28',  # Adjust based on video resolution
        'font_color': 'FFFFFF',  # White
        'outline_color': '000000',  # Black outline
        'outline': '2.0',
        'shadow': '1.0',
        'alignment': '2',  # 2 = bottom-center
        'margin_l': '20',
        'margin_r': '20',
        'margin_v': str(height - 80)  # 80px from bottom
    }
    
    # Option 2: Text at bottom-right (if you want it in the corner)
    # subtitle_style = {
    #     'font_file': "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    #     'font_size': '24',
    #     'font_color': 'FFFFFF',
    #     'outline_color': '000000',
    #     'outline': '2.0',
    #     'shadow': '1.0',
    #     'alignment': '3',  # 3 = bottom-right
    #     'margin_l': '20',
    #     'margin_r': '40',  # 40px from right edge
    #     'margin_v': str(height - 80)  # 80px from bottom
    # }
    
    # Option 3: Text INSIDE the blurred area (top-right corner of blur)
    # subtitle_style = {
    #     'font_file': "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    #     'font_size': '22',
    #     'font_color': 'FFFFFF',
    #     'outline_color': '000000',
    #     'outline': '3.0',  # Thicker outline for better visibility on blur
    #     'shadow': '1.5',
    #     'alignment': '9',  # 9 = top-right (inside blurred area)
    #     'margin_l': '0',
    #     'margin_r': '30',  # 30px from right edge of screen
    #     'margin_v': str(area_to_blur['y'] + 20)  # 20px below top of blur area
    # }
    
    print(f"\nBlur area: {area_to_blur}")
    print(f"Text style: Alignment={subtitle_style['alignment']}, "
          f"MarginV={subtitle_style['margin_v']}, "
          f"MarginR={subtitle_style['margin_r']}")
    
    # --- Run the function ---
    burn_srt_with_blur(
        input_video_path=input_file,
        srt_file_path=srt_file,
        output_video_path=output_file,
        blur_area=area_to_blur,
        font_style=subtitle_style,
        blur_strength=20,
        log_file_path=log_file
    )