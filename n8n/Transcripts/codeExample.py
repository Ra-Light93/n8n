import os
import subprocess
import json
import math
from typing import Dict, Tuple, Optional
import tempfile


class VideoSubtitleStyler:
    def __init__(self):
        self.temp_files = []
        
    def __del__(self):
        """Cleanup temp files"""
        for file in self.temp_files:
            try:
                os.remove(file)
            except:
                pass
    
    def get_video_resolution(self, path: str) -> tuple[int, int]:
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
    
    def create_neon_ass_style(self, video_width: int, video_height: int) -> str:
        """
        Create a full ASS file with Modern Neon style including animations and effects
        """
        # Modern Neon Style Parameters
        neon_glow_color = "&H00FF00&"  # Neon Green (BGR format: 00FF00 = 00 BB GG RR)
        neon_text_color = "&HFFFFFF&"   # White text
        neon_shadow_color = "&H000000&" # Black shadow
        font_size = int(video_height * 0.04)  # Responsive font size (4% of video height)
        
        ass_content = """[Script Info]
Title: Modern Neon Subtitles
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: None

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: NeonTitle,Impact,{fontsize},{primary},{secondary},{outline},{back},0,0,0,0,100,100,0,0,1,1.5,2,2,{margin_l},{margin_r},{margin_v},1
Style: NeonGlow,Impact,{fontsize},{glow},{secondary},{outline},{back},0,0,0,0,100,100,0,0,1,3.5,0,2,{margin_l},{margin_r},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""".format(
            width=video_width,
            height=video_height,
            fontsize=font_size,
            primary=neon_text_color,
            secondary=neon_text_color,
            outline=neon_shadow_color,
            back=neon_shadow_color,
            glow=neon_glow_color,
            margin_l=int(video_width * 0.05),
            margin_r=int(video_width * 0.05),
            margin_v=int(video_height * 0.08)
        )
        
        return ass_content
    
    def srt_to_ass_with_neon_effects(self, srt_path: str, video_width: int, video_height: int) -> str:
        """
        Convert SRT to ASS with Modern Neon effects and animations
        """
        def time_to_ass(timestamp: str) -> str:
            """Convert SRT time (00:00:00,000) to ASS time (0:00:00.00)"""
            time_parts = timestamp.replace(',', '.').split(':')
            if len(time_parts) == 3:
                return f"{time_parts[0]}:{time_parts[1]}:{time_parts[2]}"
            return timestamp
        
        def parse_srt_time(time_str: str) -> float:
            """Convert time string to seconds"""
            try:
                h, m, s_ms = time_str.split(':')
                s, ms = s_ms.replace(',', '.').split('.')
                return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000
            except:
                return 0
        
        # Create ASS header
        ass_content = self.create_neon_ass_style(video_width, video_height)
        
        # Parse SRT and convert to ASS with effects
        with open(srt_path, 'r', encoding='utf-8') as f:
            lines = f.read().strip().split('\n')
        
        i = 0
        events = []
        
        while i < len(lines):
            # Skip empty lines
            if not lines[i].strip():
                i += 1
                continue
            
            # Try to parse subtitle number
            try:
                subtitle_num = int(lines[i].strip())
                i += 1
            except:
                i += 1
                continue
            
            # Parse time codes
            if i < len(lines):
                time_line = lines[i].strip()
                i += 1
                
                if '-->' in time_line:
                    start_time, end_time = [t.strip() for t in time_line.split('-->')]
                    start_ass = time_to_ass(start_time)
                    end_ass = time_to_ass(end_time)
                    
                    # Parse duration for animations
                    duration = parse_srt_time(end_time) - parse_srt_time(start_time)
                    
                    # Collect text lines
                    text_lines = []
                    while i < len(lines) and lines[i].strip() and not lines[i].strip().isdigit():
                        text_lines.append(lines[i].strip())
                        i += 1
                    
                    if text_lines:
                        full_text = '\\N'.join(text_lines)
                        
                        # Add glow layer (slightly behind main text)
                        glow_event = f"Dialogue: 1,{start_ass},{end_ass},NeonGlow,,0,0,0,,{full_text}"
                        
                        # Add main text layer with fade animation
                        # \fad(in_ms, out_ms) - fade in/out
                        fade_in = min(300, int(duration * 1000 * 0.2))  # 20% of duration or 300ms
                        fade_out = min(300, int(duration * 1000 * 0.2))
                        
                        # Add neon effect with \c (color override) and \3c (outline color)
                        neon_effect = f"{{\\fad({fade_in},{fade_out})\\3c&H00FF00&\\blur2\\be1}}"
                        main_event = f"Dialogue: 2,{start_ass},{end_ass},NeonTitle,,0,0,0,{neon_effect},{full_text}"
                        
                        events.append(glow_event)
                        events.append(main_event)
            
            i += 1
        
        # Add events to ASS content
        ass_content += '\n' + '\n'.join(events)
        
        # Create temporary ASS file
        temp_ass = tempfile.NamedTemporaryFile(mode='w', suffix='.ass', delete=False, encoding='utf-8')
        temp_ass.write(ass_content)
        temp_ass.close()
        self.temp_files.append(temp_ass.name)
        
        return temp_ass.name
    
    def create_fuggy_background_filter(self, video_width: int, video_height: int, 
                                       blur_strength: float = 15.0, 
                                       opacity: float = 0.7) -> str:
        """
        Create a fancy blurry background with rounded corners and gradient overlay
        """
        # Calculate dimensions for bottom bar (responsive)
        bar_height = int(video_height * 0.18)  # 18% of video height
        bar_y = video_height - bar_height
        corner_radius = int(min(bar_height * 0.3, 40))  # Rounded corners
        
        # Create complex blur effect with multiple layers
        # 1. Crop the bottom area
        # 2. Apply multiple blur passes for smoother effect
        # 3. Add color overlay for neon effect
        # 4. Round corners
        # 5. Overlay back
        
        filter_parts = [
            # First blur pass - main blur
            f"[0:v]crop={video_width}:{bar_height}:0:{bar_y},"
            f"gblur=sigma={blur_strength}[blur1];",
            
            # Second blur pass for extra smoothness
            f"[blur1]gblur=sigma={blur_strength*0.7}[blur2];",
            
            # Add color tint (dark with green tint for neon theme)
            f"[blur2]colorchannelmixer=rr=0.2:gg=0.3:bb=0.5,"
            f"format=rgba,"
            f"colorchannelmixer=aa={opacity}[tinted];",
            
            # Create rounded corners mask
            f"color=black@0:size={video_width}x{bar_height}[mask_base];",
            f"[mask_base]drawbox=0:0:{video_width}:{bar_height}:black@1:"
            f"t=fill[filled];",
            f"[filled]geq=r='if(gt(sqrt((X-{corner_radius})^2+(Y-{corner_radius})^2),{corner_radius}),"
            f"if(gt(sqrt((X-({video_width}-{corner_radius}))^2+(Y-{corner_radius})^2),{corner_radius}),"
            f"if(gt(sqrt((X-{corner_radius})^2+(Y-({bar_height}-{corner_radius}))^2),{corner_radius}),"
            f"if(gt(sqrt((X-({video_width}-{corner_radius}))^2+(Y-({bar_height}-{corner_radius}))^2),{corner_radius}),"
            f"255,0),0),0),0)':a='if(gt(sqrt((X-{corner_radius})^2+(Y-{corner_radius})^2),{corner_radius}),"
            f"if(gt(sqrt((X-({video_width}-{corner_radius}))^2+(Y-{corner_radius})^2),{corner_radius}),"
            f"if(gt(sqrt((X-{corner_radius})^2+(Y-({bar_height}-{corner_radius}))^2),{corner_radius}),"
            f"if(gt(sqrt((X-({video_width}-{corner_radius}))^2+(Y-({bar_height}-{corner_radius}))^2),{corner_radius}),"
            f"255,0),0),0),0)'[rounded_mask];",
            
            # Apply mask to tinted blur
            f"[tinted][rounded_mask]alphamerge[blurred_bg];",
            
            # Overlay blurred background onto original video
            f"[0:v][blurred_bg]overlay=0:{bar_y}[video_with_bg];"
        ]
        
        return ''.join(filter_parts)
    
    def escape_path(self, path: str) -> str:
        """Escape path for ffmpeg filter"""
        return path.replace("'", r"'\''")
    
    def burn_subtitles_with_modern_neon(
        self,
        input_video_path: str,
        srt_file_path: str,
        output_video_path: str,
        log_file_path: str = "ffmpeg_neon_output.log"
    ):
        """
        Main function to burn subtitles with Modern Neon style and fuggy background
        """
        # Basic checks
        if not os.path.exists(input_video_path):
            print(f"Error: Input video not found: {input_video_path}")
            return
        if not os.path.exists(srt_file_path):
            print(f"Error: SRT file not found: {srt_file_path}")
            return
        
        # Get video resolution
        width, height = self.get_video_resolution(input_video_path)
        print(f"🎬 Video resolution: {width}x{height}")
        
        # Convert SRT to ASS with neon effects
        print("✨ Creating Modern Neon ASS subtitles...")
        ass_file_path = self.srt_to_ass_with_neon_effects(srt_file_path, width, height)
        
        # Create fuggy background filter
        print("🌀 Creating fuggy neon background...")
        fuggy_filter = self.create_fuggy_background_filter(width, height, blur_strength=12.0, opacity=0.65)
        
        # Escape paths
        ass_path_escaped = self.escape_path(os.path.abspath(ass_file_path))
        
        # Build complete filter chain
        filter_complex = (
            fuggy_filter +
            f"[video_with_bg]ass='{ass_path_escaped}'[final_v]"
        )
        
        # FFmpeg command
        cmd = [
            "ffmpeg",
            "-i", input_video_path,
            "-filter_complex", filter_complex,
            "-map", "[final_v]",
            "-map", "0:a?",  # Copy audio if exists
            "-c:a", "copy",
            "-y",  # Overwrite output
            "-preset", "slow",  # Better quality encoding
            "-crf", "18",  # High quality
            output_video_path,
        ]
        
        print("\n🚀 Running FFmpeg with Modern Neon effects...")
        print("Command preview:")
        print(" ".join(cmd[:10]) + " ... [filter complex] ...")
        
        # Run FFmpeg
        with open(log_file_path, "w") as log_file:
            try:
                print("\n⏳ Processing... This might take a moment...")
                subprocess.run(cmd, check=True, stdout=log_file, stderr=log_file, text=True)
                print(f"\n✅ Done! Video saved to: {output_video_path}")
                print(f"📊 Log file: {log_file_path}")
                
                # Clean up temp files
                for temp_file in self.temp_files:
                    try:
                        os.remove(temp_file)
                    except:
                        pass
                self.temp_files.clear()
                
            except subprocess.CalledProcessError as e:
                print(f"❌ FFmpeg failed with code {e.returncode}")
                print(f"📄 Check log file for details: {log_file_path}")
            except FileNotFoundError:
                print("❌ FFmpeg not found! Please install FFmpeg and add it to your PATH")
            except Exception as e:
                print(f"❌ Unexpected error: {e}")


# Easy-to-use wrapper function
def apply_modern_neon_subtitles(
    input_video: str,
    srt_file: str,
    output_video: str,
    log_file: str = "ffmpeg_neon.log"
):
    """
    Simple function to apply Modern Neon subtitles to your video
    """
    styler = VideoSubtitleStyler()
    styler.burn_subtitles_with_modern_neon(
        input_video_path=input_video,
        srt_file_path=srt_file,
        output_video_path=output_video,
        log_file_path=log_file
    )


if __name__ == "__main__":
    # --- Configuration ---
    input_file = "n8n/Downloads/Sakinah Labs/Test2.mp4"
    srt_file = "transcripts/audio_for_transcription.srt"
    output_file = "n8n/Downloads/Sakinah Labs/Modern_Neon_Output.mp4"
    
    print("=" * 50)
    print("MODERN NEON SUBTITLE STYLER")
    print("=" * 50)
    print("🎮 Features:")
    print("  • Neon Glow Effects")
    print("  • Smooth Fade Animations")
    print("  • Fuggy Rounded Bottom Bar")
    print("  • Responsive Design")
    print("  • Professional ASS Styling")
    print("=" * 50)
    
    # Apply Modern Neon effects
    apply_modern_neon_subtitles(
        input_video=input_file,
        srt_file=srt_file,
        output_video=output_file
    )
    
    print("\n✨ Tip: For even better results, try:")
    print("   1. Using a different neon color (edit line 42)")
    print("   2. Adjusting blur_strength in create_fuggy_background_filter()")
    print("   3. Changing the font to 'Arial Black' or 'Bahnschrift'")