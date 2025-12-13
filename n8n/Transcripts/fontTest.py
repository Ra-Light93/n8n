from PIL import ImageFont
import os

font_dirs = [
    "/System/Library/Fonts",
    "/Library/Fonts",
    os.path.expanduser("~/Library/Fonts")
]

fonts = []

for folder in font_dirs:
    if os.path.exists(folder):
        for f in os.listdir(folder):
            if f.lower().endswith((".ttf", ".ttc", ".otf")):
                fonts.append(f)

# Print results
print("=== Available Fonts on Your Mac ===\n")
for f in sorted(fonts):
    print(f)