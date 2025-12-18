import os, json
from pathlib import Path
from dotenv import load_dotenv
from google import genai

load_dotenv("n8n/.env") 

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-flash-latest")

PROMPT = """You are given the transcript of an audio used in a short video.
Your task is to classify it into exactly ONE background music category.

Categories:
Atmospheric, Breakbeat, Carefree, Disco, Dramatic, Generic,
Happy, Hip Hop, Hopeful, Jazz, Motivation, Rock & Roll

Rules:
- Choose exactly one category that fits best.
- Output ONLY valid JSON.
- All categories must be present.
- Exactly one category must be true.
- All others must be false.

Output format example:
{{
  "Atmospheric": false,
  "Breakbeat": false,
  "Carefree": false,
  "Disco": false,
  "Dramatic": false,
  "Generic": false,
  "Happy": false,
  "Hip Hop": false,
  "Hopeful": false,
  "Jazz": false,
  "Motivation": true,
  "Rock & Roll": false
}}

Audio transcript:
{audio}
"""

def srt_to_text(srt_path: str) -> str:
    """Extracts only subtitle text from an .srt file (drops indices/timestamps)."""
    p = Path(srt_path)
    raw = p.read_text(encoding="utf-8", errors="ignore")

    lines = []
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        # skip numeric index lines
        if s.isdigit():
            continue
        # skip timestamp lines
        if "-->" in s:
            continue
        # keep subtitle text
        lines.append(s)

    # join with spaces to form one transcript
    return " ".join(lines)

def classify_audio_text(audio_text: str) -> dict:
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=PROMPT.format(audio=audio_text),
        )
        
        # Extract the text - check different possible response structures
        if hasattr(response, 'text'):
            response_text = response.text
        elif hasattr(response, 'candidates') and len(response.candidates) > 0: # type:ignore
            response_text = response.candidates[0].content.parts[0].text # type:ignore
        else:
            # Try to access the text directly from the response object
            response_text = str(response)
            
        # Clean the response text
        response_text = response_text.strip() # type:ignore
        
        # Remove markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        print(f"Raw response: {response_text}")  # Debug print
        
        # Parse JSON
        data = json.loads(response_text)
        for k, v in data.items():
            if v is True:
                return k
        return None # type:ignore
        
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        print(f"Response that failed to parse: {response_text}")
        # Return a default structure if parsing fails
        return "Motivation" # type:ignore
    except Exception as e:
        print(f"Error in classify_audio_text: {e}")
        raise

# Test the function
if __name__ == "__main__":
    import sys

    if "--list-models" in sys.argv:
        for m in client.models.list(config={"page_size": 50}):
            print(getattr(m, "name", m))
        raise SystemExit(0)

    # Usage:
    #   python AudioAdvice.py n8n/Testing/videoOuput/last/lastV1Upper.srt
    #   python AudioAdvice.py --list-models

    if len(sys.argv) < 2:
        print("Usage: python AudioAdvice.py <path/to/file.srt> or --list-models")
        raise SystemExit(2)

    srt_path = sys.argv[1]
    if srt_path.startswith("-"):
        print("Usage: python AudioAdvice.py <path/to/file.srt> or --list-models")
        raise SystemExit(2)

    audio_text = srt_to_text(srt_path); print(audio_text)
    if not audio_text.strip():
        print("Error: SRT contained no subtitle text.")
        raise SystemExit(2)

    result = classify_audio_text(audio_text)
    print("\nSelected category:")
    print(result)

# python \
# n8n/Testing/AudioAdvice.py \
# n8n/Testing/videoOuput/last/lastV1Upper.srt