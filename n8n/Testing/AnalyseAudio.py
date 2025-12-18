import time
import hmac
import hashlib
import base64
import requests
import os
from dotenv import load_dotenv
from pathlib import Path

# Load .env from the same folder as this script (more reliable than CWD)
load_dotenv(dotenv_path="n8n/.env")

HOST = os.getenv("ACR_HOST")
ACCESS_KEY = os.getenv("audioDetectA")
ACCESS_SECRET = os.getenv("audioDetectS")

if not HOST or HOST.lower() == "none":
    raise SystemExit("Missing ACR_HOST. Put ACR_HOST=identify-eu-west-1.acrcloud.com in n8n/Testing/.env")
if not ACCESS_KEY:
    raise SystemExit("Missing audioDetectA (Access Key). Put audioDetectA=... in n8n/Testing/.env")
if not ACCESS_SECRET:
    raise SystemExit("Missing audioDetectS (Secret Key). Put audioDetectS=... in n8n/Testing/.env")

AUDIO_FILE = "n8n/Testing/videoOuput/PROB/here.mp3"   # 15–30s clip

# === SIGNATURE ===
timestamp = int(time.time())
string_to_sign = "\n".join([
    "POST",
    "/v1/identify",
    ACCESS_KEY,
    "audio",
    "1",
    str(timestamp)
])

signature = base64.b64encode(
    hmac.new(
        ACCESS_SECRET.encode(),
        string_to_sign.encode(),
        hashlib.sha1
    ).digest()
).decode()

# === REQUEST ===
with open(AUDIO_FILE, "rb") as f:
    files = {"sample": f}
    data = {
        "access_key": ACCESS_KEY,
        "sample_bytes": len(f.read())
    }

with open(AUDIO_FILE, "rb") as f:
    files = {"sample": f}
    data.update({
        "timestamp": timestamp,
        "signature": signature,
        "data_type": "audio",
        "signature_version": "1"
    })

    r = requests.post(
        f"https://{HOST}/v1/identify",
        files=files,
        data=data
    )

print(r.json())