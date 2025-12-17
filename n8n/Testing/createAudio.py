import sys
import torch

try:
    from transformers import AutoProcessor, DiaForConditionalGeneration
except ModuleNotFoundError:
    sys.exit(
        "ERROR: transformers is not installed in this env.\n"
        "Run: pip install transformers"
    )

ckpt = "nari-labs/Dia-1.6B-0626"
device = "cpu"  # braucht CUDA GPU

if device != "cpu":
    sys.exit("ERROR: Dia-1.6B requires CUDA GPU. Use device='cpu' on Mac.")

text = ["[S1] Hello. [S2] Hi!"]
processor = AutoProcessor.from_pretrained(ckpt)
inputs = processor(text=text, padding=True, return_tensors="pt").to(device)

print("Loading Dia-1.6B model on CPU (this may be very slow)...", flush=True)
model = DiaForConditionalGeneration.from_pretrained(ckpt).to(device) # type: ignore
out = model.generate(**inputs, max_new_tokens=3072, guidance_scale=3.0, temperature=1.8, top_p=0.90, top_k=45)
audio = processor.batch_decode(out)
processor.save_audio(audio, "dia.mp3")