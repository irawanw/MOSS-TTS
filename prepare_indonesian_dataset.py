"""
Download FLEURS Indonesian (id_id), save WAVs, and write train_raw.jsonl
for MOSS-TTS fine-tuning.
Uses torchaudio for audio decoding to avoid torchcodec dependency.
"""
import os
import io
import json
import torch
import torchaudio
import numpy as np
from pathlib import Path
from datasets import load_dataset, Audio

DATA_DIR = Path("/home/green-gpu/mosstrain/data/indonesian")
AUDIO_DIR = DATA_DIR / "wav"
JSONL_PATH = DATA_DIR / "train_raw.jsonl"

DATA_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

print("[INFO] Downloading FLEURS Indonesian (id_id)...")
# Load with decode=False to get raw bytes, avoiding torchcodec
ds = load_dataset("google/fleurs", "id_id", trust_remote_code=True)
ds = ds.cast_column("audio", Audio(decode=False))

splits = ["train", "validation", "test"]
records = []

for split in splits:
    if split not in ds:
        continue
    print(f"[INFO] Processing split: {split} ({len(ds[split])} samples)")
    for i, sample in enumerate(ds[split]):
        text = sample["transcription"].strip()
        if not text:
            continue

        fname = f"{split}_{i:05d}.wav"
        wav_path = AUDIO_DIR / fname

        # Decode audio bytes with torchaudio
        audio_bytes = sample["audio"]["bytes"]
        if audio_bytes:
            buf = io.BytesIO(audio_bytes)
            waveform, sr = torchaudio.load(buf)
        else:
            # Fallback: load from path if bytes not available
            src_path = sample["audio"].get("path", "")
            if not src_path or not os.path.exists(src_path):
                continue
            waveform, sr = torchaudio.load(src_path)

        torchaudio.save(str(wav_path), waveform, sr)

        records.append({
            "audio": str(wav_path),
            "text": text,
            "language": "Indonesian",
        })

        if (i + 1) % 500 == 0:
            print(f"  [{split}] {i+1}/{len(ds[split])} done")

print(f"[INFO] Total records: {len(records)}")

with open(JSONL_PATH, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"[INFO] JSONL written: {JSONL_PATH}")
print("[INFO] Done. Next step: run prepare_data.py to encode audio codes.")
print(f"""
Run:
  CUDA_VISIBLE_DEVICES=1 python moss_tts_local_v1.5/finetuning/prepare_data.py \\
    --model-path OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 \\
    --codec-path OpenMOSS-Team/MOSS-Audio-Tokenizer-v2 \\
    --codec-weight-dtype fp32 \\
    --codec-compute-dtype bf16 \\
    --device auto \\
    --input-jsonl {JSONL_PATH} \\
    --output-jsonl {DATA_DIR}/train_with_codes.jsonl \\
    --skip-reference-audio-codes
""")
