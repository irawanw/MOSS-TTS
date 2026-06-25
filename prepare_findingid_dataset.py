"""
Extract text+audio pairs from findingid segment MP3s + JSON scripts.
Outputs: data/findingid/train_raw.jsonl
"""
import json
import subprocess
from pathlib import Path

SEGMENTS_DIR = Path("/home/green-gpu/mosstrain/data/findingid/segments")
JSON_DIR = Path("/home/green-gpu/mosstrain/data/findingid")
WAV_DIR = Path("/home/green-gpu/mosstrain/data/findingid/wav")
OUT_JSONL = Path("/home/green-gpu/mosstrain/data/findingid/train_raw.jsonl")

WAV_DIR.mkdir(parents=True, exist_ok=True)

records = []
skipped = 0

for seg_dir in sorted(SEGMENTS_DIR.iterdir()):
    if not seg_dir.is_dir():
        continue
    short_id = seg_dir.name.replace("_segments", "")
    json_path = JSON_DIR / f"{short_id}.json"
    if not json_path.exists():
        skipped += 1
        continue

    with open(json_path) as f:
        script = json.load(f)

    segments = script.get("segments", [])
    for seg in segments:
        seg_id = seg.get("id", "")
        narration = seg.get("narration", "").strip()
        if not narration or not seg_id:
            continue

        mp3_path = seg_dir / f"{seg_id}.mp3"
        if not mp3_path.exists():
            continue

        wav_path = WAV_DIR / f"{short_id}_{seg_id}.wav"

        # Convert MP3 → WAV 48kHz mono using ffmpeg
        result = subprocess.run([
            "ffmpeg", "-y", "-i", str(mp3_path),
            "-ar", "48000", "-ac", "1",
            str(wav_path)
        ], capture_output=True)

        if result.returncode != 0 or not wav_path.exists():
            print(f"[WARN] ffmpeg failed: {mp3_path}")
            continue

        records.append({
            "audio": str(wav_path),
            "text": narration,
            "language": "Indonesian",
        })

print(f"[INFO] Records: {len(records)}, skipped dirs (no JSON): {skipped}")

with open(OUT_JSONL, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"[INFO] Written: {OUT_JSONL}")
