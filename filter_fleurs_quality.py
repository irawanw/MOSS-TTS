"""
Filter FLEURS Indonesian samples by audio quality.
Uses SNR estimation + spectral flatness, calibrated against known good examples.
Keeps top ~20-25% of samples.
"""
import json
import numpy as np
import torchaudio
from pathlib import Path

JSONL_IN = Path("/home/green-gpu/mosstrain/data/indonesian/train_raw.jsonl")
JSONL_OUT = Path("/home/green-gpu/mosstrain/data/indonesian/train_raw_filtered.jsonl")

# Known good quality samples (user-verified)
GOOD_EXAMPLES = [
    "data/indonesian/wav/train_00000.wav",
    "data/indonesian/wav/train_00011.wav",
    "data/indonesian/wav/train_00013.wav",
    "data/indonesian/wav/train_00025.wav",
    "data/indonesian/wav/train_00035.wav",
    "data/indonesian/wav/train_00037.wav",
    "data/indonesian/wav/train_00038.wav",
    "data/indonesian/wav/train_00040.wav",
]

def audio_quality_score(path: str) -> float:
    """
    Returns a quality score: higher = cleaner.
    Combines:
    - SNR estimate (speech energy vs noise floor)
    - Spectral flatness (low = tonal/speech, high = noisy)
    - Dynamic range
    """
    try:
        wav, sr = torchaudio.load(path)
        wav = wav.mean(0).numpy()  # mono

        # Resample to 16kHz for analysis
        if sr != 16000:
            import torchaudio.functional as F
            import torch
            wav = F.resample(torch.tensor(wav), sr, 16000).numpy()

        # Frame into 20ms windows
        frame_len = 320  # 20ms at 16kHz
        frames = [wav[i:i+frame_len] for i in range(0, len(wav)-frame_len, frame_len)]
        if len(frames) < 10:
            return 0.0

        frame_energies = np.array([np.mean(f**2) for f in frames])

        # SNR estimate: top 30% frames = speech, bottom 20% = noise floor
        sorted_e = np.sort(frame_energies)
        noise_floor = np.mean(sorted_e[:max(1, len(sorted_e)//5)])
        speech_energy = np.mean(sorted_e[int(len(sorted_e)*0.7):])
        snr = 10 * np.log10((speech_energy + 1e-10) / (noise_floor + 1e-10))

        # Spectral flatness (lower = more tonal = cleaner speech)
        fft = np.abs(np.fft.rfft(wav))
        fft = fft + 1e-10
        geo_mean = np.exp(np.mean(np.log(fft)))
        arith_mean = np.mean(fft)
        flatness = geo_mean / arith_mean  # 0..1, lower = cleaner

        # Dynamic range
        rms = np.sqrt(np.mean(wav**2))
        peak = np.max(np.abs(wav))
        dynamic = peak / (rms + 1e-10)

        # Combine: SNR weighted most
        score = snr * 0.6 + (1 - flatness) * 20 * 0.3 + min(dynamic, 10) * 0.1
        return float(score)
    except Exception as e:
        print(f"[WARN] Error scoring {path}: {e}")
        return 0.0

# Load all records
print("[INFO] Loading records...")
records = []
with open(JSONL_IN) as f:
    for line in f:
        records.append(json.loads(line))
print(f"[INFO] Total records: {len(records)}")

# Score good examples to set threshold
print("[INFO] Scoring reference good examples...")
good_scores = [audio_quality_score(p) for p in GOOD_EXAMPLES]
good_scores = [s for s in good_scores if s > 0]
print(f"[INFO] Good example scores: min={min(good_scores):.1f} avg={np.mean(good_scores):.1f} max={max(good_scores):.1f}")

# Score all samples
print("[INFO] Scoring all samples (this may take a few minutes)...")
scores = []
for i, rec in enumerate(records):
    s = audio_quality_score(rec["audio"])
    scores.append(s)
    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{len(records)} done...")

scores = np.array(scores)
print(f"[INFO] All scores: min={scores.min():.1f} avg={scores.mean():.1f} max={scores.max():.1f}")

# Threshold: use bottom of good examples as minimum, then keep top 25%
ref_min = min(good_scores) * 0.85  # 15% below worst known good sample
pct_threshold = np.percentile(scores, 75)  # top 25%
threshold = max(ref_min, pct_threshold)

print(f"[INFO] Threshold: {threshold:.1f} (ref_min={ref_min:.1f}, 75th_pct={pct_threshold:.1f})")

filtered = [(r, s) for r, s in zip(records, scores) if s >= threshold]
filtered.sort(key=lambda x: x[1], reverse=True)

print(f"[INFO] Kept: {len(filtered)}/{len(records)} ({100*len(filtered)/len(records):.1f}%)")

with open(JSONL_OUT, "w", encoding="utf-8") as f:
    for rec, _ in filtered:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"[INFO] Written: {JSONL_OUT}")

# Show score distribution of kept vs rejected
kept_scores = [s for _, s in filtered]
print(f"[INFO] Kept scores: min={min(kept_scores):.1f} avg={np.mean(kept_scores):.1f}")
