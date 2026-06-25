# MOSS-TTS Indonesian (Bahasa Indonesia) LoRA Fine-tune

Fine-tuned LoRA adapter for **MOSS-TTS-Local-Transformer-v1.5** that produces correct **Bahasa Indonesia** phonology — fixing the Malay-influenced pronunciation (vowel "a" sounding like "e") present in the base model.

## Problem

MOSS-TTS supports "Malay" but not Indonesian natively. Using `language="Malay"` produces Malaysian-English-influenced pronunciation where vowel **"a"** sounds like **"e"** (Melayu/Malaysian phonology). Indonesian requires a distinct phonology with clear open vowels.

## Solution

LoRA fine-tuning on a curated Indonesian speech dataset with `language="Indonesian"` tag — teaching the model to map "Indonesian" to correct Bahasa Indonesia phonology.

## Training Data

| Source | Samples | Duration | Quality |
|--------|---------|----------|---------|
| [FLEURS id_id](https://huggingface.co/datasets/google/fleurs) (filtered top 25%) | 904 | ~3h | Natural speech, SNR-filtered |
| FindingID short-video narrations | 532 | ~1.5h | Synthetic TTS, very clean |
| **Total** | **1,436** | **~4.5h** | |

FLEURS samples were filtered using SNR estimation calibrated against manually verified clean examples. Low-quality/noisy samples were excluded.

## Training Details

- **Base model**: `OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5` (4B params)
- **Method**: LoRA (PEFT), rank=32, alpha=64
- **Trainable params**: ~130M / 4,163M (3.1%)
- **Epochs**: 20
- **LR schedule**: Cosine decay, peak 2e-5
- **Batch size**: 8 (1 per device × 8 gradient accumulation)
- **Hardware**: Single NVIDIA RTX 3090 (24GB)
- **Training time**: ~3.5 hours

## Usage

### Installation

```bash
pip install transformers peft torchaudio
```

### Basic Inference

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torchaudio
from transformers import AutoModel, AutoProcessor
from peft import PeftModel

BASE_MODEL = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"
LORA_CKPT  = "path/to/moss_tts_indonesian_v2/checkpoint-last"
device = "cuda"

processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
processor.audio_tokenizer = processor.audio_tokenizer.to(device)

model = AutoModel.from_pretrained(
    BASE_MODEL, dtype=torch.bfloat16,
    attn_implementation="sdpa", trust_remote_code=True,
).to(device)
model = PeftModel.from_pretrained(model, LORA_CKPT)
model.eval()

text = "Selamat datang di Indonesia, negeri yang kaya akan budaya dan keindahan alam."

conversations = [
    [processor.build_user_message(text=text, language="Indonesian", tokens=125)],
]
batch = processor(conversations, mode="generation")

with torch.inference_mode():
    outputs = model.generate(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        max_new_tokens=4096,
        do_sample=True,
        audio_temperature=1.7,
        audio_top_p=0.8,
        audio_top_k=25,
    )

for message in processor.decode(outputs):
    if message:
        audio = message.audio_codes_list[0]
        torchaudio.save("output.wav", audio, processor.model_config.sampling_rate)
        print(f"Saved: output.wav ({audio.shape[-1]/processor.model_config.sampling_rate:.1f}s)")
```

### With Voice Cloning (Reference Audio)

```python
conversations = [
    [processor.build_user_message(
        text=text,
        language="Indonesian",
        reference=["path/to/reference_voice.wav"],
        tokens=125,
    )],
]
```

### Long Text (Sentence Splitting)

For texts longer than ~15 seconds, split into sentences and concatenate:

```python
import re

def split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

sentences = split_sentences(long_text)
audio_parts = []

for sentence in sentences:
    n_tokens = max(25, int(len(sentence.split()) * 4.5))
    conv = [[processor.build_user_message(text=sentence, language="Indonesian", tokens=n_tokens)]]
    batch = processor(conv, mode="generation")
    with torch.inference_mode():
        out = model.generate(input_ids=batch["input_ids"].to(device),
                             attention_mask=batch["attention_mask"].to(device),
                             max_new_tokens=2048, do_sample=True,
                             audio_temperature=1.7, audio_top_p=0.8, audio_top_k=25)
    for msg in processor.decode(out):
        if msg:
            audio_parts.append(msg.audio_codes_list[0])

import torch
combined = torch.cat(audio_parts, dim=-1)
torchaudio.save("output_long.wav", combined, processor.model_config.sampling_rate)
```

## Gradio Demo

```bash
pip install gradio
python demo_indonesian.py
```

## Token Count Guide

| Duration | Tokens |
|----------|--------|
| 5s | ~62 |
| 10s | ~125 |
| 20s | ~250 |
| 30s | ~375 |
| 60s | ~750 |

Frame rate: 12.5 Hz → tokens = seconds × 12.5

## Reproducing Training

```bash
# 1. Download & prepare FLEURS Indonesian
python prepare_indonesian_dataset.py

# 2. Filter to top 25% quality
python filter_fleurs_quality.py

# 3. Encode audio codes
CUDA_VISIBLE_DEVICES=0 python moss_tts_local_v1.5/finetuning/prepare_data.py \
    --model-path OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 \
    --codec-path OpenMOSS-Team/MOSS-Audio-Tokenizer-v2 \
    --input-jsonl data/combined/train_raw.jsonl \
    --output-jsonl data/combined/train_with_codes.jsonl \
    --skip-reference-audio-codes

# 4. Train
CUDA_HOME=/usr/local/cuda-12.6 CUDA_VISIBLE_DEVICES=0 \
accelerate launch moss_tts_local_v1.5/finetuning/sft.py \
    --model-path OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 \
    --train-jsonl data/combined/train_with_codes.jsonl \
    --output-dir output/moss_tts_indonesian_v2 \
    --per-device-batch-size 1 \
    --gradient-accumulation-steps 8 \
    --learning-rate 2.0e-5 \
    --lr-scheduler-type cosine \
    --num-epochs 20 \
    --mixed-precision bf16 \
    --gradient-checkpointing \
    --use-lora --lora-rank 32 --lora-alpha 64
```
