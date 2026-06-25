import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torchaudio
from pathlib import Path
from transformers import AutoModel, AutoProcessor
from peft import PeftModel

torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

BASE_MODEL = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"
LORA_CKPT = "/home/green-gpu/mosstrain/output/moss_tts_indonesian_v2/checkpoint-last"
device = "cuda"
dtype = torch.bfloat16

print("[INFO] Loading processor...")
processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
processor.audio_tokenizer = processor.audio_tokenizer.to(device)

print("[INFO] Loading base model...")
model = AutoModel.from_pretrained(
    BASE_MODEL,
    dtype=dtype,
    attn_implementation="sdpa",
    trust_remote_code=True,
).to(device)

print(f"[INFO] Loading LoRA weights from: {LORA_CKPT}")
model = PeftModel.from_pretrained(model, LORA_CKPT)
model.eval()
print("[INFO] Model + LoRA loaded.")

text_id = (
    "Selamat datang di Indonesia, negeri yang kaya akan budaya dan keindahan alam. "
    "Dari Sabang sampai Merauke, setiap daerah memiliki keunikan tersendiri. "
    "Bahasa Indonesia adalah bahasa persatuan yang menghubungkan lebih dari tiga ratus suku bangsa. "
    "Semoga perjalanan Anda menyenangkan dan penuh kenangan indah."
)

print(f"[INFO] Text: {text_id}")

conversations = [
    [processor.build_user_message(text=text_id, language="Indonesian", tokens=375)],
]

batch = processor(conversations, mode="generation")
input_ids = batch["input_ids"].to(device)
attention_mask = batch["attention_mask"].to(device)

print("[INFO] Generating...")
with torch.inference_mode():
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=4096,
        do_sample=True,
        audio_temperature=1.7,
        audio_top_p=0.8,
        audio_top_k=25,
        audio_repetition_penalty=1.0,
    )

out_dir = Path("/home/green-gpu/mosstrain")
for i, message in enumerate(processor.decode(outputs)):
    if message is None:
        print(f"[WARN] message {i} is None, skipping")
        continue
    audio = message.audio_codes_list[0]
    out_path = out_dir / "output_indonesian_v2.wav"
    torchaudio.save(str(out_path), audio, processor.model_config.sampling_rate)
    duration = audio.shape[-1] / processor.model_config.sampling_rate
    print(f"[INFO] Saved: {out_path}  ({duration:.1f}s, {processor.model_config.sampling_rate}Hz)")
