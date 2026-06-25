import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import re
import torch
import torchaudio
import tempfile
import gradio as gr
from transformers import AutoModel, AutoProcessor
from peft import PeftModel

BASE_MODEL = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"
LORA_CKPT  = os.path.join(os.path.dirname(__file__),
                           "output/moss_tts_indonesian_v2/checkpoint-last")
DEVICE = "cuda"

print("[demo] Loading model...")
processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
processor.audio_tokenizer = processor.audio_tokenizer.to(DEVICE)

_base = AutoModel.from_pretrained(
    BASE_MODEL, dtype=torch.bfloat16,
    attn_implementation="sdpa", trust_remote_code=True,
).to(DEVICE)
model = PeftModel.from_pretrained(_base, LORA_CKPT)
model.eval()
print("[demo] Ready.")

SR = processor.model_config.sampling_rate  # 48000


def split_sentences(text: str):
    parts = re.split(r'(?<=[.!?,;])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def generate(text: str, duration_s: float, temperature: float,
             top_p: float, top_k: int, ref_audio):
    if not text.strip():
        return None, "Please enter some text."

    sentences = split_sentences(text)
    tokens_per_sent = max(25, int((duration_s / max(len(sentences), 1)) * 12.5))

    reference = None
    if ref_audio is not None:
        reference = [ref_audio]

    audio_parts = []
    for sent in sentences:
        n_tok = max(25, int(len(sent.split()) * 4.5))
        conv = [[processor.build_user_message(
            text=sent,
            language="Indonesian",
            reference=reference,
            tokens=n_tok,
        )]]
        batch = processor(conv, mode="generation")
        with torch.inference_mode():
            out = model.generate(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
                max_new_tokens=min(int(n_tok * 1.5), 2048),
                do_sample=True,
                audio_temperature=temperature,
                audio_top_p=top_p,
                audio_top_k=top_k,
            )
        for msg in processor.decode(out):
            if msg:
                audio_parts.append(msg.audio_codes_list[0])

    if not audio_parts:
        return None, "Generation failed."

    audio = torch.cat(audio_parts, dim=-1)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    torchaudio.save(tmp.name, audio, SR)
    duration = audio.shape[-1] / SR
    return tmp.name, f"Generated {duration:.1f}s | {len(sentences)} sentence(s) | {SR}Hz stereo"


with gr.Blocks(title="MOSS-TTS Indonesian Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
# 🇮🇩 MOSS-TTS Indonesian LoRA Demo
Fine-tuned on Bahasa Indonesia — correct phonology, proper vowel "a" pronunciation.
""")

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Text (Bahasa Indonesia)",
                placeholder="Selamat datang di Indonesia...",
                lines=4,
                value="Selamat datang di Indonesia, negeri yang kaya akan budaya dan keindahan alam. Dari Sabang sampai Merauke, setiap daerah memiliki keunikan tersendiri.",
            )
            ref_audio = gr.Audio(
                label="Reference Voice (optional — for voice cloning)",
                type="filepath",
                sources=["upload", "microphone"],
            )

            with gr.Accordion("Advanced Settings", open=False):
                duration = gr.Slider(5, 120, value=30, step=5,
                                     label="Target Duration (seconds)")
                temperature = gr.Slider(0.5, 2.5, value=1.7, step=0.1,
                                        label="Temperature (higher = more expressive)")
                top_p = gr.Slider(0.1, 1.0, value=0.8, step=0.05, label="Top-p")
                top_k = gr.Slider(1, 100, value=25, step=1, label="Top-k")

            generate_btn = gr.Button("Generate", variant="primary", size="lg")

        with gr.Column(scale=2):
            audio_out = gr.Audio(label="Output Audio", type="filepath")
            status = gr.Textbox(label="Status", interactive=False)

    gr.Markdown("""
### Example Texts
""")
    gr.Examples(
        examples=[
            ["Halo semuanya! Selamat datang di channel kami. Jangan lupa subscribe dan klik tombol lonceng ya!"],
            ["Produk ini sangat berkualitas tinggi dengan harga yang terjangkau. Dapatkan sekarang sebelum kehabisan!"],
            ["Hari ini kita akan belajar tentang sejarah Indonesia yang kaya dan penuh dengan perjuangan para pahlawan bangsa."],
            ["Cuaca hari ini cerah dengan suhu sekitar dua puluh delapan derajat celsius. Cocok untuk beraktivitas di luar ruangan."],
        ],
        inputs=text_input,
    )

    generate_btn.click(
        fn=generate,
        inputs=[text_input, duration, temperature, top_p, top_k, ref_audio],
        outputs=[audio_out, status],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
