# Fine-Tuning Qwen3-TTS for Bengali — Step-by-Step

> Goal: train Qwen3-TTS to speak **Bengali** so it can be wired into Voicebox as a Bengali voice.
> Sources: [Qwen3-TTS finetuning dir](https://github.com/QwenLM/Qwen3-TTS/tree/main/finetuning) · [Issue #323 — Bengali fine-tuning](https://github.com/QwenLM/Qwen3-TTS/issues/323)
> Compiled: 2026-06-22

---

## ⚠️ Reality check before you start

1. **Qwen3-TTS does NOT ship Bengali.** Base languages = `zh en ja ko de fr ru pt es it` (10). Bengali must be *taught* via fine-tuning. There is no `<lang:bn>` switch — you create Bengali capability from data.
2. **Issue #323 is the cautionary tale.** A dev fine-tuned `Qwen3-TTS-12Hz-1.7B-Base` on ~7 hrs / ~8k Bengali samples (1 female speaker) and got **Bengali with a Hindi accent** and weak naturalness. Loss plateaued after 2–3 epochs. So expect this to be a *data + iteration* project, not a quick run. Their plan to fix it: **150+ hrs multi-speaker** data.
3. **The official finetuning only supports SINGLE-speaker** right now (multi-speaker "promised for future releases"). So you train one voice at a time.
4. **Hardware: this won't run on this machine.** Full fine-tune of the 1.7B model (AdamW, `batch_size 32`) needs a **24–80 GB datacenter GPU** (A100 / H100 / RTX 3090/4090-class). Our box is a **GTX 1650 Ti (4 GB), CPU-only torch** — it can run *inference* of small engines but **cannot train this**. Use cloud: RunPod / Lambda / Vast.ai / Colab Pro+ (A100). The 0.6B base lowers the bar but still won't fit 4 GB for full fine-tune.

---

## Method: full fine-tune (not LoRA)

`sft_12hz.py` does a **full** fine-tune — after each epoch it copies the whole model dir, updates configs with custom speaker metadata, and saves SafeTensors, **injecting the speaker embedding at index 3000** (speaker-encoder weights excluded). AdamW, weight decay 0.01.

---

## Step 0 — Environment (on the GPU box)

```bash
pip install qwen-tts
git clone https://github.com/QwenLM/Qwen3-TTS.git
cd Qwen3-TTS/finetuning
# files here: README.md  dataset.py  prepare_data.py  sft_12hz.py
```

You also need the **12 Hz tokenizer**: `Qwen/Qwen3-TTS-Tokenizer-12Hz` (auto-downloaded by `prepare_data.py`).

---

## Step 1 — Build the Bengali dataset (JSONL)

Each line needs **three** fields:

```jsonl
{"audio":"./data/utt0001.wav","text":"আমি বাংলায় কথা বলছি।","ref_audio":"./data/ref.wav"}
{"audio":"./data/utt0002.wav","text":"আজ আবহাওয়া খুব সুন্দর।","ref_audio":"./data/ref.wav"}
```

- `audio` — the training clip (wav). Clean, 1 speaker, consistent recording.
- `text` — the **Bengali transcript** (see G2P note below).
- `ref_audio` — a reference clip of the **same speaker**. The README says **use the SAME `ref_audio` for every sample** to lock speaker consistency.

**Data guidance distilled from #323:**
- 7 hrs / 8k samples = *not enough* (Hindi-accent leakage). Aim **much higher** (the reporter targets **150+ hrs**, multi-speaker once supported).
- Single clean female/male speaker to start; consistent mic, 16k+ wav, trimmed silences.
- **Script vs phonemes tradeoff (key finding):**
  - **Raw Bengali script** → *more natural* prosody **but** picks up **Hindi accent** (the base model's Indic bias).
  - **IPA / romanized phonemes** → *reduces* the Hindi accent **but** sounds **robotic**.
  - → Practical path: start with **raw Bengali script + lots of data**; only reach for a Bengali **G2P/phoneme** front-end if accent bias stays unacceptable. There is no official Bengali G2P for this model — candidates to evaluate: `bnnumerizer`/`bnUnicodeNormalizer` for text norm, `epitran` (ben-Beng) or AI4Bharat tools for IPA.

---

## Step 2 — Extract audio codes

```bash
python prepare_data.py \
  --device cuda:0 \
  --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
  --input_jsonl train_raw.jsonl \
  --output_jsonl train_with_codes.jsonl
```

Produces `train_with_codes.jsonl` (audio → discrete codec codes) used by training.

---

## Step 3 — Train

```bash
python sft_12hz.py \
  --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --output_model_path output_bengali \
  --train_jsonl train_with_codes.jsonl \
  --batch_size 32 \
  --lr 2e-6 \
  --num_epochs 10 \
  --speaker_name bn_speaker01
```

**Arguments** (defaults in parens):

| Arg | Default | Notes |
|---|---|---|
| `--init_model_path` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | base model; `0.6B-Base` for less VRAM |
| `--output_model_path` | `output` | checkpoint dir (per-epoch full copy) |
| `--train_jsonl` | *(required)* | the `*_with_codes.jsonl` from Step 2 |
| `--batch_size` | `2` | README example uses `32` (needs big GPU) |
| `--lr` | `2e-5` | README/#323 recommend **`2e-6`** for this |
| `--num_epochs` | `3` | README uses `10`; #323 saw plateau at 2–3 |
| `--speaker_name` | `speaker_test` | embedding id (stored at index 3000) |

**#323 tuning notes:** lr `2e-6`; reporter used a **higher LR specifically for the text/codec embeddings** to push language adaptation while keeping the backbone stable. If loss plateaus, the fix was **more/cleaner data**, not more epochs.

---

## Step 4 — Test inference

```python
from qwen_tts import Qwen3TTSModel
m = Qwen3TTSModel.from_pretrained("output_bengali")
wavs, sr = m.generate_custom_voice(text="আপনি কেমন আছেন?", speaker_name="bn_speaker01")
```

Listen for: native Bengali rhythm (not Hindi), correct vowels/conjuncts, naturalness. Iterate on data if accent persists.

---

## Step 5 — Wire the Bengali model into Voicebox

Once you have a good `output_bengali` checkpoint, expose it in this app. Voicebox already routes Qwen via `backend/backends/pytorch_backend.py` + the registry in `backend/backends/__init__.py`. Two options:

**A. Quick local swap (test only):** point a Qwen `ModelConfig.hf_repo_id` (or a local path) at your fine-tuned checkpoint and add `"bn"` to the Qwen `languages=[...]` list, plus map it in `LANGUAGE_CODE_TO_NAME` (`"bn": "bengali"`).

**B. Proper integration (recommended):** add Bengali as its own engine/profile entry so it doesn't shadow stock Qwen. Required edits (see `.agents/skills/add-tts-engine/SKILL.md` + `docs/content/docs/developer/tts-engines.mdx`):
- `backend/models.py` — add `bn` to the language regex (3 spots: `VoiceProfileCreate`, `GenerationRequest`, `SpeakRequest`).
- `backend/backends/__init__.py` — register the checkpoint as a `ModelConfig` (engine `qwen` or a new id), add `bn` to its `languages`; add `"bn"` to `LANGUAGE_CODE_TO_NAME`.
- `app/src/lib/constants/languages.ts` — add `bn: 'Bengali'` to `ALL_LANGUAGES` and to the engine's list in `ENGINE_LANGUAGES`.
- Frontend (if new engine): `app/src/lib/api/types.ts`, `EngineModelSelector.tsx`, `useGenerationForm.ts`, `ModelManagement.tsx`.
- `backend/requirements.txt` / `build_binary.py` if new deps or bundling.

Then `just dev` → create a profile with language **Bengali** → generate.

---

## Realistic plan

1. **Collect data** — 1 speaker, target tens of hours of clean Bengali wav + accurate transcripts. (Biggest lever per #323.)
2. **Rent a GPU** (≥24 GB; A100 ideal) — local 1650 Ti cannot train.
3. Run Steps 1→4, **raw Bengali script first**, lr `2e-6`.
4. If Hindi accent persists → add more/diverse data; only then experiment with a Bengali G2P front-end.
5. When quality is acceptable → integrate via Step 5 into Voicebox.

**Open question tracked upstream:** native Bengali prosody vs the base model's Indic/Hindi bias is unresolved in #323 (no maintainer answer yet) — watch that issue for an official recommendation.
