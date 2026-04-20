# Story-to-Image Pipeline

This pipeline takes a JSON input with:
- `title`
- `characters`
- `story`

It produces:
1. `story_bible.json`
2. `scenes.json`
3. `character_sheet_prompt.json`
4. `prompts.json`
5. `images/scene_01.png ... scene_06.png`
6. `manifest.json`

## Install

```bash
pip install -r requirements_story_pipeline.txt
```

## Set token

```bash
export HF_TOKEN=your_huggingface_token
```

## Run with the provided Buddy Bears example

```bash
python story_pipeline.py --input buddy_bears_input.json --out outputs/buddy_bears
```

## Run only up to prompt generation

```bash
python story_pipeline.py --input buddy_bears_input.json --out outputs/buddy_bears --skip-images
```

## Useful model switches

### LLM
- `meta-llama/Meta-Llama-3-8B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct`

### Image
- `black-forest-labs/FLUX.1-schnell`
- `stabilityai/stable-diffusion-2-1`

Example:

```bash
python story_pipeline.py \
  --input buddy_bears_input.json \
  --out outputs/buddy_bears \
  --llm-model Qwen/Qwen2.5-7B-Instruct \
  --image-model black-forest-labs/FLUX.1-schnell
```

## Notes
- The pipeline is modular, so you can inspect and evaluate each stage independently.
- For stronger visual consistency, keep the generated `story_bible.json` fixed and reuse it across reruns.
- If you later want even stronger identity consistency, add an IP-Adapter or reference-image stage before generating the final scenes.
