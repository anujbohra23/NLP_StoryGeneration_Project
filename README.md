# Final Connected Children's Storybook Pipeline

This is the streamlined final project folder. It keeps only the important components needed for the project demo/submission:

1. **Story generation** using the fine-tuned Mistral LoRA checkpoint.
2. **Scene extraction** for 6 storybook frames.
3. **Image prompt generation** for each scene.
4. **Optional image generation** using Hugging Face Inference API.
5. **Saved output artifacts** in one folder: story, story JSON, story bible, scenes, prompts, images, and manifest.

## Why this file was chosen as the final story generator

The project had multiple story-generation notebooks. The best final choice is the fine-tuned Mistral checkpoint path because it represents the trained model output, not just baseline prompting or preprocessing. The notebooks are useful for experimentation, but the final submission/demo needs one clean runnable script.

## Folder Structure

```text
final_storybook_project/
├── src/
│   └── final_storybook_pipeline.py
├── checkpoints/
│   └── Checkpoint_100/              # LoRA adapter files
├── examples/
│   └── sample_existing_story.json
├── outputs/
├── requirements.txt
└── README.md
```

## Install

```bash
pip install -r requirements.txt
```

## Run story + prompts only

Use this first because it does not require image generation:

```bash
python src/final_storybook_pipeline.py \
  --theme "kindness" \
  --out outputs/kindness_demo \
  --skip-images
```

Expected output files:

```text
outputs/kindness_demo/story.txt
outputs/kindness_demo/story.json
outputs/kindness_demo/story_bible.json
outputs/kindness_demo/scenes.json
outputs/kindness_demo/image_prompts.json
outputs/kindness_demo/manifest.json
```

## Run with an existing story

This is useful if you already generated a story and only want to create scenes/prompts/images:

```bash
python src/final_storybook_pipeline.py \
  --input-story examples/sample_existing_story.json \
  --out outputs/sample_story \
  --skip-images
```

## Run full story + image generation

```bash
export HF_TOKEN=your_huggingface_token

python src/final_storybook_pipeline.py \
  --theme "teamwork" \
  --out outputs/teamwork_demo
```

The generated images will be saved as:

```text
outputs/teamwork_demo/images/scene_01.png
outputs/teamwork_demo/images/scene_02.png
...
```

## What was removed from the final streamlined version

- Colab/Kaggle setup cells
- preprocessing experiments
- duplicate fine-tuning notebooks
- comparison/evaluation notebook code
- separate disconnected image-only pipeline
- demo-only hardcoded story logic

## What remains

- the trained LoRA adapter checkpoint
- one runnable final pipeline
- one dependency file
- one example input file
- clean outputs and logs

## Notes

The first run may take time because the base Mistral model must be downloaded. For machines without enough GPU memory, use `--input-story` and `--skip-images` to test the connected story-to-scenes-to-prompts pipeline without loading the full story model.
