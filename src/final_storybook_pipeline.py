"""
Final connected pipeline for the Children's Story Generation project.

What this file does:
1. Generates a child-friendly story from a theme using the fine-tuned LoRA checkpoint when available.
2. Builds a story bible for character/style consistency.
3. Splits the story into visual scenes.
4. Creates image prompts for each scene.
5. Optionally generates images using Hugging Face Inference API.
6. Saves all outputs with clear logs.

Example:
python src/final_storybook_pipeline.py --theme "kindness" --out outputs/kindness_demo --skip-images

With image generation:
export HF_TOKEN=your_token
python src/final_storybook_pipeline.py --theme "kindness" --out outputs/kindness_demo
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
DEFAULT_ADAPTER_PATH = "checkpoints/Checkpoint_100"
DEFAULT_LLM_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
DEFAULT_IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"

DEFAULT_ART_STYLE = (
    "children's picture-book illustration, soft watercolor, pastel palette, warm light, "
    "rounded friendly shapes, expressive faces, cozy whimsical environment, safe for young children"
)

DEFAULT_NEGATIVE_PROMPT = (
    "photorealistic, horror, scary, violence, text, watermark, logo, extra limbs, "
    "deformed face, blurry, dark mood, inconsistent clothing, duplicate characters"
)

BANNED_WORDS = {
    "kill", "killed", "dead", "die", "blood", "weapon", "gun", "knife", "horror", "scary"
}


@dataclass
class StoryOutput:
    title: str
    theme: str
    characters: list
    story: str
    moral: str

    def to_text(self) -> str:
        return (
            f"Title: {self.title}\n\n"
            f"Characters: {', '.join(self.characters)}\n\n"
            f"Story:\n{self.story.strip()}\n\n"
            f"Moral: {self.moral.strip()}\n"
        )

    def to_payload(self, theme: str) -> Dict[str, Any]:
        return {
            "title": self.title,
            "theme": theme,
            "characters": self.characters,
            "story": self.story,
            "moral": self.moral,
        }


class StoryGenerator:
    """Loads the fine-tuned LoRA checkpoint and generates structured children stories."""

    def __init__(
        self,
        base_model: str = DEFAULT_BASE_MODEL,
        adapter_path: str = DEFAULT_ADAPTER_PATH,
        max_new_tokens: int = 700,
    ) -> None:
        self.base_model = base_model
        self.adapter_path = Path(adapter_path)
        self.max_new_tokens = max_new_tokens
        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError(
                "Story generation dependencies are missing. Install requirements.txt first."
            ) from exc

        if not self.adapter_path.exists():
            raise FileNotFoundError(f"LoRA adapter folder not found: {self.adapter_path}")

        print(f"[INIT] Loading tokenizer: {self.base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quant_config = None
        if torch.cuda.is_available():
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        print(f"[INIT] Loading base model: {self.base_model}")
        base = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            device_map="auto" if torch.cuda.is_available() else None,
            quantization_config=quant_config,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

        print(f"[INIT] Attaching LoRA adapter: {self.adapter_path}")
        self.model = PeftModel.from_pretrained(base, str(self.adapter_path))
        self.model.eval()
        print("[OK] Story model loaded.")

    @staticmethod
    def build_prompt(theme: str) -> str:
        return f"""Create a children's story for ages 6-8.
Theme: {theme}

Requirements:
- Use simple vocabulary and a warm tone.
- Include 2-3 named main characters.
- Make the story suitable for 6 illustrated scenes.
- Avoid scary, violent, dark, or adult content.
- End with a clear moral.

Return exactly this format:
Title: ...
Characters: ...
Story:
...
Moral: ..."""

    def generate(self, theme: str) -> StoryOutput:
        if self.model is None or self.tokenizer is None:
            self.load()

        import torch

        prompt = self.build_prompt(theme)
        messages = [{"role": "user", "content": prompt}]
        model_input = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(model_input, return_tensors="pt").to(self.model.device)

        print(f"[STORY] Generating story for theme: {theme}")
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.08,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = output_ids[0][inputs["input_ids"].shape[-1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        story = self.parse_story(text)
        self.validate_story(story)
        print(f"[OK] Story created: {story.title}")
        return story

    @staticmethod
    def parse_story(text: str) -> StoryOutput:
        title = _extract_field(text, "Title", fallback="A Little Adventure")
        characters_raw = _extract_field(text, "Characters", fallback="Mira, Rafi")
        story = _extract_block(text, "Story", "Moral", fallback=text)
        moral = _extract_field(text, "Moral", fallback="Kindness makes every adventure brighter.")
        characters = [c.strip() for c in re.split(r",| and ", characters_raw) if c.strip()][:3]
        if len(characters) < 2:
            characters = ["Mira", "Rafi"]
        return StoryOutput(title=title, characters=characters, story=story.strip(), moral=moral.strip())

    @staticmethod
    def validate_story(story: StoryOutput) -> None:
        text = story.to_text().lower()
        found = sorted(word for word in BANNED_WORDS if re.search(rf"\b{re.escape(word)}\b", text))
        if found:
            print(f"[WARN] Safety check found possible banned words: {found}")


def _extract_field(text: str, label: str, fallback: str) -> str:
    pattern = rf"{label}\s*:\s*(.+)"
    match = re.search(pattern, text, flags=re.I)
    return match.group(1).strip() if match else fallback


def _extract_block(text: str, start_label: str, end_label: str, fallback: str) -> str:
    pattern = rf"{start_label}\s*:\s*(.*?)(?:\n\s*{end_label}\s*:|$)"
    match = re.search(pattern, text, flags=re.I | re.S)
    return match.group(1).strip() if match else fallback


class IllustrationPipeline:
    """Builds scene cards, prompts, and optional images from a generated story."""

    def __init__(
        self,
        hf_token: Optional[str],
        llm_model: str = DEFAULT_LLM_MODEL,
        image_model: str = DEFAULT_IMAGE_MODEL,
        image_width: int = 1024,
        image_height: int = 768,
    ) -> None:
        self.hf_token = hf_token
        self.llm_model = llm_model
        self.image_model = image_model
        self.image_width = image_width
        self.image_height = image_height
        self.client = None
        if hf_token:
            from huggingface_hub import InferenceClient
            self.client = InferenceClient(api_key=hf_token)

    def build_story_bible(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "title": payload["title"],
            "theme": payload.get("theme", "kindness"),
            "age_group": "6-8",
            "art_direction": {
                "style": DEFAULT_ART_STYLE,
                "palette": "pastel, warm, gentle, high clarity",
                "lighting": "soft morning or sunset light",
                "camera_feel": "storybook page, medium-wide framing",
                "negative_rules": [DEFAULT_NEGATIVE_PROMPT],
            },
            "characters": [
                {
                    "name": name,
                    "species": "child or friendly animal, based on the story context",
                    "core_personality": ["kind", "curious", "expressive"],
                    "default_outfit": "simple colorful child-friendly outfit",
                    "consistency_rules": [
                        "Keep the same face, outfit colors, size, and friendly expression across scenes."
                    ],
                }
                for name in payload.get("characters", [])
            ],
            "world": {
                "primary_setting": "safe cozy storybook world",
                "visual_motifs": ["warm light", "rounded shapes", "gentle nature details"],
            },
        }

    def extract_scenes(self, payload: Dict[str, Any], num_scenes: int = 6) -> List[Dict[str, Any]]:
        story_text = payload["story"]
        sentences = re.split(r"(?<=[.!?])\s+", story_text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        chunks = _balanced_chunks(sentences, num_scenes)

        scene_roles = ["Opening", "Journey", "Interaction", "High Point", "Challenge", "Resolution"]
        scenes = []
        for idx, chunk in enumerate(chunks, start=1):
            beat = " ".join(chunk)
            scenes.append({
                "scene_id": idx,
                "scene_title": f"Scene {idx}: {scene_roles[idx-1] if idx <= len(scene_roles) else 'Story Beat'}",
                "narrative_role": scene_roles[idx-1] if idx <= len(scene_roles) else "Story Beat",
                "plot_beat": beat,
                "visual_summary": beat[:350],
                "characters_present": payload.get("characters", []),
                "location": "storybook setting from this part of the story",
                "action": beat[:220],
                "mood": "warm, gentle, child-friendly",
                "continuity_notes": ["Keep character designs and outfits consistent."],
            })
        print(f"[OK] Scenes extracted: {len(scenes)}")
        return scenes

    def prompts_from_scenes(self, story_bible: Dict[str, Any], scenes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        character_text = "; ".join(
            f"{c['name']} wearing {c['default_outfit']}" for c in story_bible.get("characters", [])
        )
        prompts = []
        for scene in scenes:
            prompt = (
                f"{DEFAULT_ART_STYLE}. Illustration for '{story_bible['title']}'. "
                f"{character_text}. {scene['visual_summary']} "
                f"Scene mood: {scene['mood']}. Medium-wide storybook composition, clean background, "
                f"consistent character design, no text on image."
            )
            prompts.append({
                "scene_id": scene["scene_id"],
                "prompt": prompt,
                "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
                "size_hint": f"{self.image_width}x{self.image_height}",
            })
        print(f"[OK] Image prompts created: {len(prompts)}")
        return prompts

    def generate_images(self, prompts: List[Dict[str, Any]], out_dir: Path) -> List[str]:
        if self.client is None:
            raise ValueError("HF token is required for image generation. Set HF_TOKEN or use --skip-images.")

        images_dir = out_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        image_paths = []
        for item in prompts:
            print(f"[IMAGE] Generating scene {item['scene_id']}...")
            image = self.client.text_to_image(
                prompt=item["prompt"],
                negative_prompt=item["negative_prompt"],
                model=self.image_model,
                width=self.image_width,
                height=self.image_height,
                guidance_scale=3.5,
            )
            path = images_dir / f"scene_{item['scene_id']:02d}.png"
            image.save(path)
            image_paths.append(str(path))
            print(f"[OK] Image saved: {path}")
            time.sleep(0.5)
        return image_paths


def _balanced_chunks(items: List[str], n: int) -> List[List[str]]:
    if not items:
        return [[""] for _ in range(n)]
    chunks = []
    for i in range(n):
        start = round(i * len(items) / n)
        end = round((i + 1) * len(items) / n)
        chunk = items[start:end] or [items[min(i, len(items) - 1)]]
        chunks.append(chunk)
    return chunks


def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INIT] Starting final connected storybook pipeline...")

    if args.input_story:
        payload = json.loads(Path(args.input_story).read_text(encoding="utf-8"))
        story_output = StoryOutput(
            title=payload.get("title", "Untitled Story"),
            characters=payload.get("characters", []),
            story=payload.get("story", ""),
            moral=payload.get("moral", ""),
        )
        theme = payload.get("theme", args.theme)
        print(f"[STORY] Loaded existing story: {story_output.title}")
    else:
        # 1. Generate story
        if args.story_backend == "local":
            story_generator = StoryGenerator(
            base_model=args.base_model,
            adapter_path=args.adapter_path,
            max_new_tokens=args.max_new_tokens,
        )
            story_output = story_generator.generate(args.theme)
        else:
            story_text = f""" Mira was a curious child who loved exploring the world around her, but she often liked doing things by herself. One bright morning, her teacher gave the class a special task: each student had to create something that showed the meaning of {args.theme}. Mira immediately decided she would make the best project in the class.

                On her way home, Mira met Rafi, who was carrying a box of colored paper, glue, and tiny wooden sticks. He smiled and asked if they could work together. Mira hesitated at first because she already had her own plan. But when a sudden breeze scattered her papers across the path, Rafi quickly helped her collect every sheet before they blew away.

                Mira realized that Rafi had ideas she had never considered. He suggested building a small paper village where every house represented a helpful action. Mira designed a school, Rafi made a garden, and together they created a little bridge connecting everything. As they worked, they disagreed about the colors, the shapes, and where each piece should go, but they learned to listen instead of arguing.

                By sunset, their project looked better than either of them could have made alone. The village had bright houses, smiling paper people, and a tiny sign that read: “Small acts can build something big.” Mira felt proud, not because the project was perfect, but because it carried both of their ideas.

                The next day, their teacher asked them to explain their work. Mira said that she had learned {args.theme} was not just a word. It meant sharing ideas, helping when things go wrong, and making space for someone else’s voice. Rafi added that the best creations often happen when people trust each other.

                From that day on, Mira still loved having her own ideas, but she no longer wanted to do everything alone. She understood that {args.theme} can turn an ordinary task into a meaningful adventure.
                """.strip()

            story_output = StoryOutput(
                title=f"The Bridge of {args.theme.title()}",
                theme=args.theme,
                characters=["Mira", "Rafi"],
                story=story_text,
                moral=f"{args.theme.title()} becomes stronger when we listen, help, and create together."
            )



        theme = args.theme
        payload = story_output.to_payload(theme)

    (out_dir / "story.txt").write_text(story_output.to_text(), encoding="utf-8")
    (out_dir / "story.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[SAVE] Story saved: {out_dir / 'story.txt'}")

    illustrator = IllustrationPipeline(
        hf_token=args.hf_token,
        llm_model=args.llm_model,
        image_model=args.image_model,
        image_width=args.image_width,
        image_height=args.image_height,
    )

    story_bible = illustrator.build_story_bible(payload)
    (out_dir / "story_bible.json").write_text(json.dumps(story_bible, indent=2), encoding="utf-8")
    print(f"[SAVE] Story bible saved: {out_dir / 'story_bible.json'}")

    scenes = illustrator.extract_scenes(payload, num_scenes=args.num_scenes)
    (out_dir / "scenes.json").write_text(json.dumps(scenes, indent=2), encoding="utf-8")
    print(f"[SAVE] Scenes saved: {out_dir / 'scenes.json'}")

    prompts = illustrator.prompts_from_scenes(story_bible, scenes)
    (out_dir / "image_prompts.json").write_text(json.dumps(prompts, indent=2), encoding="utf-8")
    print(f"[SAVE] Image prompts saved: {out_dir / 'image_prompts.json'}")

    image_paths: List[str] = []
    if args.skip_images:
        print("[SKIP] Image generation skipped. Prompts are ready for generation.")
    else:
        image_paths = illustrator.generate_images(prompts, out_dir)

    manifest = {
        "theme": theme,
        "title": story_output.title,
        "base_model": args.base_model,
        "adapter_path": args.adapter_path,
        "image_model": args.image_model,
        "num_scenes": args.num_scenes,
        "output_files": {
            "story_txt": str(out_dir / "story.txt"),
            "story_json": str(out_dir / "story.json"),
            "story_bible": str(out_dir / "story_bible.json"),
            "scenes": str(out_dir / "scenes.json"),
            "image_prompts": str(out_dir / "image_prompts.json"),
            "images": image_paths,
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[DONE] Pipeline complete. Manifest saved: {out_dir / 'manifest.json'}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final connected story + image pipeline")
    parser.add_argument(
    "--story-backend",
    type=str,
    default="template",
    choices=["template", "local"],
    help="Use template for fast generation or local for fine-tuned Mistral."
)
    parser.add_argument("--theme", type=str, default="kindness", help="Story theme, e.g., kindness or teamwork")
    parser.add_argument("--input-story", type=str, default=None, help="Optional existing story JSON")
    parser.add_argument("--out", type=str, default="outputs/storybook_run", help="Output folder")
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-path", type=str, default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=700)
    parser.add_argument("--num-scenes", type=int, default=6)
    parser.add_argument("--hf-token", type=str, default=os.getenv("HF_TOKEN"))
    parser.add_argument("--llm-model", type=str, default=DEFAULT_LLM_MODEL)
    parser.add_argument("--image-model", type=str, default=DEFAULT_IMAGE_MODEL)
    parser.add_argument("--image-width", type=int, default=1024)
    parser.add_argument("--image-height", type=int, default=768)
    parser.add_argument("--skip-images", action="store_true", help="Create story/scenes/prompts only")
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
