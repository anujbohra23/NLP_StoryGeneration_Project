import os
import re
import json
import time
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List

from huggingface_hub import InferenceClient


DEFAULT_SYSTEM_STYLE = (
    "children's picture-book illustration, soft watercolor, pastel palette, warm light, "
    "rounded friendly shapes, expressive faces, cozy whimsical environment, safe for young children"
)

DEFAULT_NEGATIVE_PROMPT = (
    "photorealistic, horror, scary, violence, text, watermark, logo, extra limbs, deformed face, "
    "blurry, dark mood, realistic adult proportions, inconsistent clothing, duplicate characters"
)


@dataclass
class SceneCard:
    scene_id: int
    scene_title: str
    narrative_role: str
    plot_beat: str
    visual_summary: str
    characters_present: List[str]
    location: str
    action: str
    mood: str
    continuity_notes: List[str]


class StoryIllustrationPipeline:
    def __init__(
        self,
        hf_token: str,
        llm_model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        image_model: str = "black-forest-labs/FLUX.1-schnell",
        llm_provider: str | None = None,
        image_provider: str | None = None,
        image_width: int = 1024,
        image_height: int = 768,
    ):
        if not hf_token:
            raise ValueError("HF token is required. Set HF_TOKEN or pass --hf-token.")

        self.hf_token = hf_token
        self.llm_model = llm_model
        self.image_model = image_model
        self.image_width = image_width
        self.image_height = image_height

        self.llm_client = InferenceClient(api_key=hf_token, provider=llm_provider)
        self.image_client = InferenceClient(api_key=hf_token, provider=image_provider)

    def _chat_json(self, system_prompt: str, user_prompt: str, max_tokens: int = 1800) -> Dict[str, Any]:
        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=max_tokens,
        )
        text = response.choices[0].message.content
        return self._parse_json_from_text(text)

    @staticmethod
    def _parse_json_from_text(text: str) -> Dict[str, Any]:
        text = text.strip()
        codeblock = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", text, flags=re.S)
        if codeblock:
            text = codeblock.group(1)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.S)
            if not match:
                raise
            return json.loads(match.group(1))

    def build_story_bible(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        system_prompt = (
            "You are a story-illustration planner. Return valid JSON only. "
            "Extract a canonical story bible for consistent children's book images. "
            "Never invent conflicting character details. If the story does not specify a detail, choose a gentle, "
            "child-friendly default and keep it stable across every scene."
        )
        user_prompt = f"""
Create a story bible from the following input.
Return JSON with this exact top-level schema:
{{
  "title": str,
  "theme": str,
  "age_group": str,
  "art_direction": {{
    "style": str,
    "palette": str,
    "lighting": str,
    "camera_feel": str,
    "background_rules": [str],
    "negative_rules": [str]
  }},
  "characters": [
    {{
      "name": str,
      "species": str,
      "gender_presentation": str,
      "age_group": str,
      "core_personality": [str],
      "physical_description": {{
        "fur_or_skin": str,
        "hair": str,
        "eyes": str,
        "body_size": str,
        "signature_features": [str]
      }},
      "default_outfit": str,
      "consistency_rules": [str]
    }}
  ],
  "world": {{
    "primary_setting": str,
    "secondary_settings": [str],
    "time_progression": str,
    "visual_motifs": [str]
  }}
}}

Input:
{json.dumps(payload, indent=2)}
"""
        return self._chat_json(system_prompt, user_prompt, max_tokens=1800)

    def extract_scenes(self, payload: Dict[str, Any], story_bible: Dict[str, Any], num_scenes: int = 6) -> List[Dict[str, Any]]:
        system_prompt = (
            "You are a scene planner for children's picture books. Return valid JSON only. "
            "Split the story into exactly the requested number of scenes. "
            "Each scene must be visually distinct, chronologically ordered, and appropriate for illustration. "
            "Preserve all named characters and major events."
        )
        user_prompt = f"""
Split the story into exactly {num_scenes} scenes.
Return JSON in this schema:
{{
  "scenes": [
    {{
      "scene_id": int,
      "scene_title": str,
      "narrative_role": str,
      "plot_beat": str,
      "visual_summary": str,
      "characters_present": [str],
      "location": str,
      "action": str,
      "mood": str,
      "continuity_notes": [str]
    }}
  ]
}}

Rules:
- Use exactly {num_scenes} scenes.
- Include opening, journey, social interaction, high point, challenge, and resolution if possible.
- Keep each scene easy to draw as one image.
- Mention only characters that are visibly present.
- continuity_notes should mention what must stay consistent from prior scenes.

Story input:
{json.dumps(payload, indent=2)}

Story bible:
{json.dumps(story_bible, indent=2)}
"""
        data = self._chat_json(system_prompt, user_prompt, max_tokens=2200)
        scenes = data["scenes"]
        if len(scenes) != num_scenes:
            raise ValueError(f"Expected {num_scenes} scenes but received {len(scenes)}")
        return scenes

    def create_character_sheet_prompt(self, story_bible: Dict[str, Any]) -> Dict[str, str]:
        names = [c["name"] for c in story_bible.get("characters", [])]
        character_descriptions = []
        for c in story_bible.get("characters", []):
            d = c.get("physical_description", {})
            character_descriptions.append(
                f'{c["name"]}: {c.get("species", "character")}, {c.get("gender_presentation", "childlike")}, '
                f'{d.get("fur_or_skin", "soft brown fur")}, {d.get("eyes", "bright eyes")}, '
                f'{c.get("default_outfit", "simple cute outfit")}, signature features: {", ".join(d.get("signature_features", [])) or "none"}.'
            )
        prompt = (
            f"Character sheet for a children's picture book titled '{story_bible.get('title', '')}'. "
            f"Show {', '.join(names)} standing side by side, full body, front-facing and three-quarter pose, neutral background, "
            f"clean turnaround style, consistent proportions, {story_bible['art_direction']['style']}, "
            f"{story_bible['art_direction']['palette']}, warm inviting look. "
            + " ".join(character_descriptions)
        )
        return {
            "prompt": prompt,
            "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
        }

    def prompts_from_scenes(self, story_bible: Dict[str, Any], scenes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        system_prompt = (
            "You write high-quality prompts for children's storybook illustration models. Return valid JSON only. "
            "Every prompt must preserve character identity, outfit, mood, and global art style from the story bible."
        )
        user_prompt = f"""
Using the story bible and scene cards below, produce one image prompt per scene.
Return JSON with this schema:
{{
  "prompts": [
    {{
      "scene_id": int,
      "prompt": str,
      "negative_prompt": str,
      "size_hint": str
    }}
  ]
}}

Prompt rules:
- Keep wording visual, concrete, and child-friendly.
- Re-state the stable appearance of each main character whenever present.
- Re-state the overall style in every prompt.
- Mention camera framing and background.
- Keep the prompt single-image and non-cinematic.
- Negative prompt should remove horror, photorealism, text, watermarks, and inconsistent anatomy.

Story bible:
{json.dumps(story_bible, indent=2)}

Scenes:
{json.dumps(scenes, indent=2)}
"""
        data = self._chat_json(system_prompt, user_prompt, max_tokens=2600)
        prompts = data["prompts"]
        prompt_map = {p["scene_id"]: p for p in prompts}
        ordered = []
        for scene in scenes:
            item = prompt_map[scene["scene_id"]]
            ordered.append(item)
        return ordered

    def generate_images(self, prompts: List[Dict[str, Any]], out_dir: Path, guidance_scale: float = 3.5) -> List[str]:
        image_paths = []
        images_dir = out_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        for item in prompts:
            scene_id = item["scene_id"]
            image = self.image_client.text_to_image(
                prompt=item["prompt"],
                negative_prompt=item.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT),
                model=self.image_model,
                width=self.image_width,
                height=self.image_height,
                guidance_scale=guidance_scale,
            )
            img_path = images_dir / f"scene_{scene_id:02d}.png"
            image.save(img_path)
            image_paths.append(str(img_path))
            time.sleep(0.5)
        return image_paths

    def run(self, payload: Dict[str, Any], out_dir: str, num_scenes: int = 6, generate_images: bool = True) -> Dict[str, Any]:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        (out_path / "input.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

        story_bible = self.build_story_bible(payload)
        (out_path / "story_bible.json").write_text(json.dumps(story_bible, indent=2), encoding="utf-8")

        scenes = self.extract_scenes(payload, story_bible, num_scenes=num_scenes)
        (out_path / "scenes.json").write_text(json.dumps(scenes, indent=2), encoding="utf-8")

        character_sheet = self.create_character_sheet_prompt(story_bible)
        (out_path / "character_sheet_prompt.json").write_text(json.dumps(character_sheet, indent=2), encoding="utf-8")

        prompts = self.prompts_from_scenes(story_bible, scenes)
        (out_path / "prompts.json").write_text(json.dumps(prompts, indent=2), encoding="utf-8")

        image_paths = []
        if generate_images:
            image_paths = self.generate_images(prompts, out_path)

        manifest = {
            "title": payload.get("title"),
            "llm_model": self.llm_model,
            "image_model": self.image_model,
            "num_scenes": num_scenes,
            "art_style": story_bible.get("art_direction", {}).get("style"),
            "output_files": {
                "input": str(out_path / "input.json"),
                "story_bible": str(out_path / "story_bible.json"),
                "scenes": str(out_path / "scenes.json"),
                "character_sheet_prompt": str(out_path / "character_sheet_prompt.json"),
                "prompts": str(out_path / "prompts.json"),
                "images": image_paths,
            },
        }
        (out_path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest


def load_input(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_demo_input() -> Dict[str, Any]:
    return {
        "title": "Buddy Bears",
        "characters": ["Timmy", "Benny"],
        "story": (
            "Once upon a time in Bearville Forest lived two little bear cubs named Timmy and Benny. "
            "They were friends who loved playing together, exploring their home, and making new friends. "
            "One sunny morning, they decided to take a picnic lunch and go on a hike up Mount Peak. "
            "They packed sandwiches, juice boxes, and lots of yummy treats! "
            "As they climbed the mountain trail, they met many animals along the way. "
            "They shared some delicious treats and even made some new friends like Tilly the Tortoise and Freddie the Fox. "
            "Everyone was so happy, playing games, sharing stories, and learning from one another. "
            "After hours of fun, they reached the top of the mountain where the most beautiful view of Bearville Forest awaited them. "
            "Timmy and Benny sat down to enjoy their delicious picnic, watching the sunset with their new friends. "
            "They felt grateful for the adventure and excited about all the new memories they had created that day. "
            "Suddenly, nightfall arrived, and it was time for everyone to head back home. "
            "As they started their descent, Benny began to feel tired and his tummy was rumbling. "
            "He asked Timmy if they could share some food since they still had plenty left. "
            "Without hesitation, kind-hearted Timmy happily shared his sandwiches and treats with Benny. "
            "Together, they continued their journey home, feeling warm and content. "
            "Moral: Sharing is caring! When we help others, we not only make them happy but also create beautiful memories that last a lifetime. "
            "By being kind and generous, we spread love and happiness around us."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="End-to-end story-to-scene-to-prompt-to-image pipeline")
    parser.add_argument("--input", type=str, default=None, help="Path to input JSON with title, characters, story")
    parser.add_argument("--out", type=str, default="outputs/buddy_bears_run", help="Output directory")
    parser.add_argument("--hf-token", type=str, default=os.getenv("HF_TOKEN"), help="Hugging Face token")
    parser.add_argument("--llm-model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--image-model", type=str, default="black-forest-labs/FLUX.1-schnell")
    parser.add_argument("--num-scenes", type=int, default=6)
    parser.add_argument("--skip-images", action="store_true")
    parser.add_argument("--llm-provider", type=str, default=None, help="Optional provider, e.g. hf-inference, together, groq")
    parser.add_argument("--image-provider", type=str, default=None, help="Optional provider, e.g. hf-inference, fal-ai, replicate")
    args = parser.parse_args()

    payload = load_input(args.input) if args.input else build_demo_input()

    pipeline = StoryIllustrationPipeline(
        hf_token=args.hf_token,
        llm_model=args.llm_model,
        image_model=args.image_model,
        llm_provider=args.llm_provider,
        image_provider=args.image_provider,
    )

    manifest = pipeline.run(
        payload=payload,
        out_dir=args.out,
        num_scenes=args.num_scenes,
        generate_images=not args.skip_images,
    )

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
