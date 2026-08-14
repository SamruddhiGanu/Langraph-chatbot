# graph/blog/reducer.py — merge sections → decide images → generate & place images

import os
import re
import base64
import json
from pathlib import Path
from datetime import date

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from graph.llm import get_llm
from graph.blog.schemas import GlobalImagePlan

# ---------------------------------------------------------------------------
# 1. merge_content — sort + stitch sections
# ---------------------------------------------------------------------------

def merge_content(state: dict) -> dict:
    """Sort accumulated sections by index and stitch into one markdown document."""
    sections = state.get("sections") or []
    plan_dict = state.get("plan") or {}
    topic = state.get("topic", "Blog Post")
    title = plan_dict.get("title", topic)

    # Sort: intro (-1) first, conclusion (9999) last, body sections by index
    sorted_sections = sorted(sections, key=lambda t: t[0])

    header = f"# {title}\n\n"
    body = "\n\n---\n\n".join(md for _, md in sorted_sections)
    merged = header + body

    return {"merged_md": merged}


# ---------------------------------------------------------------------------
# 2. decide_images — LLM decides where to place images
# ---------------------------------------------------------------------------

DECIDE_IMAGES_SYSTEM = """You are an image placement planner for a blog post.

Given the blog markdown, decide where 1-3 images would most enhance the reader experience.
For each image:
- Choose a descriptive placeholder like {{IMAGE_1}}, {{IMAGE_2}}, etc.
- Write a detailed, vivid image generation prompt (describe visual style, subject, colours).
- Specify appropriate alt text.
- Indicate which section heading the image should appear after.

Avoid placing images in the introduction unless essential. Max 3 images total.
"""

def decide_images(state: dict) -> dict:
    """Decide image placements and generate placeholder-annotated markdown."""
    merged_md = state.get("merged_md", "")

    planner = get_llm("writer", temperature=0.3).with_structured_output(GlobalImagePlan)
    try:
        image_plan = planner.invoke([
            SystemMessage(content=DECIDE_IMAGES_SYSTEM),
            HumanMessage(content=f"Blog markdown:\n\n{merged_md[:5000]}"),
        ])
        specs = [s.model_dump() for s in image_plan.images]
    except Exception as e:
        print(f"[reducer] decide_images failed: {e}")
        specs = []

    # Insert placeholders into markdown at specified section locations
    md_with_placeholders = merged_md
    for spec in specs:
        placeholder = spec.get("placeholder", "")
        hint = spec.get("placement_hint", "")
        if placeholder and hint:
            # Find the section heading and insert placeholder after it
            pattern = re.compile(
                rf"(##\s+{re.escape(hint)}[^\n]*\n)",
                re.IGNORECASE
            )
            if pattern.search(md_with_placeholders):
                md_with_placeholders = pattern.sub(
                    rf"\1\n> {placeholder}\n\n", md_with_placeholders, count=1
                )
            else:
                # Append at end if heading not found
                md_with_placeholders += f"\n\n> {placeholder}\n"

    return {
        "md_with_placeholders": md_with_placeholders,
        "image_specs": specs,
    }


# ---------------------------------------------------------------------------
# 3. generate_and_place_images — call free Pollinations AI image generation
# ---------------------------------------------------------------------------

def _generate_image_bytes(prompt: str) -> bytes | None:
    """Generate an image via Pollinations AI. Returns PNG bytes or None."""
    import requests
    import urllib.parse
    try:
        # Encode prompt safely for GET URL
        sanitized_prompt = urllib.parse.quote(prompt[:250])
        # Pollinations AI flux model endpoint
        url = f"https://image.pollinations.ai/prompt/{sanitized_prompt}?width=1024&height=1024&nologo=true&private=true"
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            return resp.content
    except Exception as e:
        print(f"[reducer] Image generation error: {e}")
    return None


def generate_and_place_images(state: dict) -> dict:
    """Generate each image and replace placeholders with actual markdown image tags."""
    md = state.get("md_with_placeholders") or state.get("merged_md", "")
    specs = state.get("image_specs") or []

    images_dir = Path("data/images")
    images_dir.mkdir(parents=True, exist_ok=True)

    topic_slug = re.sub(r"[^\w]+", "_", state.get("topic", "blog"))[:30]
    today = state.get("as_of") or str(date.today())

    for i, spec in enumerate(specs):
        placeholder = spec.get("placeholder", f"{{{{IMAGE_{i+1}}}}}")
        prompt = spec.get("prompt", "")
        alt_text = spec.get("alt_text", f"Image {i+1}")

        if not prompt:
            continue

        img_bytes = _generate_image_bytes(prompt)
        if img_bytes:
            filename = f"{topic_slug}_{today}_img{i+1}.png"
            img_path = images_dir / filename
            img_path.write_bytes(img_bytes)

            # Replace placeholder with relative markdown image reference
            img_tag = f"![{alt_text}](data/images/{filename})"
            md = md.replace(f"> {placeholder}", img_tag)
            md = md.replace(placeholder, img_tag)
        else:
            # Graceful fallback: replace placeholder with a note
            fallback = f"*[Image: {alt_text} — generation unavailable]*"
            md = md.replace(f"> {placeholder}", fallback)
            md = md.replace(placeholder, fallback)

    # Append blog output as an AIMessage so it lands in conversation history
    final_messages = [AIMessage(content=md)]

    return {
        "final": md,
        "messages": final_messages,
    }
