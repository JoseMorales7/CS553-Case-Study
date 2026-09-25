import base64
import re
import io
from PIL import Image

from huggingface_hub import InferenceClient

from config import EVALUATION_HEADING, REMOTE_MODEL, REMOTE_PROVIDER, remote_prompt, REMOTE_MAX_TOKENS
from critique import Critique

SCORE_PATTERN = re.compile(r"##\s*Score:\s*(\d+)\s*/\s*100")
HEADING_PATTERN = re.compile(rf"###\s*{re.escape(EVALUATION_HEADING)}\s*", re.IGNORECASE)
MAX_IMAGE_SIDE = 768

def image_as_data_url(image_path: str) -> str:
    # downscale before sending to model to reduce tokens
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        if max(image.size) > MAX_IMAGE_SIDE:
            image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=88)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"


def parse_response(text: str) -> tuple[int | None, str]:
    # split the model markdown into (score, evaluation body)
    # notice the Qwen3 output can be duplicated sometime
    # so we only take the first part of the output
    headings = list(SCORE_PATTERN.finditer(text))
    if len(headings) > 1:
        text = text[: headings[1].start()]

    score = int(headings[0].group(1)) if headings else None

    body = HEADING_PATTERN.split(text, maxsplit=1)
    evaluation = body[1].strip() if len(body) > 1 else text.strip()

    repeat = HEADING_PATTERN.search(evaluation)
    if repeat:
        evaluation = evaluation[: repeat.start()].strip()

    return score, evaluation


def remote_critique(image_path, aspect, temperature, top_p, hf_token) -> Critique:
    client = InferenceClient(provider=REMOTE_PROVIDER, token=hf_token)
    response = client.chat.completions.create(
        model=REMOTE_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_as_data_url(image_path)}},
                    {"type": "text", "text": remote_prompt(aspect)},
                ],
            }
        ],
        max_tokens=REMOTE_MAX_TOKENS,
        temperature=temperature,
        top_p=top_p,
    )

    choice = response.choices[0]
    content = (choice.message.content or "").strip()

    # catch the case where the reasoning exhausts the token budget
    if not content:
        raise RuntimeError(
            f"Hosted model returned no visible content (finish_reason="
            f"{choice.finish_reason}); the token budget was likely exhausted by "
            "internal reasoning."
        )

    score, evaluation = parse_response(content)
    return Critique(score=score, evaluation=evaluation, model_name=REMOTE_MODEL, route="Hosted")
