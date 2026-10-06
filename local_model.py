# Large portions of this file was editted by codex to make the new model we chose work.
# Query: Modify the local_model.py file to switch the model to HuggingFaceTB/SmolVLM-256M-Instruct.
# Ensure that the correct functions are used to interact with the model.

import logging
import re
from threading import RLock, Thread

from PIL import Image, ImageOps

from config import LOCAL_MODEL, LOCAL_SCORE_QUESTION, advice_question, LOCAL_MAX_TOKENS
from critique import Critique

logger = logging.getLogger(__name__)

_model = None
_processor = None
# Avoid duplicate checkpoint loads and concurrent generation on the same device.
_model_lock = RLock()


def parse_local_score(text: str) -> int | None:
    """Accept a numeric answer, optionally labeled or expressed out of 100."""
    match = re.fullmatch(
        r"(?:#{1,6}\s*)?(?:score\s*:\s*)?(\d{1,3})(?:\s*/\s*100)?\s*[.!]?",
        text.strip(),
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    score = int(match.group(1))
    return score if 0 <= score <= 100 else None


def _device_and_dtype():
    import torch

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return "cuda", dtype
    if torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def _ensure_loaded():
    global _model, _processor
    with _model_lock:
        if _model is not None and _processor is not None:
            return _model, _processor

        logger.info("Importing the local model inference dependencies")
        from transformers import AutoModelForImageTextToText, AutoProcessor

        device, dtype = _device_and_dtype()
        logger.info("Loading %s on %s (%s)", LOCAL_MODEL, device, dtype)
        processor = AutoProcessor.from_pretrained(
            LOCAL_MODEL, size={"longest_edge": 1024}
        )
        model = AutoModelForImageTextToText.from_pretrained(
            LOCAL_MODEL,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        ).eval().to(device)
        # Publish the cache only after both objects have loaded successfully.
        _model, _processor = model, processor
        return model, processor


def _generate(model, processor, image, question, max_new_tokens, temperature=0.0, top_p=0.9):
    import torch

    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": question},
        ],
    }]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    inputs = {
        name: tensor.to(
            device=model.device,
            dtype=model.dtype if tensor.is_floating_point() else tensor.dtype,
        )
        for name, tensor in inputs.items()
    }
    generation = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0}
    if temperature > 0:
        generation.update(temperature=float(temperature), top_p=float(top_p))

    with torch.inference_mode():
        generated = model.generate(**inputs, **generation)
    # Decoder-only generation includes the prompt; never parse its example text.
    answer_ids = generated[:, inputs["input_ids"].shape[-1]:]
    return processor.batch_decode(answer_ids, skip_special_tokens=True)[0].strip()


def local_critique(image_path, aspect, temperature, top_p) -> Critique:
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")

    with _model_lock:
        model, processor = _ensure_loaded()
        raw_score = _generate(model, processor, image, LOCAL_SCORE_QUESTION, 16)
        score = parse_local_score(raw_score)
        if score is None:
            raise RuntimeError(f"SmolVLM returned an invalid score: {raw_score!r}")

        evaluation = _generate(
            model, processor, image, advice_question(aspect), LOCAL_MAX_TOKENS,
            temperature, top_p,
        )
        if not evaluation:
            raise RuntimeError("SmolVLM returned no improvement suggestions.")

    return Critique(
        score=score,
        evaluation=evaluation,
        model_name=LOCAL_MODEL,
        route="Local",
    )


def preload_in_background():
    """Warm the checkpoint without blocking the web server's startup."""
    def load():
        try:
            _ensure_loaded()
            logger.info("Local model ready")
        except Exception:
            logger.exception("Local model preload failed; it will retry on first use")

    thread = Thread(target=load, name="local-model-preload", daemon=True)
    thread.start()
    return thread
