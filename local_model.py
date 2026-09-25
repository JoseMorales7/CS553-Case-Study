import logging
import os
import spaces
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from config import ARTIMUSE_SCORE_QUESTION,LOCAL_MODEL, advice_question, LOCAL_MAX_TOKENS
from critique import Critique

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INPUT_SIZE = 448

# First letter selects the block of 25, second letter is the offset within it.
_TOKEN_BASE = {"a": 0, "c": 26, "d": 51, "e": 76}

_model = None
_tokenizer = None


def aestoken2score(token: str) -> int | None:
    # decode ArtiMuse's two-letter score code, e.g. 'dc' -> 53
    # return None if receiving unexpected output
    token = token.strip().lower()
    if len(token) < 2:
        return None
    first, second = token[0], token[1]
    if first not in _TOKEN_BASE or not ("a" <= second <= "z"):
        return None
    score = _TOKEN_BASE[first] + (ord(second) - ord("a"))
    return score if 0 <= score <= 100 else None


def load_image(image_path: str) -> torch.Tensor:
    # ArtiMuse was trained on 448px tiles, so we resize to that size
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((INPUT_SIZE, INPUT_SIZE), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    with Image.open(image_path) as image:
        return transform(image).unsqueeze(0)


def _device_and_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.bfloat16
    return "cpu", torch.float32


def _artimuse_class():
    # Fetch ArtiMuse's bundled model class, patched for transformers 5.x
    cls = get_class_from_dynamic_module(
        "modeling_internvl_chat.InternVLChatModel", LOCAL_MODEL
    )
    if not hasattr(cls, "all_tied_weights_keys"):
        cls.all_tied_weights_keys = {}
    return cls


def _ensure_loaded():
    # Load the 15.9GB checkpoint once at import
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    device, dtype = _device_and_dtype()
    logger.info("Loading %s on %s (%s)", LOCAL_MODEL, device, dtype)
    _model = (
        _artimuse_class()
        .from_pretrained(
            LOCAL_MODEL,
            dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            use_flash_attn=False,
        )
        .eval()
        .to(device)
    )
    _tokenizer = AutoTokenizer.from_pretrained(
        LOCAL_MODEL, trust_remote_code=True, use_fast=False
    )
    return _model, _tokenizer


@spaces.GPU(duration=120)
def local_critique(image_path, aspect, temperature, top_p) -> Critique:
    model, tokenizer = _ensure_loaded()
    device, dtype = _device_and_dtype()
    pixel_values = load_image(image_path).to(dtype).to(device)

    # Scoring is a single forward pass: we only need the first generated token.
    score_config = dict(max_new_tokens=4, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    raw_score = model.chat(tokenizer, pixel_values, ARTIMUSE_SCORE_QUESTION, score_config)
    score = aestoken2score(str(raw_score))
    if score is None:
        raise RuntimeError(f"ArtiMuse returned an unparseable score token: {raw_score!r}")

    text_config = dict(
        max_new_tokens=LOCAL_MAX_TOKENS,
        do_sample=temperature > 0,
        temperature=max(float(temperature), 0.01),
        top_p=float(top_p),
        pad_token_id=tokenizer.eos_token_id,
    )
    evaluation = model.chat(tokenizer, pixel_values, advice_question(aspect), text_config)

    return Critique(
        score=score,
        evaluation=str(evaluation),
        model_name=LOCAL_MODEL,
        route="Local",
    )


# Ensure the model is loaded at boot time of Space, not on first request
if os.environ.get("SPACE_ID"):
    _ensure_loaded()
