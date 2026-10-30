# Chooses which model handles a request, and what happens when one fails

import logging
from typing import Callable

import gradio as gr
from critique import Critique
from hf_auth import resolve_token
from local_model import local_critique
from remote_model import remote_critique

logger = logging.getLogger(__name__)


def _accepting_requests() -> bool:
    return True


_capacity_check = _accepting_requests


def set_capacity_check(check: Callable[[], bool]) -> None:
    """Install the application-owned capacity check without creating a monitor."""
    global _capacity_check
    _capacity_check = check


def score_artwork(
    image_path,
    aspect,
    temperature,
    top_p,
    use_local_model,
    hf_token,
):
    # Return (markdown, status) for the UI
    if not _capacity_check():
        raise gr.Error("The server is currently near capacity. Please try again shortly.")

    if not image_path:
        raise gr.Error("Please upload an image before requesting a critique.")

    try:
        if use_local_model:
            result = _local_with_fallback(
                image_path, aspect, temperature, top_p, hf_token
            )
            return result.to_markdown(), result.to_status()

        token = resolve_token(hf_token)
        if not token:
            gr.Warning(
                "No Hugging Face token was provided. "
                "Automatically switching to the local model."
            )
            result = local_critique(image_path, aspect, temperature, top_p)
            result.route = "Local (no Hugging Face token)"
            return result.to_markdown(), result.to_status()

        try:
            result = remote_critique(image_path, aspect, temperature, top_p, token)
            return result.to_markdown(), result.to_status()
        except Exception:
            logger.warning("Hosted model failed, failing over to local", exc_info=True)
            gr.Warning(
                "The hosted model is unavailable. "
                "Automatically switching to the local model."
            )
            result = local_critique(image_path, aspect, temperature, top_p)
            result.route = "Failover (hosted unavailable)"
            return result.to_markdown(), result.to_status()

    except gr.Error:
        raise
    except Exception as error:
        raise gr.Error(f"The model could not score this image: {error}") from error


def _local_with_fallback(image_path, aspect, temperature, top_p, hf_token) -> Critique:
    # fall back to the remote model if local model ran out of GPU quota or failed
    try:
        return local_critique(image_path, aspect,temperature, top_p)
    except Exception:
        token = resolve_token(hf_token)
        if not token:
            raise
        logger.warning("Local model failed, falling back to hosted", exc_info=True)
        gr.Warning(
            "The local model is unavailable (GPU quota may be exhausted). "
            "Falling back to the hosted model."
        )
        result = remote_critique(image_path, aspect, temperature, top_p, token)
        result.route = "Failover (local unavailable)"
        return result
