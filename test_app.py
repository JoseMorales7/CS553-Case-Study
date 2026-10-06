import os
import subprocess
import sys
from unittest.mock import Mock

import pytest
import torch
from PIL import Image

import local_model
from local_model import parse_local_score
from remote_model import parse_response
from images import preview_upload
from hf_auth import resolve_token


def test_preview_upload_none():
    # Test the preview logic when no file path is provided
    preview, path, status = preview_upload(None)
    assert preview is None
    assert path is None
    assert status == "Upload an image to begin."


def test_preview_upload_invalid_file():
    # Test the preview logic when an invalid file is provided
    preview, path, status = preview_upload("non_existent_file.jpg")
    assert preview is None
    assert path is None
    assert "Could not read that image" in status


def test_resolve_token_uses_visitor_token():
    assert resolve_token("  hf_visitor_token  ") == "hf_visitor_token"


@pytest.mark.parametrize("token", [None, "", "   "])
def test_resolve_token_rejects_empty_values(token):
    assert resolve_token(token) is None


@pytest.mark.parametrize(
    "text,expected",
    [("0", 0), ("100", 100), (" 72\n", 72), ("Score: 65", 65),
     ("## Score: 58 / 100", 58), ("80/100.", 80)],
)
def test_parse_local_score(text, expected):
    assert parse_local_score(text) == expected


@pytest.mark.parametrize("bad", ["aa", "", "  ", "8/10", "101", "-1", "80.5",
                                "Rate from 0 to 100", "75 or 80"])
def test_parse_local_score_rejects_invalid_answers(bad):
    assert parse_local_score(bad) is None


def test_parse_response_extracts_score_and_body():
    text = "## Score: 58 / 100\n\n### How to improve\n- Crop the foreground."
    score, evaluation = parse_response(text)
    assert score == 58
    assert evaluation == "- Crop the foreground."


def test_parse_response_discards_duplicated_answer():
    # Qwen3 sometimes emits the whole answer twice; keep only the first
    block = "## Score: 58 / 100\n\n### How to improve\n- Crop the foreground.\n"
    score, evaluation = parse_response(block + "\n" + block)
    assert score == 58
    assert evaluation == "- Crop the foreground."


def test_parse_response_without_score_keeps_text():
    score, evaluation = parse_response("The model rambled without a score.")
    assert score is None
    assert evaluation == "The model rambled without a score."


@pytest.fixture
def artwork(tmp_path):
    path = tmp_path / "artwork.png"
    Image.new("RGBA", (64, 48), (60, 100, 160, 255)).save(path)
    return str(path)


@pytest.mark.parametrize("temperature", [0.0, 0.7])
def test_generate_passes_image_and_decodes_only_answer(temperature):
    processor = Mock()
    processor.apply_chat_template.return_value = "image and question prompt"
    processor.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.int64),
        "attention_mask": torch.ones((1, 3), dtype=torch.int64),
        "pixel_values": torch.ones((1, 1, 3, 2, 2), dtype=torch.float32),
        "pixel_attention_mask": torch.ones((1, 1, 2, 2), dtype=torch.bool),
    }
    processor.batch_decode.return_value = [" 72 "]
    model = Mock(device=torch.device("cpu"), dtype=torch.float16)
    model.generate.return_value = torch.tensor([[1, 2, 3, 7, 8]])
    image = Image.new("RGB", (2, 2))

    answer = local_model._generate(model, processor, image, "Rate this image", 16,
                                   temperature, 0.8)

    assert answer == "72"
    messages = processor.apply_chat_template.call_args.args[0]
    assert messages[0]["content"][0] == {"type": "image"}
    assert processor.call_args.kwargs["images"] == [image]
    kwargs = model.generate.call_args.kwargs
    assert kwargs["input_ids"].dtype == torch.int64
    assert kwargs["pixel_values"].dtype == torch.float16
    assert kwargs["pixel_attention_mask"].dtype == torch.bool
    assert kwargs["do_sample"] == (temperature > 0)
    if temperature > 0:
        assert kwargs["temperature"] == temperature
        assert kwargs["top_p"] == 0.8
    else:
        assert "temperature" not in kwargs and "top_p" not in kwargs
    assert processor.batch_decode.call_args.args[0].tolist() == [[7, 8]]


def test_local_critique_uses_separate_score_and_advice_calls(monkeypatch, artwork):
    monkeypatch.setattr(local_model, "_ensure_loaded", lambda: ("model", "processor"))
    generate = Mock(side_effect=["72", "- Crop the edges.\n- Brighten shadows.\n- Reduce clutter."])
    monkeypatch.setattr(local_model, "_generate", generate)
    result = local_model.local_critique(artwork, "Composition & Design", 0.7, 0.8)
    assert result.score == 72
    assert result.model_name == "HuggingFaceTB/SmolVLM-256M-Instruct"
    assert result.route == "Local"
    assert "Crop the edges" in result.to_markdown()
    score_call, advice_call = generate.call_args_list
    assert score_call.args[2].mode == "RGB"
    assert len(score_call.args) == 5  # Score always uses greedy decoding.
    assert "Composition & Design" in advice_call.args[3]
    assert advice_call.args[-2:] == (0.7, 0.8)


@pytest.mark.parametrize("answers,error", [
    (["101"], "invalid score"), (["72", ""], "no improvement suggestions"),
])
def test_local_critique_rejects_invalid_output(monkeypatch, artwork, answers, error):
    monkeypatch.setattr(local_model, "_ensure_loaded", lambda: ("model", "processor"))
    monkeypatch.setattr(local_model, "_generate", Mock(side_effect=answers))
    with pytest.raises(RuntimeError, match=error):
        local_model.local_critique(artwork, "", 0.0, 0.9)


def test_loader_retries_after_failure_and_caches_complete_pair(monkeypatch):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    monkeypatch.setattr(local_model, "_model", None)
    monkeypatch.setattr(local_model, "_processor", None)
    monkeypatch.setattr(local_model, "_device_and_dtype", lambda: ("cpu", torch.float32))
    processor = Mock()
    load_processor = Mock(return_value=processor)
    model = Mock()
    model.eval.return_value = model
    model.to.return_value = model
    load_model = Mock(side_effect=[RuntimeError("download failed"), model])
    monkeypatch.setattr(AutoProcessor, "from_pretrained", load_processor)
    monkeypatch.setattr(AutoModelForImageTextToText, "from_pretrained", load_model)
    with pytest.raises(RuntimeError, match="download failed"):
        local_model._ensure_loaded()
    assert local_model._model is None and local_model._processor is None
    assert local_model._ensure_loaded() == (model, processor)
    assert local_model._ensure_loaded() == (model, processor)
    assert load_model.call_count == 2
    assert load_model.call_args.args == ("HuggingFaceTB/SmolVLM-256M-Instruct",)


def test_invalid_local_score_falls_back_to_hosted(monkeypatch, artwork):
    import router
    from critique import Critique

    monkeypatch.setattr(local_model, "_ensure_loaded", lambda: ("model", "processor"))
    monkeypatch.setattr(local_model, "_generate", Mock(return_value="not a score"))
    remote = Mock(return_value=Critique(80, "- Crop the edge.", "hosted-model", "Hosted"))
    monkeypatch.setattr(router, "remote_critique", remote)
    monkeypatch.setattr(router.gr, "Warning", Mock())
    markdown, status = router.score_artwork(artwork, "", 0, 0.9, True, "hf_visitor")
    assert "80 / 100" in markdown
    assert "Failover (local unavailable)" in status
    assert remote.call_args.args[-1] == "hf_visitor"


@pytest.mark.parametrize("token", [None, "", "   "])
def test_missing_token_automatically_uses_local_and_notifies(monkeypatch, artwork, token):
    import router
    from critique import Critique

    local = Mock(return_value=Critique(72, "- Crop the edges.", local_model.LOCAL_MODEL, "Local"))
    remote = Mock()
    warning = Mock()
    monkeypatch.setattr(router, "local_critique", local)
    monkeypatch.setattr(router, "remote_critique", remote)
    monkeypatch.setattr(router.gr, "Warning", warning)
    markdown, status = router.score_artwork(artwork, "Composition & Design", 0, 0.9, False, token)
    local.assert_called_once_with(artwork, "Composition & Design", 0, 0.9)
    remote.assert_not_called()
    assert "72 / 100" in markdown
    assert "Local (no Hugging Face token)" in status
    warning.assert_called_once_with(
        "No Hugging Face token was provided. Automatically switching to the local model."
    )

# Method generated by GPT5.6-Luna
def test_overloaded_router_rejects_without_calling_models(monkeypatch, artwork):
    import router

    local = Mock()
    remote = Mock()
    monkeypatch.setattr(router, "local_critique", local)
    monkeypatch.setattr(router, "remote_critique", remote)
    monkeypatch.setattr(router, "_capacity_check", lambda: False)

    with pytest.raises(router.gr.Error, match="currently near capacity"):
        router.score_artwork(artwork, "", 0, 0.9, False, "hf_visitor")

    local.assert_not_called()
    remote.assert_not_called()


@pytest.mark.parametrize("use_local", [False, True])
def test_token_and_checkbox_select_requested_model(monkeypatch, artwork, use_local):
    import router
    from critique import Critique

    local = Mock(return_value=Critique(72, "- Crop.", "local", "Local"))
    remote = Mock(return_value=Critique(80, "- Brighten.", "hosted", "Hosted"))
    warning = Mock()
    monkeypatch.setattr(router, "local_critique", local)
    monkeypatch.setattr(router, "remote_critique", remote)
    monkeypatch.setattr(router.gr, "Warning", warning)
    _, status = router.score_artwork(artwork, "", 0, 0.9, use_local, " hf_visitor ")
    if use_local:
        local.assert_called_once()
        remote.assert_not_called()
        assert "Local" in status
    else:
        remote.assert_called_once_with(artwork, "", 0, 0.9, "hf_visitor")
        local.assert_not_called()
        assert "Hosted" in status
    warning.assert_not_called()


def test_hosted_failure_falls_back_to_smolvlm(monkeypatch, artwork):
    import router

    monkeypatch.setattr(local_model, "_ensure_loaded", lambda: ("model", "processor"))
    monkeypatch.setattr(local_model, "_generate", Mock(side_effect=["72", "- Crop the edges."]))
    monkeypatch.setattr(router, "remote_critique", Mock(side_effect=RuntimeError("unavailable")))
    monkeypatch.setattr(router.gr, "Warning", Mock())
    markdown, status = router.score_artwork(artwork, "", 0, 0.9, False, "hf_visitor")
    assert "72 / 100" in markdown
    assert "HuggingFaceTB/SmolVLM-256M-Instruct" in status
    assert "Failover (hosted unavailable)" in status


@pytest.mark.parametrize("cuda,bf16,mps,device,dtype", [
    (True, True, False, "cuda", torch.bfloat16),
    (True, False, False, "cuda", torch.float16),
    (False, False, True, "mps", torch.float32),
    (False, False, False, "cpu", torch.float32),
])
def test_device_selection_uses_supported_precision(monkeypatch, cuda, bf16, mps, device, dtype):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: bf16)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert local_model._device_and_dtype() == (device, dtype)


def test_local_module_import_does_not_load_inference_dependencies():
    # Check a fresh interpreter so earlier tests cannot hide an eager import.
    env = dict(os.environ, PRELOAD_LOCAL_MODEL="1")
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import local_model; "
         "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules; "
         "assert local_model._model is None"],
        env=env, capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr


def test_background_preload_does_not_block_startup(monkeypatch):
    from threading import Event

    entered = Event()
    release = Event()

    def slow_load():
        entered.set()
        assert release.wait(timeout=5)

    monkeypatch.setattr(local_model, "_ensure_loaded", slow_load)
    thread = local_model.preload_in_background()
    try:
        assert entered.wait(timeout=2)
        assert thread.is_alive() and thread.daemon
    finally:
        release.set()
        thread.join(timeout=2)
    assert not thread.is_alive()


@pytest.mark.skipif(os.getenv("SMOLVLM_SMOKE_TEST") != "1",
                    reason="Opt-in test downloads the real model and runs inference")
def test_real_smolvlm_inference(artwork):
    result = local_model.local_critique(artwork, "Composition & Design", 0, 0.9)
    assert result.score is not None
    assert 0 <= result.score <= 100
    assert result.evaluation.strip()

# Method generated by GPT5.6-Luna
def test_recovered_router_accepts_requests(monkeypatch, artwork):
    import router
    from critique import Critique

    remote = Mock(return_value=Critique(80, "- Brighten.", "hosted", "Hosted"))
    monkeypatch.setattr(router, "remote_critique", remote)
    monkeypatch.setattr(router, "_capacity_check", lambda: True)

    markdown, status = router.score_artwork(
        artwork, "", 0, 0.9, False, "hf_visitor"
    )

    remote.assert_called_once()
    assert "80 / 100" in markdown
    assert "Hosted" in status
