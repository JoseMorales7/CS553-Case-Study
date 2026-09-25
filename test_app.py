import pytest
from local_model import aestoken2score
from remote_model import parse_response
from images import preview_upload


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


@pytest.mark.parametrize(
    "token,expected",
    [("aa", 0), ("az", 25), ("ca", 26), ("cy", 50), ("da", 51), ("dc", 53), ("ey", 100)],
)
def test_aestoken2score_decodes_artimuse_codes(token, expected):
    # ArtiMuse answers with a two-letter code standing for a 0-100 score
    assert aestoken2score(token) == expected


@pytest.mark.parametrize("bad", ["zz", "x", "", "  ", "1/10"])
def test_aestoken2score_rejects_garbage(bad):
    # An unparseable answer must return None so the caller can fail over
    assert aestoken2score(bad) is None


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
