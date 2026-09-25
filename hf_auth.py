from huggingface_hub import get_token

# mock oauth token for local development
MOCK_OAUTH_TOKEN = "mock-oauth-token-for-local-dev"


def resolve_token(hf_token) -> str | None:
    """Return the visitor's HF token, or None if they are not signed in."""
    token = getattr(hf_token, "token", None)
    if token == MOCK_OAUTH_TOKEN:
        return get_token()
    return token
