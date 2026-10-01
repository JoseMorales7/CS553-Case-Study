def resolve_token(hf_token: str | None) -> str | None:
    """Normalize a token supplied by the current visitor."""
    token = hf_token.strip() if hf_token else ""
    return token or None
