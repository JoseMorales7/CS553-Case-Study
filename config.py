# Model identifiers, prompts and the shared output
# Both models must produce same markdown shape so the UI treats them identically:
# Score: N / 100
# How to improve

REMOTE_MODEL = "Qwen/Qwen3.8-27B"
LOCAL_MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"

# Select model provider
REMOTE_PROVIDER = "auto"
REMOTE_MAX_TOKENS = 2048
LOCAL_MAX_TOKENS = 512

# The heading advice for both models
EVALUATION_HEADING = "How to improve"

# Declare the aspect of the image to evaluate
ASPECTS = [
    "Overall Gestalt",
    "Composition & Design",
    "Visual Elements & Structure",
    "Technical Execution",
    "Originality & Creativity",
]


LOCAL_SCORE_QUESTION = (
    "Rate the overall aesthetic quality of this artwork from 0 to 100. "
    "Reply with only one whole number between 0 and 100."
)

# Keep the local requests simple for the small instruction model.
def advice_question(aspect: str) -> str:
    focus = f" in terms of {aspect}" if aspect else ""
    return f"""Suggest three specific ways to improve the aesthetic quality of this image{focus}.
Answer as exactly three short bullet points, each naming one concrete change the artist should make. 
Do not describe what the image shows."""



# Ask for advice from Qwen3
def remote_prompt(aspect: str) -> str:
    focus = f" in terms of {aspect}" if aspect else ""
    return f"""Rate this image's aesthetic quality.

Reply in exactly this format, once, and nothing else:

## Score: N / 100

### {EVALUATION_HEADING}
- <one concrete change, at most 15 words>
- <one concrete change, at most 15 words>
- <one concrete change, at most 15 words>

Rules:
- N is a whole number from 0 to 100, judging the image's overall aesthetic quality.
- Do NOT describe what the image shows. The user can already see it.
- Do NOT justify the score or add any prose outside the three bullets.
- Each bullet must name an action the artist can take{focus}.
- Output the format once. Do not repeat it."""
