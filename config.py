# Model identifiers, prompts and the shared output
# Both models must produce same markdown shape so the UI treats them identically:
# Score: N / 100
# How to improve

REMOTE_MODEL = "Qwen/Qwen3.8-27B"
LOCAL_MODEL = "Thunderbolt215215/ArtiMuse"

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


# ArtiMuse's own scoring question
# The model is trained to answer with a two-letter code that maps onto 0-100
ARTIMUSE_SCORE_QUESTION = """
Rate the aesthetics score of the image in 0-100.
In the output format, numbers are replaced by 2 corresponding letters,
and the mapping relationship is:
score 0 to 25: 0-aa, 1-ab, 2-ac, 3-ad, ... , 25-az,
score 26 to 50: 26-ca, 27-cb, 28-cc, 29-cd, ..., 50-cy,
score 51 to 75: 51-da, 52-db, 53-dc, 54-dd, ..., 75-dy,
score 76 to 100: 76-ea, 77-eb, 78-ec, 79-ed, ..., 100-ey.

The answer only outputs 2 corresponding letters.
"""

# Ask for advice from ArtiMuse
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
