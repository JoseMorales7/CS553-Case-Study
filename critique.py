from dataclasses import dataclass
from config import EVALUATION_HEADING

# The shared result type for both models
@dataclass
class Critique:
    score: int | None
    evaluation: str
    model_name: str
    route: str  # "Hosted", "Local", or a "Failover (...)" description

    def to_markdown(self) -> str:
        heading = (
            f"## Score: {self.score} / 100"
            if self.score is not None
            else "## Score: unavailable"
        )
        return f"{heading}\n\n### {EVALUATION_HEADING}\n{self.evaluation.strip()}"

    def to_status(self) -> str:
        return f"Critique complete · {self.route} · `{self.model_name}`"
