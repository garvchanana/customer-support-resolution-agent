
from functools import lru_cache
from pathlib import Path
import re

from sentence_transformers import SentenceTransformer
from sentence_transformers import util


KNOWLEDGE_BASE_DIR = (
    Path(__file__).resolve().parents[2] / "knowledge_base"
)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Conservative starting threshold, to be tuned using evaluation data.
MIN_SIMILARITY = 0.30


def _source_id(path: Path) -> str:
    if path.stem == "refund_policy":
        return "REFUND-DEMO-001"

    return f"POLICY-{path.stem.upper()}"


def _load_sections() -> list[dict[str, str]]:
    """Load Markdown sections with their source metadata."""
    sections = []

    for path in sorted(KNOWLEDGE_BASE_DIR.glob("*.md")):
        document = path.read_text(encoding="utf-8")
        parts = re.split(
            r"(?=^## )",
            document,
            flags=re.MULTILINE,
        )

        for part in parts:
            part = part.strip()

            if not part:
                continue

            sections.append({
                "source": _source_id(path),
                "section": part.splitlines()[0].lstrip("# ").strip(),
                "text": part,
            })

    return sections


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Load the embedding model once per process."""
    return SentenceTransformer(MODEL_NAME)


def retrieve_policy_semantic(
    query: str,
    top_k: int = 3,
) -> list[dict[str, str]]:
    """Retrieve policy sections by embedding cosine similarity."""
    if not query.strip() or top_k <= 0:
        return []

    sections = _load_sections()

    if not sections:
        return []

    model = _get_model()

    query_embedding = model.encode(
        [query],
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    document_embeddings = model.encode(
        [section["text"] for section in sections],
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    scores = util.cos_sim(
        query_embedding,
        document_embeddings,
    )[0]

    ranked_indices = scores.argsort(descending=True).tolist()

    ranked_results = []

    for index in ranked_indices:
        score = float(scores[index].item())

        if score < MIN_SIMILARITY:
            continue

        result = sections[index].copy()
        result["score"] = f"{score:.4f}"
        ranked_results.append(result)

    # Prefer source coverage before adding extra sections
    # from a source already represented.
    diverse_results = []
    seen_sources = set()

    for result in ranked_results:
        if result["source"] not in seen_sources:
            diverse_results.append(result)
            seen_sources.add(result["source"])

    for result in ranked_results:
        if result not in diverse_results:
            diverse_results.append(result)

    return diverse_results[:top_k]
