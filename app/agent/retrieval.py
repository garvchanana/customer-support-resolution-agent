
from pathlib import Path
import re


KNOWLEDGE_BASE_DIR = (
    Path(__file__).resolve().parents[2] / "knowledge_base"
)

STOP_WORDS = {
    "the", "a", "an", "is", "my", "i", "for", "to",
    "and", "of", "please", "want", "would", "like",
    "this", "it", "can", "you", "me", "do", "does",
    "did", "have", "has", "had", "am", "are", "was",
    "were", "be", "been", "being", "with", "on", "in",
    "at", "by", "from", "that", "your", "we", "our",
    "need", "about", "what", "how",
}


def _tokenize(text: str) -> set[str]:
    """Return normalized, non-generic words."""
    return (
        set(re.findall(r"\b[a-z0-9]+\b", text.lower()))
        - STOP_WORDS
    )


def _source_id(path: Path) -> str:
    """Keep the existing refund-policy source ID stable."""
    if path.stem == "refund_policy":
        return "REFUND-DEMO-001"

    return f"POLICY-{path.stem.upper()}"


def retrieve_policy(
    query: str,
    top_k: int = 3,
) -> list[dict[str, str]]:
    """Retrieve relevant sections from Markdown policy documents."""
    if top_k <= 0 or not query.strip():
        return []

    if not KNOWLEDGE_BASE_DIR.exists():
        return []

    query_words = _tokenize(query)

    if not query_words:
        return []

    results = []

    policy_paths = sorted(KNOWLEDGE_BASE_DIR.glob("*.md"))

    query_lower = query.lower()

    refund_terms = (
        "refund", "payment", "reimbursement", "money back"
    )
    damage_terms = (
        "damaged", "damage", "broken", "defective", "cracked"
    )

    has_refund_intent = any(
        term in query_lower for term in refund_terms
    )
    has_damage_intent = any(
        term in query_lower for term in damage_terms
    )

    # Restrict to refund policy only for refund-only queries.
    # Mixed damage/refund queries must search all policies.
    if has_refund_intent and not has_damage_intent:
        policy_paths = [
            path for path in policy_paths
            if path.stem == "refund_policy"
        ]

    for path in policy_paths:
        document = path.read_text(encoding="utf-8")
        sections = re.split(
            r"(?=^## )",
            document,
            flags=re.MULTILINE,
        )

        for section in sections:
            section = section.strip()

            if not section:
                continue

            section_words = _tokenize(section)
            score = len(query_words & section_words)

            # Require stronger overlap for longer queries.
            # Short, focused queries can still retrieve evidence.
            min_score = min(2, max(1, len(query_words) // 3))

            if score < min_score:
                continue

            heading = section.splitlines()[0].lstrip("# ").strip()

            results.append({
                "source": _source_id(path),
                "section": heading,
                "text": section,
                "score": str(score),
            })

    results.sort(
    key=lambda item: int(item["score"]),
    reverse=True,
    )

    # Include the strongest match from each source before
    # returning additional sections from already-seen sources.
    diverse_results = []
    seen_sources = set()

    for result in results:
        if result["source"] not in seen_sources:
            diverse_results.append(result)
            seen_sources.add(result["source"])

    for result in results:
        if result not in diverse_results:
            diverse_results.append(result)

    return diverse_results[:top_k]
