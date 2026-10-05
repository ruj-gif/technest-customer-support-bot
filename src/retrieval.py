"""Load TechNest knowledge files and find relevant text with local TF-IDF."""

import csv
import re
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Keep each Markdown passage small enough to retrieve independently while
# repeating its heading so that a chunk still makes sense on its own.
DEFAULT_CHUNK_SIZE = 700
DEFAULT_OVERLAP = 100
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _split_markdown(text: str, filename: str, chunk_size: int) -> list[dict[str, str]]:
    """Split Markdown into heading-aware chunks, retaining source metadata."""
    headings: list[str] = []
    sections: list[tuple[str, list[str]]] = []
    current_lines: list[str] = []

    # A heading starts a new section; its text is included in that section's
    # chunks to preserve context after splitting.
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            if current_lines:
                sections.append((" > ".join(headings), current_lines))
                current_lines = []
            level = len(line) - len(line.lstrip("#"))
            title = line.lstrip("#").strip()
            headings = headings[: max(level - 1, 0)] + [title]
            current_lines.append(line)
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((" > ".join(headings), current_lines))

    chunks: list[dict[str, str]] = []
    for heading, lines in sections:
        # The heading is added in a normalized form below, so avoid storing
        # the original heading line twice in each chunk.
        content_lines = lines[1:] if lines and lines[0].lstrip().startswith("#") else lines
        section_text = "\n".join(content_lines).strip()
        if not section_text and lines:
            # Keep a document made only of a heading searchable.
            section_text = lines[0].strip()
        if not section_text:
            continue

        # Split long sections by words, adding the heading to every passage.
        prefix = f"{heading}\n" if heading else ""
        words = section_text.split()
        current_words: list[str] = []
        for word in words:
            candidate = " ".join(current_words + [word])
            if current_words and len(prefix) + len(candidate) > chunk_size:
                body = " ".join(current_words)
                chunks.append({"text": prefix + body, "source": filename})
                # Retain a little context at the chunk boundary.
                overlap_words = max(1, DEFAULT_OVERLAP // 6)
                current_words = current_words[-overlap_words:]
            current_words.append(word)

        if current_words:
            chunks.append(
                {"text": prefix + " ".join(current_words), "source": filename}
            )

    return chunks


def load_documents(
    data_dir: str | Path | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> list[dict[str, str]]:
    """Read the FAQ, return policy, and product catalogue into text records.

    Missing files are skipped so a partial knowledge base can still be used.
    Each returned record contains searchable ``text`` and its ``source`` name.
    """
    folder = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    documents: list[dict[str, str]] = []

    for filename in ("faq.md", "return_policy.md"):
        path = folder / filename
        try:
            markdown = path.read_text(encoding="utf-8")
        except (FileNotFoundError, IsADirectoryError, PermissionError):
            continue
        documents.extend(_split_markdown(markdown, filename, max(1, chunk_size)))

    products_path = folder / "products.csv"
    try:
        with products_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                # Convert each catalogue row into readable text while keeping
                # every available field searchable (including the product ID).
                product_text = "; ".join(
                    f"{key.replace('_', ' ').title()}: {value.strip()}"
                    for key, value in row.items()
                    if key and value and value.strip()
                )
                if product_text:
                    # Keep the individual CSV fields as metadata as well as
                    # including them in searchable text. This makes it easy
                    # to inspect a product result without parsing the text.
                    product_document = {
                        "text": product_text,
                        "source": products_path.name,
                    }
                    product_document.update(
                        {
                            key: value.strip()
                            for key, value in row.items()
                            if key and value
                        }
                    )
                    documents.append(product_document)
    except (FileNotFoundError, IsADirectoryError, PermissionError):
        pass

    return documents


def retrieve(
    query: str,
    documents: list[dict[str, Any]] | None = None,
    top_k: int = 3,
    data_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Return the most similar knowledge chunks for a query.

    Results include ``text``, ``source``, and a cosine ``score`` between 0 and
    1. An empty query, empty knowledge base, or query with no shared terms
    returns an empty list instead of raising an error.
    """
    if not query or not query.strip() or top_k <= 0:
        return []

    knowledge = documents if documents is not None else load_documents(data_dir)
    usable = [item for item in knowledge if item.get("text", "").strip()]
    if not usable:
        return []

    product_documents = [item for item in usable if item.get("source") == "products.csv"]

    # Broad inventory questions need a catalog answer, not whichever general
    # document happens to share words like "products" or "sell".
    if product_documents and _is_product_listing_query(query):
        return [dict(item) for item in product_documents]

    # For a specific product question, rank only product rows. Ignore obvious
    # FAQ and policy topics so a question such as "warranty on laptops" keeps
    # using the established FAQ retrieval behavior.
    if product_documents and _is_specific_product_query(query, product_documents):
        usable = product_documents
        vectorizer = TfidfVectorizer(stop_words="english")
    else:
        # Preserve the original all-document TF-IDF path for non-product
        # questions, including its existing FAQ and return-policy behavior.
        vectorizer = TfidfVectorizer()

    try:
        document_vectors = vectorizer.fit_transform(
            [item["text"] for item in usable]
        )
        query_vector = vectorizer.transform([query.strip()])
    except ValueError:
        # For example, every document may contain only punctuation.
        return []

    scores = cosine_similarity(query_vector, document_vectors).ravel()
    ranked_indices = scores.argsort()[::-1]

    results: list[dict[str, Any]] = []
    for index in ranked_indices:
        score = float(scores[index])
        if score <= 0:
            continue
        result = dict(usable[int(index)])
        result["score"] = score
        results.append(result)
        if len(results) >= top_k:
            break

    return results


def _is_product_listing_query(query: str) -> bool:
    """Recognize questions asking for the catalog or inventory as a whole."""
    normalized = query.lower()
    mentions_catalog = any(
        term in normalized
        for term in ("product", "catalog", "catalogue", "inventory", "items")
    )
    asks_to_list = any(
        term in normalized
        for term in ("sell", "offer", "carry", "list", "available", "have")
    )
    asks_what_sell = bool(re.search(r"\bwhat\s+do\s+you\s+sell\b", normalized))
    return (mentions_catalog and asks_to_list) or asks_what_sell


def _is_specific_product_query(
    query: str, product_documents: list[dict[str, Any]]
) -> bool:
    """Return true when a query names a catalog item or one of its features."""
    # Policy and store questions should continue to use the FAQ/policy chunks,
    # even when they mention a product category such as laptops.
    faq_topics = {
        "warranty", "return", "refund", "delivery", "shipping", "store",
        "hours", "payment", "tracking", "damaged", "defective",
    }
    query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
    if query_terms & faq_topics:
        return False

    # Match searchable product name, category, and description terms. Also
    # normalize a simple plural (headphones -> headphone) for natural queries.
    def normalize_terms(value: str) -> set[str]:
        terms = set(re.findall(r"[a-z0-9]+", value.lower()))
        return {term[:-1] if term.endswith("s") and len(term) > 4 else term for term in terms}

    normalized_query = normalize_terms(query)
    meaningful_query_terms = normalized_query - {
        "what", "which", "do", "does", "you", "your", "have", "sell",
        "the", "a", "an", "is", "are", "of", "for", "me", "please",
        "price", "cost", "much", "how", "in", "stock",
    }
    if not meaningful_query_terms:
        return False

    searchable_fields = ("product_name", "category", "description")
    for product in product_documents:
        product_terms: set[str] = set()
        for field in searchable_fields:
            product_terms.update(normalize_terms(product.get(field, "")))
        if meaningful_query_terms & product_terms:
            return True
    return False
