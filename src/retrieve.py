"""Keyword-based retrieval over PDF pages (level 3)."""
import re
from collections import Counter
import math

import pymupdf

# Common English words that appear on almost every page and carry no meaning for search.
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "be", "to", "of", "in", "on", "for", "and", "or",
    "what", "which", "how", "does", "do", "can", "when", "where", "why", "this", "that",
    "it", "its", "by", "with", "from", "as", "at", "i", "you",
}


def tokenize(text):
    """Lowercase the text and split it into words, dropping stopwords and 1-char tokens."""
    words = re.findall(r"[a-z0-9_]+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 1]


def load_pages(pdf_path):
    """Read every page once and pre-count its words, so each search is fast."""
    doc = pymupdf.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        pages.append({
            "page": i + 1,                      # 1-based, as shown in the PDF viewer
            "text": text,
            "counts": Counter(tokenize(text)),  # word -> how many times it occurs on this page
        })
    return pages


def search(question, pages, k=5):
    """Return the k pages with the highest TF-IDF score for the question's words."""
    query_words = set(tokenize(question))
    n_pages = len(pages)

    # IDF: rare words across the manual weigh more than common ones.
    idf = {}
    for w in query_words:
        df = sum(1 for p in pages if w in p["counts"])
        idf[w] = math.log(n_pages / df) if df else 0.0

    results = []
    for p in pages:
        score = sum(p["counts"][w] * idf[w] for w in query_words)
        if score > 0:
            results.append({"page": p["page"], "score": round(score, 1), "text": p["text"]})
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:k]