import re
from typing import Tuple, Optional

def find_substring_span(text: str, query: str) -> Optional[Tuple[int, int]]:
    """
    Locates the exact start and end character indices of query within text.
    If exact match fails (e.g. slight punctuation/whitespace differences),
    it tries normalized whitespace matching and regex fuzzy matching.
    """
    if not text or not query:
        return None

    clean_query = query.strip()
    if not clean_query:
        return None

    # 1. Exact Substring Match
    idx = text.find(clean_query)
    if idx != -1:
        return (idx, idx + len(clean_query))

    # 2. Case-insensitive exact match
    idx_lower = text.lower().find(clean_query.lower())
    if idx_lower != -1:
        return (idx_lower, idx_lower + len(clean_query))

    # 3. Whitespace-flexible regex match
    escaped_tokens = [re.escape(token) for token in re.split(r'\s+', clean_query) if token]
    if escaped_tokens:
        pattern = r'\s+'.join(escaped_tokens)
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return (match.start(), match.end())

    # 4. Partial prefix match fallback (at least 75% of query)
    min_len = int(len(clean_query) * 0.75)
    if min_len >= 10:
        sub_query = clean_query[:min_len]
        idx = text.lower().find(sub_query.lower())
        if idx != -1:
            return (idx, min(len(text), idx + len(clean_query)))

    return None

def extract_sentences_with_offsets(text: str) -> list[dict]:
    """
    Splits text into sentences while retaining exact character offsets.
    """
    sentences = []
    # Match sentences ending with . ! ? or newlines
    pattern = r'[^.!?\n]+[.!?\n]?'
    for match in re.finditer(pattern, text):
        s_text = match.group().strip()
        if s_text:
            sentences.append({
                "text": s_text,
                "start_char": match.start(),
                "end_char": match.end()
            })
    if not sentences and text.strip():
        sentences.append({
            "text": text.strip(),
            "start_char": 0,
            "end_char": len(text.strip())
        })
    return sentences

def compute_lexical_grounding(span_text: str, criterion_condition: str, target_keywords: Optional[list] = None) -> float:
    """
    Computes S_lexical = |LemmatizedTokens(Span) ∩ Keywords(Criterion)| / |Keywords(Criterion)|
    Filters out common English stopwords and normalizes alphanumeric lemmas.
    """
    if not span_text or (not criterion_condition and not target_keywords):
        return 0.0

    stopwords = {
        "a", "an", "the", "in", "on", "at", "to", "for", "of", "with", "by", "from",
        "and", "or", "but", "is", "are", "was", "were", "be", "been", "being", "that",
        "this", "these", "those", "it", "its", "as", "if", "when", "than", "then"
    }

    def tokenize(text: str) -> set:
        raw_tokens = re.findall(r'[a-zA-Z0-9_\-\u00b2\u2082\u03bc\u03c3]+', text.lower())
        return {t for t in raw_tokens if t not in stopwords and (len(t) > 1 or t.isdigit() or t in {'x', 'y', 'z', 'p', 'μ', 'σ'})}

    span_tokens = tokenize(span_text)

    if target_keywords:
        criterion_keywords = {k.lower() for k in target_keywords if k.lower() not in stopwords}
    else:
        criterion_keywords = tokenize(criterion_condition)

    if not criterion_keywords:
        return 1.0 if span_tokens else 0.0

    matched = span_tokens.intersection(criterion_keywords)
    return round(len(matched) / len(criterion_keywords), 4)
