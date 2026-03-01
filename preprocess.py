import re
import html

# --- Contraction Map ---
CONTRACTION_MAP = {
    r"\b(cant)\b": "cannot",
    r"\b(wont)\b": "will not",
    r"\b(dont)\b": "do not",
    r"\b(i'm|im)\b": "i am",
    r"\b(he's|hes)\b": "he is",
}


def expand_contractions(text: str) -> str:
    """Expand common contractions in text."""
    for pattern, expansion in CONTRACTION_MAP.items():
        text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
    return text


def preprocess_text(text: str) -> str:
    """
    Full preprocessing pipeline:
    - Expand contractions
    - Unescape HTML entities
    - Replace URLs and mentions with tokens
    - Normalize whitespace
    - Strip non-ASCII characters
    """
    text = expand_contractions(text)
    text = html.unescape(text)
    text = re.sub(r'http\S+|www\S+', 'URLS', text)
    text = re.sub(r'@\w+', 'MEN', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.encode("ascii", errors="ignore").decode()
    return text
