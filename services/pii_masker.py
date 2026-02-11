"""PII / Sensitive-ID masking service for RAG responses.

Applies regex-based redaction to text before it reaches the user.
Designed as a post-generation filter — runs on the streamed LLM output.

Covers common patterns found in founder documents (finance sheets,
loan policies, HR handbooks, etc.).
"""

from __future__ import annotations

import re
from typing import List, Tuple

from loguru import logger

# Each rule is (pattern_name, compiled_regex, replacement)
_RULES: List[Tuple[str, re.Pattern, str]] = [
    # Social Security Numbers: 123-45-6789 or 123 45 6789
    (
        "SSN",
        re.compile(r"\b\d{3}[-\s]\d{2}[-\s]\d{4}\b"),
        "[SSN REDACTED]",
    ),
    # Credit / Debit card numbers: 16 digits, optional dashes/spaces
    (
        "CARD",
        re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
        "[CARD REDACTED]",
    ),
    # IBAN (international bank account)
    (
        "IBAN",
        re.compile(r"\b[A-Z]{2}\d{2}[\s]?[\dA-Z]{4}[\s]?(?:[\dA-Z]{4}[\s]?){2,7}[\dA-Z]{1,4}\b"),
        "[IBAN REDACTED]",
    ),
    # Generic bank account numbers: 8-17 digits (only when preceded by contextual keyword)
    (
        "ACCOUNT",
        re.compile(r"(?i)(?:account|acct|a/c)[\s.#:_-]*(\d[\d\s-]{6,16}\d)"),
        "[ACCOUNT REDACTED]",
    ),
    # Email addresses
    (
        "EMAIL",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
        "[EMAIL REDACTED]",
    ),
    # Phone numbers: international and local formats
    (
        "PHONE",
        re.compile(r"(?:\+\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b"),
        "[PHONE REDACTED]",
    ),
    # Passport numbers: 1-2 letters + 6-9 digits (common formats)
    (
        "PASSPORT",
        re.compile(r"(?i)(?:passport)[\s.#:_-]*([A-Z]{1,2}\d{6,9})\b"),
        "[PASSPORT REDACTED]",
    ),
    # National ID / Tax ID patterns with keyword context
    (
        "ID_NUMBER",
        re.compile(r"(?i)(?:national\s*id|tax\s*id|tin|ein|ssn|id\s*number|id\s*no)[\s.#:_-]*(\d[\d\s-]{4,12}\d)"),
        "[ID REDACTED]",
    ),
]


def mask_pii(text: str) -> str:
    """Apply all PII masking rules to *text* and return the redacted version.

    This is intentionally lightweight (regex-only) for low latency in
    streaming contexts.  For production, consider adding NER-based
    detection (e.g. presidio, spaCy) for names and addresses.
    """
    masked = text
    redaction_count = 0

    for name, pattern, replacement in _RULES:
        new_text, n = pattern.subn(replacement, masked)
        if n > 0:
            redaction_count += n
            masked = new_text

    if redaction_count > 0:
        logger.info("PII masker: redacted {} item(s)", redaction_count)

    return masked
