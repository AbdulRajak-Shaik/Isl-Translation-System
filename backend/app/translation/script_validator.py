"""
Telugu Script Validator and Unicode Normalizer.
Validates that output text is rendered in native Telugu script (U+0C00 to U+0C7F)
rather than Latin transliteration, and cleans diacritics.
"""

import re
import unicodedata
from typing import Tuple


class TeluguScriptValidator:
    """Validates and normalizes native Telugu Unicode strings."""

    # Unicode Range for Telugu: U+0C00 - U+0C7F
    TELUGU_UNICODE_REGEX = re.compile(r"[\u0C00-\u0C7F]")

    @classmethod
    def contains_telugu_script(cls, text: str) -> bool:
        """Returns True if the text contains native Telugu script characters."""
        if not text:
            return False
        return bool(cls.TELUGU_UNICODE_REGEX.search(text))

    @classmethod
    def telugu_character_ratio(cls, text: str) -> float:
        """
        Calculates the proportion of non-whitespace characters that are native Telugu.
        Used to detect mixed or Latin-transliterated text.
        """
        clean_text = "".join(text.split())
        if not clean_text:
            return 0.0

        telugu_chars = sum(1 for ch in clean_text if '\u0C00' <= ch <= '\u0C7F')
        return telugu_chars / len(clean_text)

    @classmethod
    def normalize_telugu_text(cls, text: str) -> str:
        """
        Performs Unicode NFC normalization and strips non-printable control codes.
        """
        if not text:
            return ""
        # NFC standard normalization for Indic scripts
        normalized = unicodedata.normalize("NFC", text.strip())
        # Clean double spaces
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized

    @classmethod
    def validate_translation(cls, text: str, threshold: float = 0.5) -> Tuple[bool, str]:
        """
        Validates translation output.
        Returns: (is_valid, message)
        """
        if not text or not text.strip():
            return False, "Translation text is empty."

        ratio = cls.telugu_character_ratio(text)
        if ratio < threshold:
            return False, f"Output text has low Telugu script ratio ({ratio:.1%}). May contain Roman transliteration."

        return True, "Valid native Telugu script."
