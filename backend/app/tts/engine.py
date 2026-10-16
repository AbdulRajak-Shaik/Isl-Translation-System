"""
Independent English and Telugu Text-to-Speech (TTS) Engine.
Supports sentence-level speech generation, base64 audio streaming,
and graceful fallback handling.
"""

import base64
from io import BytesIO
import os
import re
from typing import Any, Dict, Optional

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False


class TTSEngine:
    """
    Synthesizes natural, intelligible speech for English and Telugu.
    """

    SUPPORTED_LANGUAGES = {"en": "en", "te": "te"}

    def __init__(self, default_lang: str = "en"):
        self.default_lang = default_lang

    @staticmethod
    def _clean_text_for_speech(text: str) -> str:
        """Removes bracketed uncertainty tags, URLs, or markdown symbols before vocalization."""
        # Remove tags like [ధృవీకరణ అవసరం] or [Uncertain]
        clean = re.sub(r"\[.*?\]", "", text)
        clean = re.sub(r"[*_#`]", "", clean)
        return clean.strip()

    def generate_audio_base64(self, text: str, lang: str = "en") -> Dict[str, Any]:
        """
        Synthesizes speech into an in-memory MP3 buffer and encodes as base64 data URI.
        Returns:
            {
                "success": bool,
                "audio_base64": Optional[str],
                "format": "audio/mp3",
                "lang": str,
                "error": Optional[str],
            }
        """
        clean_text = self._clean_text_for_speech(text)
        if not clean_text:
            return {
                "success": False,
                "audio_base64": None,
                "format": "audio/mp3",
                "lang": lang,
                "error": "Cannot synthesize audio for empty or invalid text.",
            }

        target_lang = self.SUPPORTED_LANGUAGES.get(lang.lower(), self.default_lang)

        if not GTTS_AVAILABLE:
            return {
                "success": False,
                "audio_base64": None,
                "format": "audio/mp3",
                "lang": target_lang,
                "error": "gTTS library is not installed in the environment.",
            }

        try:
            # Generate speech into memory buffer
            tts = gTTS(text=clean_text, lang=target_lang, slow=False)
            fp = BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            audio_bytes = fp.read()
            b64_audio = base64.b64encode(audio_bytes).decode("utf-8")

            return {
                "success": True,
                "audio_base64": f"data:audio/mp3;base64,{b64_audio}",
                "format": "audio/mp3",
                "lang": target_lang,
                "error": None,
            }
        except Exception as e:
            return {
                "success": False,
                "audio_base64": None,
                "format": "audio/mp3",
                "lang": target_lang,
                "error": f"TTS synthesis failed: {str(e)}",
            }
