"""
Unit Tests for Telugu Script Validation, English-to-Telugu Translation, and TTS Audio Generation.
"""

import pytest

from backend.app.translation.script_validator import TeluguScriptValidator
from backend.app.translation.translator import EnglishToTeluguTranslator
from backend.app.tts.engine import TTSEngine


class TestTranslationAndTTS:

    def test_telugu_script_validator(self):
        # Native Telugu string
        telugu_sample = "ఆసుపత్రి ఎక్కడ ఉంది?"
        assert TeluguScriptValidator.contains_telugu_script(telugu_sample) is True
        ratio = TeluguScriptValidator.telugu_character_ratio(telugu_sample)
        assert ratio > 0.70

        is_valid, msg = TeluguScriptValidator.validate_translation(telugu_sample)
        assert is_valid is True

        # Pure Roman transliteration should fail Telugu script check
        roman_sample = "Asupatri ekkada undi?"
        assert TeluguScriptValidator.contains_telugu_script(roman_sample) is False
        is_valid_roman, _ = TeluguScriptValidator.validate_translation(roman_sample)
        assert is_valid_roman is False

    def test_english_to_telugu_translator(self):
        translator = EnglishToTeluguTranslator()
        res = translator.translate("Where is the hospital?")
        assert res.script_valid is True
        assert TeluguScriptValidator.contains_telugu_script(res.telugu_text)
        assert "ఆసుపత్రి" in res.telugu_text

        # Test Uncertainty Flag Annotation
        res_uncertain = translator.translate("Where is the hospital?", is_uncertain=True)
        assert res_uncertain.is_uncertain is True
        assert "[ధృవీకరణ అవసరం]" in res_uncertain.telugu_text

    def test_tts_engine(self):
        tts = TTSEngine()
        # Empty text should fail gracefully without crash
        empty_res = tts.generate_audio_base64("", "en")
        assert empty_res["success"] is False
        assert empty_res["audio_base64"] is None

        # English speech test
        en_res = tts.generate_audio_base64("Good morning, doctor.", "en")
        assert "success" in en_res
        if en_res["success"]:
            assert en_res["audio_base64"].startswith("data:audio/mp3;base64,")
        else:
            assert en_res["error"] is not None

        # Telugu speech test
        te_res = tts.generate_audio_base64("నమస్కారం డాక్టర్.", "te")
        assert "success" in te_res
        if te_res["success"]:
            assert te_res["audio_base64"].startswith("data:audio/mp3;base64,")
        else:
            assert te_res["error"] is not None
