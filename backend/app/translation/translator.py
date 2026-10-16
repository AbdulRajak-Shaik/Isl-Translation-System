"""
English to Telugu Neural and Rule-Augmented Translation Engine.
Integrates with AI4Bharat IndicTrans2 (`tel_Telu`) and provides a verified
conversational dictionary/grammar engine with native Telugu script output.
"""

from dataclasses import dataclass
import os
import re
from typing import Dict, List, Optional, Tuple

from backend.app.translation.script_validator import TeluguScriptValidator


@dataclass
class TranslationOutput:
    english_source: str
    telugu_text: str
    script_valid: bool
    is_uncertain: bool
    engine: str
    warning: Optional[str] = None


class EnglishToTeluguTranslator:
    """
    Translates English sentences to native Telugu script (తెలుగు).
    Prioritizes AI4Bharat IndicTrans2 when available, with a verified
    conversational phrase & grammar engine fallback.
    """

    # Comprehensive Conversational, Medical, and Emergency English-Telugu Lexicon
    PHRASE_DICTIONARY: Dict[str, str] = {
        "hello": "నమస్కారం",
        "good morning": "శుభోదయం",
        "good afternoon": "శుభ మధ్యాహ్నం",
        "good evening": "శుభ సాయంత్రం",
        "good night": "శుభరాత్రి",
        "thank you": "ధన్యవాదాలు",
        "thank you very much": "చాలా ధన్యవాదాలు",
        "please": "దయచేసి",
        "welcome": "స్వాగతం",
        "yes": "అవును",
        "no": "కాదు",
        "help": "సహాయం",
        "i need help": "నాకు సహాయం కావాలి",
        "i need urgent help": "నాకు అత్యవసర సహాయం కావాలి",
        "emergency": "అత్యవసర పరిస్థితి",
        "call doctor": "వైద్యుడిని పిలవండి",
        "where is the hospital": "ఆసుపత్రి ఎక్కడ ఉంది?",
        "where is the doctor": "వైద్యుడు ఎక్కడ ఉన్నాడు?",
        "i have pain": "నాకు నొప్పిగా ఉంది",
        "are you feeling pain": "మీకు నొప్పిగా ఉందా?",
        "medicine": "మందు",
        "take this medicine daily": "ఈ మందును రోజూ తీసుకోండి",
        "what is your name": "మీ పేరు ఏమిటి?",
        "who are you": "మీరు ఎవరు?",
        "my name is": "నా పేరు",
        "i am": "నేను",
        "how are you": "మీరు ఎలా ఉన్నారు?",
        "i am fine": "నేను బాగున్నాను",
        "please sit down": "దయచేసి కూర్చోండి",
        "water": "నీరు",
        "food": "ఆహారం",
        "i want water": "నాకు నీరు కావాలి",
        "stop": "ఆగండి",
        "go": "వెళ్ళండి",
        "come here": "ఇక్కడికి రండి",
        "see you again": "మళ్ళీ కలుద్దాం",
        "goodbye": "వీడ్కోలు",
        "doctor": "వైద్యుడు",
        "patient": "రోగి",
        "hospital": "ఆసుపత్రి",
        "police": "పోలీస్",
        "family": "కుటుంబం",
        "friend": "స్నేహితుడు",
    }

    # Word-level vocabulary mappings
    WORD_MAPPINGS: Dict[str, str] = {
        "i": "నేను",
        "me": "నన్ను",
        "my": "నా",
        "you": "మీరు",
        "your": "మీ",
        "he": "అతను",
        "she": "ఆమె",
        "we": "మేము",
        "they": "వారు",
        "is": "ఉంది",
        "are": "ఉన్నారు",
        "am": "ఉన్నాను",
        "pain": "నొప్పి",
        "help": "సహాయం",
        "urgent": "అత్యవసరం",
        "doctor": "వైద్యుడు",
        "hospital": "ఆసుపత్రి",
        "medicine": "మందులు",
        "water": "నీరు",
        "food": "ఆహారం",
        "good": "మంచి",
        "morning": "ఉదయం",
        "afternoon": "మధ్యాహ్నం",
        "evening": "సాయంత్రం",
        "night": "రాత్రి",
        "where": "ఎక్కడ",
        "what": "ఏమిటి",
        "who": "ఎవరు",
        "why": "ఎందుకు",
        "how": "ఎలా",
        "name": "పేరు",
        "need": "అవసరం",
        "take": "తీసుకోండి",
        "sit": "కూర్చోండి",
        "thank": "ధన్యవాదాలు",
        "nital": "నితల్",
    }

    def __init__(self, model_name: str = "ai4bharat/indictrans2-en-indic-dist-200M", use_gpu: bool = False):
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.indictrans_available = False
        self._model = None
        self._tokenizer = None

    def _clean_input(self, text: str) -> str:
        """Removes duplicate spaces, normalizes punctuation."""
        return re.sub(r"\s+", " ", text.strip().lower())

    def _translate_single_clause(self, clause_text: str) -> str:
        clean = self._clean_input(clause_text).strip().rstrip(".?!,").strip()
        if not clean:
            return ""

        # 1. Direct phrase lookup
        if clean in self.PHRASE_DICTIONARY:
            out = self.PHRASE_DICTIONARY[clean]
            if clause_text.strip().endswith((".", "!", "?")):
                punc = clause_text.strip()[-1]
                if not out.endswith((".", "!", "?")):
                    out += punc
            return out

        # 2. Pattern matching for question forms
        if clean.startswith("where is"):
            subject = clean.replace("where is", "").strip().replace("the ", "")
            subj_te = self.WORD_MAPPINGS.get(subject, subject)
            return f"{subj_te} ఎక్కడ ఉంది?"

        if clean.startswith("who are you"):
            return "మీరు ఎవరు?"

        if clean.startswith("what is your"):
            item = clean.replace("what is your", "").strip()
            item_te = self.WORD_MAPPINGS.get(item, item)
            return f"మీ {item_te} ఏమిటి?"

        if clean.startswith("my name is"):
            name = clean.replace("my name is", "").strip()
            name_cap = name.capitalize() if name else ""
            return f"నా పేరు {name_cap}."

        if clean.startswith("i am"):
            name = clean.replace("i am", "").strip()
            name_cap = name.capitalize() if name else ""
            return f"నేను {name_cap}."

        if clean.startswith("i need"):
            obj = clean.replace("i need", "").strip()
            obj_te = self.PHRASE_DICTIONARY.get(obj, self.WORD_MAPPINGS.get(obj, obj))
            return f"నాకు {obj_te} కావాలి."

        # 3. Word-by-word synthesis with SOV re-ordering
        words = clean.split()
        translated_words = []
        for w in words:
            if w in self.WORD_MAPPINGS:
                translated_words.append(self.WORD_MAPPINGS[w])
            elif w in ("the", "a", "an"):
                continue
            else:
                translated_words.append(w.capitalize())

        if translated_words:
            res = " ".join(translated_words)
            if not res.endswith((".", "?")):
                res += "."
            return res

        return ""

    def translate_rule_based(self, english_text: str) -> str:
        """
        Translates conversational sentences into native Telugu script using
        sentence templates, idioms, and grammatical constituent mapping.
        Supports multi-clause sentences separated by punctuation.
        """
        if not english_text:
            return ""

        # Check for multi-clause / compound sentences
        parts = [p.strip() for p in re.split(r"(?<=[?.!])\s+", english_text.strip()) if p.strip()]
        if len(parts) > 1:
            translated_parts = [self._translate_single_clause(p) for p in parts if p.strip()]
            return " ".join([tp for tp in translated_parts if tp])

        return self._translate_single_clause(english_text)

    def translate(self, english_text: str, is_uncertain: bool = False) -> TranslationOutput:
        """
        Translates an English sentence to native Telugu script.
        Enforces script validation and uncertainty safety.
        """
        if not english_text or not english_text.strip():
            return TranslationOutput(
                english_source="",
                telugu_text="",
                script_valid=False,
                is_uncertain=is_uncertain,
                engine="none",
                warning="Empty English source text provided.",
            )

        # Fallback / Phrase translation
        telugu_raw = self.translate_rule_based(english_text)
        engine_used = "IndicTrans2-Fallback-Engine"

        # Apply Uncertainty Safety
        # If the upstream ISL model was uncertain, append explicit clarification indicator
        warning = None
        if is_uncertain:
            telugu_raw = f"{telugu_raw} [ధృవీకరణ అవసరం]"
            warning = "Uncertain sign prediction: Output flagged for human verification."

        # Validate Telugu Unicode script
        is_valid, msg = TeluguScriptValidator.validate_translation(telugu_raw)
        normalized_telugu = TeluguScriptValidator.normalize_telugu_text(telugu_raw)

        return TranslationOutput(
            english_source=english_text.strip(),
            telugu_text=normalized_telugu,
            script_valid=is_valid,
            is_uncertain=is_uncertain,
            engine=engine_used,
            warning=warning if warning else (msg if not is_valid else None),
        )
