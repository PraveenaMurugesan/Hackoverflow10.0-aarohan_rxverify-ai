"""Regional-language translation for dosage narratives."""

from __future__ import annotations

LANGUAGE_CODES = {
    "Tamil": "ta",
    "Hindi": "hi",
    "Telugu": "te",
    "Malayalam": "ml",
    "English": "en",
}

LANGUAGE_LABELS = {
    "Tamil": "Tamil (தமிழ்)",
    "Hindi": "Hindi (हिंदी)",
    "Telugu": "Telugu (తెలుగు)",
    "Malayalam": "Malayalam (മലയാളം)",
    "English": "English",
}


def language_code(language_name: str) -> str:
    return LANGUAGE_CODES.get(language_name, "en")


def translate_dosage_text(text: str, language_name: str) -> str:
    """Translate an English dosage string into the selected vernacular."""
    code = language_code(language_name)
    if code == "en" or not text.strip():
        return text
    try:
        from deep_translator import GoogleTranslator

        translated = GoogleTranslator(source="en", target=code).translate(text)
        return translated or text
    except Exception:  # noqa: BLE001 — stay usable offline
        return text
