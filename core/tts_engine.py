"""gTTS vernacular audio renderer."""

from __future__ import annotations

from pathlib import Path

from core.translator import language_code

ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"
AUDIO_DIR = ASSETS_DIR / "audio"


def synthesize_voice(text: str, language_name: str, stem: str = "rxverify_voice") -> Path | None:
    """Render translated text to MP3. Returns the file path, or None if TTS fails."""
    if not text.strip():
        return None
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    code = language_code(language_name)
    out_path = AUDIO_DIR / f"{stem}_{code}.mp3"
    try:
        from gtts import gTTS

        tts = gTTS(text=text, lang=code)
        tts.save(str(out_path))
        return out_path
    except Exception:  # noqa: BLE001 — network / codec issues should not crash the UI
        return None
