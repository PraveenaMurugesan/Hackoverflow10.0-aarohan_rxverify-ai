"""EasyOCR extraction plus regex parsing of medicine, dose, and frequency."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

KNOWN_MEDICINES = [
    "paracetamol",
    "acetaminophen",
    "amoxicillin",
    "aspirin",
    "warfarin",
    "ibuprofen",
    "methotrexate",
    "metformin",
    "azithromycin",
    "omeprazole",
    "atorvastatin",
    "amlodipine",
    "cetirizine",
    "pantoprazole",
    "dolo",
    "crocin",
    "combiflam",
]

FREQUENCY_MAP = {
    "od": {"code": "OD", "label": "Once daily", "morning": True, "afternoon": False, "night": False},
    "qd": {"code": "OD", "label": "Once daily", "morning": True, "afternoon": False, "night": False},
    "bd": {"code": "BD", "label": "Twice daily", "morning": True, "afternoon": False, "night": True},
    "bid": {"code": "BD", "label": "Twice daily", "morning": True, "afternoon": False, "night": True},
    "tds": {"code": "TDS", "label": "Thrice daily", "morning": True, "afternoon": True, "night": True},
    "tid": {"code": "TDS", "label": "Thrice daily", "morning": True, "afternoon": True, "night": True},
    "qid": {"code": "QID", "label": "Four times daily", "morning": True, "afternoon": True, "night": True},
    "hs": {"code": "HS", "label": "At bedtime", "morning": False, "afternoon": False, "night": True},
    "sos": {"code": "SOS", "label": "As needed", "morning": False, "afternoon": False, "night": False},
}

TIMING_PATTERNS = {
    "1-0-1": {"code": "1-0-1", "label": "Morning and night", "morning": True, "afternoon": False, "night": True},
    "1-1-1": {"code": "1-1-1", "label": "Morning, afternoon, night", "morning": True, "afternoon": True, "night": True},
    "1-0-0": {"code": "1-0-0", "label": "Morning only", "morning": True, "afternoon": False, "night": False},
    "0-0-1": {"code": "0-0-1", "label": "Night only", "morning": False, "afternoon": False, "night": True},
    "1-1-0": {"code": "1-1-0", "label": "Morning and afternoon", "morning": True, "afternoon": True, "night": False},
    "0-1-0": {"code": "0-1-0", "label": "Afternoon only", "morning": False, "afternoon": True, "night": False},
    "0-1-1": {"code": "0-1-1", "label": "Afternoon and night", "morning": False, "afternoon": True, "night": True},
}

DEMO_FALLBACK = [
    {
        "name": "Paracetamol",
        "dosage": "500mg",
        "frequency_code": "1-0-1",
        "frequency_label": "Morning and night",
        "morning": True,
        "afternoon": False,
        "night": True,
        "instructions": "after food",
        "source": "demo-fallback",
    },
    {
        "name": "Amoxicillin",
        "dosage": "250mg",
        "frequency_code": "TDS",
        "frequency_label": "Thrice daily",
        "morning": True,
        "afternoon": True,
        "night": True,
        "instructions": "after food for 5 days",
        "source": "demo-fallback",
    },
]


@lru_cache(maxsize=1)
def _get_reader():
    import easyocr

    return easyocr.Reader(["en"], gpu=False, verbose=False)


def _normalize_freq(token: str) -> dict[str, Any] | None:
    cleaned = token.lower().strip().replace(" ", "")
    cleaned = cleaned.replace("–", "-").replace("—", "-")
    if cleaned in TIMING_PATTERNS:
        return TIMING_PATTERNS[cleaned]
    if cleaned in FREQUENCY_MAP:
        return FREQUENCY_MAP[cleaned]
    return None


def _extract_dosage(text: str) -> str:
    match = re.search(r"(\d+(?:\.\d+)?\s?(?:mg|mcg|g|ml|iu)\b)", text, flags=re.I)
    if match:
        return match.group(1).replace(" ", "").lower().replace("mg", "mg")
    return "—"


def _canonical_name(blob: str) -> str | None:
    lower = blob.lower()
    for name in sorted(KNOWN_MEDICINES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(name)}\b", lower):
            return name.title() if name != "dolo" else "Paracetamol"
    return None


def parse_prescription_text(raw_text: str) -> list[dict[str, Any]]:
    """Turn OCR / typed text into structured medicine rows."""
    medicines: list[dict[str, Any]] = []
    seen: set[str] = set()
    lines = [ln.strip() for ln in re.split(r"[\n;]+", raw_text) if ln.strip()]
    joined = " | ".join(lines)

    freq_from_line: dict[str, Any] | None = None
    for pattern in list(TIMING_PATTERNS) + list(FREQUENCY_MAP):
        if re.search(rf"\b{re.escape(pattern)}\b", joined, flags=re.I):
            freq_from_line = _normalize_freq(pattern)
            break
    timing_hit = re.search(r"\b([01])\s*[-/]\s*([01])\s*[-/]\s*([01])\b", joined)
    if timing_hit:
        freq_from_line = _normalize_freq("-".join(timing_hit.groups()))

    after_food = bool(re.search(r"after\s+food|p\.?\s*c\.?|post\s*prandial", joined, flags=re.I))
    before_food = bool(re.search(r"before\s+food|a\.?\s*c\.?|empty\s+stomach", joined, flags=re.I))
    instructions = "after food" if after_food else ("before food" if before_food else "as directed")

    for line in lines + [joined]:
        name = _canonical_name(line)
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        local_freq = None
        local_timing = re.search(r"\b([01])\s*[-/]\s*([01])\s*[-/]\s*([01])\b", line)
        if local_timing:
            local_freq = _normalize_freq("-".join(local_timing.groups()))
        else:
            for token in re.findall(r"\b[A-Za-z]{2,4}\b", line):
                local_freq = _normalize_freq(token)
                if local_freq:
                    break
        freq = local_freq or freq_from_line or FREQUENCY_MAP["bd"]
        medicines.append(
            {
                "name": name,
                "dosage": _extract_dosage(line if _extract_dosage(line) != "—" else joined),
                "frequency_code": freq["code"],
                "frequency_label": freq["label"],
                "morning": freq["morning"],
                "afternoon": freq["afternoon"],
                "night": freq["night"],
                "instructions": instructions,
                "source": "ocr",
            }
        )
    return medicines


def _read_image(image_path: str | Path | np.ndarray | Image.Image) -> np.ndarray:
    if isinstance(image_path, np.ndarray):
        return image_path
    if isinstance(image_path, Image.Image):
        return np.array(image_path.convert("RGB"))
    return np.array(Image.open(image_path).convert("RGB"))


def extract_medical_data(
    image_path: str | Path | np.ndarray | Image.Image,
    demo_key: str | None = None,
) -> dict[str, Any]:
    """Run EasyOCR and parse medicine rows. Falls back to demo structure if OCR is thin."""
    raw_blocks: list[str] = []
    used_fallback = False
    try:
        image = _read_image(image_path)
        reader = _get_reader()
        results = reader.readtext(image, detail=1, paragraph=False)
        raw_blocks = [item[1] for item in results if item and item[1]]
    except Exception as exc:  # noqa: BLE001 — keep the demo path alive on missing models
        raw_blocks = [f"[ocr-unavailable] {exc}"]

    raw_text = "\n".join(raw_blocks)
    medicines = parse_prescription_text(raw_text)

    expected = _demo_pack(demo_key) if demo_key else []
    parsed_names = {m["name"].lower() for m in medicines}
    expected_names = {m["name"].lower() for m in expected}
    incomplete_demo = bool(expected_names) and not expected_names.issubset(parsed_names)

    if not medicines or incomplete_demo:
        used_fallback = True
        medicines = expected or _demo_pack(demo_key)

    return {
        "raw_text": raw_text if raw_text else "(no text detected)",
        "medicines": medicines,
        "used_fallback": used_fallback,
        "engine": "easyocr" if not used_fallback else "easyocr+fallback",
    }


def _demo_pack(demo_key: str | None) -> list[dict[str, Any]]:
    packs = {
        "safe": DEMO_FALLBACK,
        "bleeding": [
            {
                "name": "Aspirin",
                "dosage": "75mg",
                "frequency_code": "OD",
                "frequency_label": "Once daily",
                "morning": True,
                "afternoon": False,
                "night": False,
                "instructions": "after food",
                "source": "demo-fallback",
            },
            {
                "name": "Warfarin",
                "dosage": "5mg",
                "frequency_code": "OD",
                "frequency_label": "Once daily",
                "morning": False,
                "afternoon": False,
                "night": True,
                "instructions": "same time daily",
                "source": "demo-fallback",
            },
        ],
        "nsaid": [
            {
                "name": "Ibuprofen",
                "dosage": "400mg",
                "frequency_code": "TDS",
                "frequency_label": "Thrice daily",
                "morning": True,
                "afternoon": True,
                "night": True,
                "instructions": "after food",
                "source": "demo-fallback",
            },
            {
                "name": "Aspirin",
                "dosage": "75mg",
                "frequency_code": "OD",
                "frequency_label": "Once daily",
                "morning": True,
                "afternoon": False,
                "night": False,
                "instructions": "after food",
                "source": "demo-fallback",
            },
        ],
        "liver": [
            {
                "name": "Paracetamol",
                "dosage": "1000mg",
                "frequency_code": "QID",
                "frequency_label": "Four times daily",
                "morning": True,
                "afternoon": True,
                "night": True,
                "instructions": "do not exceed 4g/day; avoid alcohol",
                "source": "demo-fallback",
            },
            {
                "name": "Alcohol",
                "dosage": "—",
                "frequency_code": "SOS",
                "frequency_label": "As needed",
                "morning": False,
                "afternoon": False,
                "night": False,
                "instructions": "avoid while on paracetamol",
                "source": "demo-fallback",
            },
        ],
        "toxicity": [
            {
                "name": "Amoxicillin",
                "dosage": "500mg",
                "frequency_code": "TDS",
                "frequency_label": "Thrice daily",
                "morning": True,
                "afternoon": True,
                "night": True,
                "instructions": "after food for 7 days",
                "source": "demo-fallback",
            },
            {
                "name": "Methotrexate",
                "dosage": "15mg",
                "frequency_code": "OD",
                "frequency_label": "Once weekly",
                "morning": True,
                "afternoon": False,
                "night": False,
                "instructions": "weekly dose — confirm with rheumatology",
                "source": "demo-fallback",
            },
        ],
    }
    return packs.get(demo_key or "safe", DEMO_FALLBACK)


def build_dosage_narrative(medicines: list[dict[str, Any]]) -> str:
    if not medicines:
        return "No medicines could be decoded from this prescription."
    parts = []
    for med in medicines:
        slots = []
        if med.get("morning"):
            slots.append("morning")
        if med.get("afternoon"):
            slots.append("afternoon")
        if med.get("night"):
            slots.append("night")
        when = " and ".join(slots) if slots else med.get("frequency_label", "as directed")
        dose = med.get("dosage") or "the prescribed dose"
        how = med.get("instructions") or "as directed"
        parts.append(f"Take {med['name']} {dose} {when} {how}.")
    parts.append("If you feel unwell, contact your doctor or pharmacist before changing any dose.")
    return " ".join(parts)
