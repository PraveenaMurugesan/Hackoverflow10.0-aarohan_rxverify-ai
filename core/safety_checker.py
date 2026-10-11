"""Local JSON interaction engine with an OpenFDA label fallback."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

import requests

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "drug_interactions.json"
OPENFDA_URL = "https://api.fda.gov/drug/label.json"
OPENFDA_TIMEOUT = 6


def _load_rules() -> list[dict[str, Any]]:
    with DATA_PATH.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("interactions", [])


def _norm(name: str) -> str:
    return (name or "").strip().lower()


def _local_matches(names: list[str]) -> list[dict[str, Any]]:
    rules = _load_rules()
    present = {_norm(n) for n in names if n}
    hits: list[dict[str, Any]] = []
    for rule in rules:
        drugs = [_norm(d) for d in rule.get("drugs", [])]
        if drugs and all(drug in present for drug in drugs):
            hits.append(rule)
    # Also catch unordered pairs even if a rule lists extras incompletely
    for left, right in combinations(sorted(present), 2):
        for rule in rules:
            rule_drugs = [_norm(d) for d in rule.get("drugs", [])]
            if len(rule_drugs) == 2 and tuple(sorted(rule_drugs)) == (left, right) and rule not in hits:
                hits.append(rule)
    return hits


def _openfda_interaction_blurb(drug_name: str) -> str | None:
    try:
        response = requests.get(
            OPENFDA_URL,
            params={
                "search": f'openfda.generic_name:"{drug_name}"',
                "limit": 1,
            },
            timeout=OPENFDA_TIMEOUT,
        )
        if response.status_code != 200:
            return None
        results = response.json().get("results") or []
        if not results:
            return None
        label = results[0]
        for field in ("drug_interactions", "warnings", "boxed_warning"):
            block = label.get(field)
            if isinstance(block, list) and block:
                text = " ".join(block)[:420]
                return text
            if isinstance(block, str) and block.strip():
                return block.strip()[:420]
    except requests.RequestException:
        return None
    return None


def evaluate_prescription_safety(
    medicines_list: list[dict[str, Any] | str],
) -> dict[str, Any]:
    """Score a decoded prescription and return flags plus clinician-facing warnings."""
    names: list[str] = []
    for item in medicines_list:
        if isinstance(item, dict):
            names.append(str(item.get("name", "")))
        else:
            names.append(str(item))
    names = [n for n in names if n and n.strip() != "—"]

    flags = _local_matches(names)
    high = sum(1 for f in flags if str(f.get("risk_level", "")).lower() == "high")
    moderate = sum(1 for f in flags if str(f.get("risk_level", "")).lower() == "moderate")

    if high:
        status = "danger"
        score = max(12, 38 - high * 8)
        banner = "Dangerous interaction detected"
    elif moderate:
        status = "caution"
        score = 64 - moderate * 6
        banner = "Use with caution"
    elif names:
        status = "safe"
        score = 96
        banner = "No local interaction flags"
    else:
        status = "unknown"
        score = 0
        banner = "No medicines to evaluate"

    openfda_notes: list[str] = []
    if flags:
        for flag in flags:
            for drug in flag.get("drugs", []):
                blurb = _openfda_interaction_blurb(drug)
                if blurb:
                    openfda_notes.append(f"OpenFDA ({drug}): {blurb}")
                    break
    elif names:
        # No local pair, still try a single-label lookup for transparency.
        blurb = _openfda_interaction_blurb(names[0])
        if blurb:
            openfda_notes.append(f"OpenFDA ({names[0]}): {blurb}")

    warnings = [
        {
            "pair": " + ".join(f.get("drugs", [])).title(),
            "risk": f.get("risk"),
            "risk_level": f.get("risk_level"),
            "warning": f.get("warning"),
        }
        for f in flags
    ]

    return {
        "score": int(score),
        "status": status,
        "banner": banner,
        "flags": flags,
        "warnings": warnings,
        "openfda_notes": openfda_notes[:2],
        "evaluated_drugs": names,
    }
