"""RxVerify AI — Streamlit dashboard for HACOVERFLOW 10.0 AAROHAN 2026."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from core.image_processor import preprocess_prescription_image
from core.ocr_engine import build_dosage_narrative, extract_medical_data
from core.safety_checker import evaluate_prescription_safety
from core.translator import LANGUAGE_LABELS, translate_dosage_text
from core.tts_engine import synthesize_voice

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
SAMPLES = ASSETS / "samples"

PAGE_BG = "#0F172A"
CYAN = "#00E5FF"
SAFE = "#22C55E"
DANGER = "#EF4444"
CAUTION = "#F59E0B"

SAMPLE_SPECS = {
    "Safe combo — Paracetamol + Amoxicillin": {
        "file": "sample_safe.png",
        "demo_key": "safe",
        "lines": [
            "Dr. Meera Nair  MBBS, MD",
            "Rx",
            "1. Paracetamol 500mg   1-0-1  after food",
            "2. Amoxicillin 250mg   TDS x 5 days",
            "Advice: Hydration. Review if fever persists.",
        ],
    },
    "High risk — Aspirin + Warfarin": {
        "file": "sample_bleeding.png",
        "demo_key": "bleeding",
        "lines": [
            "City Cardiac Clinic",
            "Rx",
            "Aspirin 75mg  OD  after breakfast",
            "Warfarin 5mg  OD  at night",
            "Monitor INR. Report bleeding or bruising.",
        ],
    },
    "Caution — Ibuprofen + Aspirin": {
        "file": "sample_nsaid.png",
        "demo_key": "nsaid",
        "lines": [
            "Ortho OPD",
            "Rx",
            "Ibuprofen 400mg  TDS  after food",
            "Aspirin 75mg  OD  for cardioprotection",
            "Avoid empty stomach. Short course only.",
        ],
    },
    "Liver warning — Paracetamol + Alcohol": {
        "file": "sample_liver.png",
        "demo_key": "liver",
        "lines": [
            "Emergency Rx",
            "Paracetamol 1000mg  QID",
            "Counsel: NO alcohol. Max 4g / 24h.",
        ],
    },
    "Toxicity — Amoxicillin + Methotrexate": {
        "file": "sample_toxicity.png",
        "demo_key": "toxicity",
        "lines": [
            "Rheumatology follow-up",
            "Rx",
            "Methotrexate 15mg  once weekly",
            "Amoxicillin 500mg  TDS x 7 days",
            "Check CBC. Call if mucositis / fever.",
        ],
    },
}


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "segoeui.ttf", "calibri.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def ensure_sample_prescriptions() -> None:
    SAMPLES.mkdir(parents=True, exist_ok=True)
    for spec in SAMPLE_SPECS.values():
        path = SAMPLES / spec["file"]
        if path.exists():
            continue
        img = Image.new("RGB", (900, 640), "#F4EFE4")
        draw = ImageDraw.Draw(img)
        draw.rectangle((28, 28, 872, 612), outline="#1E3A5F", width=3)
        draw.rectangle((28, 28, 872, 118), fill="#0F172A")
        draw.text((48, 48), "RxVerify  ·  Sample Prescription", font=_font(28), fill=CYAN)
        draw.text((48, 84), "HACOVERFLOW 10.0  ·  Healthcare", font=_font(16), fill="#94A3B8")
        y = 150
        for i, line in enumerate(spec["lines"]):
            fill = "#0F172A" if i else "#155E75"
            draw.text((56, y), line, font=_font(26 if i else 22), fill=fill)
            y += 52
        draw.line((56, 520, 260, 560), fill="#334155", width=2)
        draw.text((56, 568), "Physician sign", font=_font(16), fill="#64748B")
        img.save(path)


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
          html, body, [class*="css"]  {{
            font-family: 'Plus Jakarta Sans', sans-serif;
          }}
          .stApp {{
            background:
              radial-gradient(1200px 500px at 10% -10%, rgba(0,229,255,0.12), transparent 50%),
              radial-gradient(900px 400px at 100% 0%, rgba(34,197,94,0.08), transparent 45%),
              {PAGE_BG};
            color: #E2E8F0;
          }}
          .block-container {{ padding-top: 1.2rem; max-width: 1280px; }}
          [data-testid="stSidebar"] {{
            background: #0B1220;
            border-right: 1px solid rgba(0,229,255,0.18);
          }}
          .hero-kicker {{
            letter-spacing: 0.22em;
            text-transform: uppercase;
            color: {CYAN};
            font-size: 0.72rem;
            font-weight: 700;
          }}
          .hero-title {{
            font-size: 2.35rem;
            font-weight: 800;
            margin: 0.15rem 0 0.35rem 0;
            color: #F8FAFC;
          }}
          .hero-sub {{ color: #94A3B8; margin-bottom: 1.1rem; }}
          .badge {{
            display: inline-block;
            padding: 0.28rem 0.7rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 700;
            margin-right: 0.4rem;
            border: 1px solid transparent;
          }}
          @keyframes pulse-cyan {{
            0%, 100% {{ box-shadow: 0 0 0 0 rgba(0,229,255,0.35); }}
            50% {{ box-shadow: 0 0 0 7px rgba(0,229,255,0); }}
          }}
          .badge-cyan {{
            background: rgba(0,229,255,0.12);
            color: {CYAN};
            border-color: rgba(0,229,255,0.35);
            animation: pulse-cyan 2.6s ease-in-out infinite;
          }}
          .badge-green {{ background: rgba(34,197,94,0.12); color: {SAFE}; border-color: rgba(34,197,94,0.35); }}
          .badge-red {{ background: rgba(239,68,68,0.14); color: {DANGER}; border-color: rgba(239,68,68,0.4); }}
          .badge-amber {{ background: rgba(245,158,11,0.14); color: {CAUTION}; border-color: rgba(245,158,11,0.4); }}
          .rx-card {{
            background: linear-gradient(180deg, rgba(15,23,42,0.2), rgba(2,6,23,0.55));
            border: 1px solid rgba(0,229,255,0.22);
            border-radius: 18px;
            padding: 1.1rem 1.2rem;
            box-shadow: 0 0 0 1px rgba(15,23,42,0.4), 0 18px 40px rgba(0,0,0,0.28);
            margin-bottom: 1rem;
          }}
          .metric-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.75rem;
            margin: 0.4rem 0 1.1rem 0;
          }}
          .metric {{
            background: #111827;
            border: 1px solid rgba(148,163,184,0.16);
            border-radius: 14px;
            padding: 0.85rem 1rem;
          }}
          .metric .k {{ color: #94A3B8; font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; }}
          .metric .v {{ color: #F8FAFC; font-size: 1.45rem; font-weight: 800; margin-top: 0.15rem; }}
          .banner-safe {{
            background: linear-gradient(90deg, rgba(34,197,94,0.18), rgba(34,197,94,0.05));
            border: 1px solid {SAFE};
            color: #BBF7D0;
          }}
          .banner-danger {{
            background: linear-gradient(90deg, rgba(239,68,68,0.28), rgba(239,68,68,0.08));
            border: 1px solid {DANGER};
            color: #FECACA;
          }}
          .banner-caution {{
            background: linear-gradient(90deg, rgba(245,158,11,0.22), rgba(245,158,11,0.06));
            border: 1px solid {CAUTION};
            color: #FDE68A;
          }}
          .banner-unknown {{
            background: rgba(148,163,184,0.12);
            border: 1px solid #64748B;
            color: #CBD5E1;
          }}
          .status-banner {{
            border-radius: 16px;
            padding: 1rem 1.15rem;
            font-weight: 700;
            margin-bottom: 0.85rem;
          }}
          .status-banner p {{ font-weight: 500; margin: 0.35rem 0 0 0; opacity: 0.92; }}
          .chip {{
            display: inline-block;
            margin: 0.15rem 0.25rem 0.15rem 0;
            padding: 0.2rem 0.55rem;
            border-radius: 8px;
            background: #1E293B;
            border: 1px solid rgba(0,229,255,0.2);
            font-size: 0.8rem;
          }}
          .warn-item {{
            background: rgba(239,68,68,0.08);
            border-left: 3px solid {DANGER};
            padding: 0.7rem 0.85rem;
            border-radius: 8px;
            margin-bottom: 0.55rem;
          }}
          div[data-testid="stFileUploader"] section {{
            border: 1.5px dashed rgba(0,229,255,0.45) !important;
            background: rgba(0,229,255,0.04) !important;
            border-radius: 16px !important;
          }}
          .stButton>button {{
            background: {CYAN};
            color: #042f36;
            font-weight: 800;
            border: 0;
            border-radius: 12px;
            width: 100%;
          }}
          .stButton>button:hover {{
            background: #67f0ff;
            color: #042f36;
          }}
          [data-testid="stDataFrame"] {{ border-radius: 12px; overflow: hidden; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def timing_badges(row: dict) -> str:
    bits = []
    if row.get("morning"):
        bits.append("☀️ Morning")
    if row.get("afternoon"):
        bits.append("🌤️ Afternoon")
    if row.get("night"):
        bits.append("🌙 Night")
    return "  ".join(bits) if bits else "⏱️ As directed"


def banner_class(status: str) -> str:
    return {
        "safe": "banner-safe",
        "danger": "banner-danger",
        "caution": "banner-caution",
    }.get(status, "banner-unknown")


def load_image_bytes(source: bytes | Path) -> bytes:
    if isinstance(source, Path):
        return source.read_bytes()
    return source


def main() -> None:
    st.set_page_config(
        page_title="RxVerify AI",
        page_icon="💊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()
    ensure_sample_prescriptions()

    st.markdown(
        """
        <div class="hero-kicker">HACOVERFLOW 10.0 · AAROHAN 2026 · Healthcare</div>
        <h1 class="hero-title">RxVerify AI</h1>
        <p class="hero-sub">Multi-lingual prescription decoder &amp; medication safety assistant — scan, verify, translate, speak.</p>
        <div>
          <span class="badge badge-cyan">OCR + CLAHE</span>
          <span class="badge badge-green">Interaction engine</span>
          <span class="badge badge-cyan">ta · hi · te · ml · en</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### 🩺 Control deck")
        language = st.selectbox(
            "Audio & translation language",
            options=list(LANGUAGE_LABELS.keys()),
            format_func=lambda key: LANGUAGE_LABELS[key],
            index=0,
        )
        st.caption("Voice instructions are generated in the language you pick.")

        sample_name = st.selectbox(
            "Sample demo prescriptions",
            options=["— Upload your own —"] + list(SAMPLE_SPECS.keys()),
        )
        uploaded = st.file_uploader(
            "Drop a prescription photo",
            type=["png", "jpg", "jpeg", "webp"],
            help="Smartphone photos of handwritten or printed Rx work best.",
        )
        run = st.button("Decode prescription", type="primary")
        st.markdown("---")
        st.caption(
            "Local rules cover Aspirin–Warfarin, NSAID–Aspirin, Paracetamol–alcohol, "
            "Amoxicillin–Methotrexate, and related pairs. OpenFDA is queried as a live fallback."
        )

    image_bytes: bytes | None = None
    demo_key: str | None = None
    if uploaded is not None:
        image_bytes = uploaded.getvalue()
        demo_key = None
    elif sample_name != "— Upload your own —":
        spec = SAMPLE_SPECS[sample_name]
        image_bytes = (SAMPLES / spec["file"]).read_bytes()
        demo_key = spec["demo_key"]

    if not run:
        st.markdown(
            '<div class="rx-card">Upload a prescription or pick a sample, then press '
            "<strong>Decode prescription</strong>. The workspace stays on-device except for "
            "optional translation, TTS, and OpenFDA lookups.</div>",
            unsafe_allow_html=True,
        )
        cols = st.columns(3)
        cols[0].markdown(
            '<div class="metric"><div class="k">Pipeline</div>'
            '<div class="v">CLAHE → OCR</div></div>',
            unsafe_allow_html=True,
        )
        cols[1].markdown(
            '<div class="metric"><div class="k">Safety DB</div>'
            '<div class="v">Local JSON</div></div>',
            unsafe_allow_html=True,
        )
        cols[2].markdown(
            '<div class="metric"><div class="k">Voice</div>'
            '<div class="v">gTTS vernacular</div></div>',
            unsafe_allow_html=True,
        )
        return

    if not image_bytes:
        st.warning("Add an image or choose a sample prescription first.")
        return

    with st.spinner("Enhancing scan, reading text, and checking interactions…"):
        original_rgb, processed_rgb, _processed_pil = preprocess_prescription_image(image_bytes)
        ocr = extract_medical_data(processed_rgb, demo_key=demo_key)
        medicines = ocr["medicines"]
        safety = evaluate_prescription_safety(medicines)
        narrative = build_dosage_narrative(medicines)
        translated = translate_dosage_text(narrative, language)
        audio_path = synthesize_voice(translated, language)

    left, right = st.columns([1.05, 1.15], gap="large")

    with left:
        st.markdown("#### Scan comparison")
        c1, c2 = st.columns(2)
        c1.image(original_rgb, caption="Original upload", use_container_width=True)
        c2.image(processed_rgb, caption="CLAHE + denoise", use_container_width=True)
        with st.expander("Raw OCR text", expanded=False):
            st.code(ocr.get("raw_text") or "", language=None)
            if ocr.get("used_fallback"):
                st.caption("OCR was ambiguous — structured demo parser filled the medicine table.")

    with right:
        klass = banner_class(safety["status"])
        warning_html = ""
        if safety["warnings"]:
            warning_html = "".join(
                f"<p><strong>{w['pair']}</strong> — {w['risk']}. {w['warning']}</p>"
                for w in safety["warnings"]
            )
        else:
            warning_html = "<p>No conflicting pairs in the local rule set for this list.</p>"

        st.markdown(
            f'<div class="status-banner {klass}">{safety["banner"]} · safety score {safety["score"]}/100'
            f"{warning_html}</div>",
            unsafe_allow_html=True,
        )

        m1, m2, m3 = st.columns(3)
        m1.markdown(
            f'<div class="metric"><div class="k">Medicines</div><div class="v">{len(medicines)}</div></div>',
            unsafe_allow_html=True,
        )
        m2.markdown(
            f'<div class="metric"><div class="k">Flags</div><div class="v">{len(safety["warnings"])}</div></div>',
            unsafe_allow_html=True,
        )
        lang_short = LANGUAGE_LABELS[language].split(" ")[0]
        m3.markdown(
            f'<div class="metric"><div class="k">Voice</div><div class="v">{lang_short}</div></div>',
            unsafe_allow_html=True,
        )

        st.markdown("#### Extracted medicines")
        if medicines:
            table = pd.DataFrame(
                [
                    {
                        "Medicine": row["name"],
                        "Dosage": row["dosage"],
                        "Code": row["frequency_code"],
                        "Schedule": timing_badges(row),
                        "How": row.get("instructions", ""),
                    }
                    for row in medicines
                ]
            )
            st.dataframe(table, hide_index=True, use_container_width=True)
        else:
            st.info("No medicines parsed.")

        st.markdown("#### Regional voice explanation")
        st.write(translated)
        if audio_path and Path(audio_path).exists():
            st.audio(str(audio_path), format="audio/mp3")
        else:
            st.caption("Audio could not be generated (gTTS needs network). Text above is still translated when possible.")

        if safety.get("openfda_notes"):
            with st.expander("OpenFDA fallback notes"):
                for note in safety["openfda_notes"]:
                    st.write(note)

    st.caption("Educational MVP for a hackathon — not medical advice. Confirm every flagged interaction with a licensed clinician.")


if __name__ == "__main__":
    main()
