# RxVerify AI

Multi-lingual prescription decoder and medication safety assistant for **HACOVERFLOW 10.0 · AAROHAN 2026** (Healthcare).

RxVerify AI reads a prescription photo, extracts medicine names / doses / timing, flags dangerous drug–drug interactions, then speaks the schedule in Tamil, Hindi, Telugu, Malayalam, or English.

## Quickstart

```bash
cd rxverify-ai
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`).

**First OCR run:** EasyOCR downloads English recognition weights (~100MB). Later launches reuse the cache. If models are missing, the app still runs using structured demo packs from the sidebar.

**Voice + translation:** `deep-translator` and `gTTS` need internet. Safety scoring works fully offline from `data/drug_interactions.json`. OpenFDA is an optional live fallback.

## Demo path (hackathon)

1. Pick a language in the sidebar (try **Tamil** or **Hindi**).
2. Load a **sample prescription** — *Aspirin + Warfarin* shows the red bleeding banner.
3. Compare original vs CLAHE-enhanced scan.
4. Read timing badges (☀️ morning / 🌤️ afternoon / 🌙 night) and play the vernacular audio.

## Project layout

```
rxverify-ai/
├── app.py
├── requirements.txt
├── core/          # vision, OCR, safety, translate, TTS
├── data/          # local interaction rules
└── assets/        # generated samples + audio
```

Not a substitute for a licensed clinician. Always confirm high-risk flags with a pharmacist or doctor.
