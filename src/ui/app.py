"""
NLP Document Classification System — Modern AI Dashboard
src/ui/app.py

Run with: streamlit run src/ui/app.py
"""

import io
import sys
import time
import json
import uuid
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ─── History persistence file ────────────────────────────────────────────────
HISTORY_FILE = Path(__file__).parent.parent.parent / "artifacts" / "activity_history.json"
HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DocuClassify AI — NLP Document Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — SaaS Glassmorphism & Modern Dark Theme
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* ── Global Typography & Background ── */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    .stApp {
        background: radial-gradient(circle at top left, #121324 0%, #0c0d18 100%);
        color: #f1f5f9;
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* ── Hero Header ── */
    .hero-banner {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.12) 0%, rgba(168, 85, 247, 0.12) 50%, rgba(236, 72, 153, 0.08) 100%);
        border: 1px solid rgba(168, 85, 247, 0.25);
        border-radius: 20px;
        padding: 1.8rem 2.2rem;
        margin-bottom: 1.6rem;
        box-shadow: 0 12px 36px rgba(0, 0, 0, 0.35);
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: "";
        position: absolute;
        top: -40px;
        right: -40px;
        width: 140px;
        height: 140px;
        background: radial-gradient(circle, rgba(168, 85, 247, 0.3) 0%, transparent 70%);
        border-radius: 50%;
        pointer-events: none;
    }
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.8px;
        background: linear-gradient(135deg, #ffffff 0%, #c4b5fd 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.02rem;
        margin-bottom: 1rem;
        font-weight: 400;
    }
    .hero-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(30, 32, 54, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.2);
        padding: 0.35rem 0.85rem;
        border-radius: 30px;
        font-size: 0.82rem;
        font-weight: 600;
        color: #cbd5e1;
        backdrop-filter: blur(8px);
    }
    .hero-badge.active {
        border-color: rgba(52, 211, 153, 0.4);
        background: rgba(16, 185, 129, 0.12);
        color: #6ee7b7;
    }

    /* ── Cards & Surfaces ── */
    .card-surface {
        background: linear-gradient(145deg, rgba(26, 27, 46, 0.85), rgba(20, 21, 38, 0.9));
        border: 1px solid rgba(99, 102, 241, 0.18);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.28);
        backdrop-filter: blur(12px);
        margin-bottom: 1.2rem;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .card-surface:hover {
        border-color: rgba(168, 85, 247, 0.35);
    }
    .card-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.9rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ── Mode Switcher Pills ── */
    .mode-bar {
        display: flex;
        gap: 0.5rem;
        background: rgba(15, 16, 28, 0.8);
        padding: 0.4rem;
        border-radius: 12px;
        border: 1px solid rgba(75, 85, 99, 0.3);
        margin-bottom: 1rem;
    }

    /* ── Sample Chip Buttons ── */
    .sample-pill {
        display: inline-block;
        padding: 0.3rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s;
    }

    /* ── Drag & Drop Uploader Styling ── */
    [data-testid="stFileUploader"] {
        background: rgba(18, 20, 36, 0.7) !important;
        border: 2px dashed rgba(129, 140, 248, 0.45) !important;
        border-radius: 16px !important;
        padding: 1rem !important;
        transition: all 0.25s ease !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #a855f7 !important;
        background: rgba(26, 29, 52, 0.85) !important;
        box-shadow: 0 0 24px rgba(168, 85, 247, 0.18) !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        padding: 1.2rem !important;
    }
    .file-summary-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(168, 85, 247, 0.15));
        border: 1px solid rgba(129, 140, 248, 0.35);
        border-radius: 12px;
        padding: 0.9rem 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.9rem;
        margin: 0.8rem 0;
    }

    /* ── Keyword Explainer Badges ── */
    .keyword-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(129, 140, 248, 0.35);
        border-radius: 8px;
        padding: 0.3rem 0.7rem;
        font-size: 0.82rem;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        color: #c7d2fe;
        margin: 0.25rem;
    }

    /* ── Results Banner ── */
    .result-hero {
        border-radius: 18px;
        padding: 1.8rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        margin-bottom: 1.2rem;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.35);
    }
    .result-category {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-top: 0.3rem;
        margin-bottom: 0.2rem;
    }
    .result-meta {
        font-size: 0.88rem;
        color: #94a3b8;
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 1.2rem;
        margin-top: 0.6rem;
    }

    /* ── Multi-Model Consensus Card ── */
    .consensus-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
        gap: 0.7rem;
        margin: 1rem 0;
    }
    .consensus-item {
        background: rgba(18, 19, 34, 0.75);
        border: 1px solid rgba(75, 85, 99, 0.3);
        border-radius: 12px;
        padding: 0.9rem 0.6rem;
        text-align: center;
        transition: transform 0.2s;
    }
    .consensus-item:hover {
        transform: translateY(-2px);
    }

    /* ── Streamlit Tabs Styling ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.6rem;
        background: transparent;
        border-bottom: 2px solid rgba(75, 85, 99, 0.35);
        padding-bottom: 0.2rem;
        margin-bottom: 1.4rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 0.65rem 1.3rem;
        font-weight: 600;
        font-size: 0.94rem;
        color: #94a3b8;
        background: transparent;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #e2e8f0;
        background: rgba(255, 255, 255, 0.04);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.25), rgba(168, 85, 247, 0.25));
        color: #e0e7ff !important;
        border-bottom: 3px solid #818cf8;
    }

    /* ── Action Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        padding: 0.65rem 2rem !important;
        box-shadow: 0 4px 18px rgba(99, 102, 241, 0.35) !important;
        transition: all 0.2s ease !important;
        letter-spacing: 0.3px;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 24px rgba(139, 92, 246, 0.5) !important;
        opacity: 0.95;
    }

    /* ── Textarea Polish ── */
    .stTextArea textarea {
        background: rgba(17, 18, 31, 0.8) !important;
        border: 1px solid rgba(99, 102, 241, 0.25) !important;
        border-radius: 12px !important;
        color: #f1f5f9 !important;
        font-size: 0.95rem !important;
        line-height: 1.6 !important;
    }
    .stTextArea textarea:focus {
        border-color: #a855f7 !important;
        box-shadow: 0 0 0 2px rgba(168, 85, 247, 0.3) !important;
    }

    /* ── Hide Sidebar Completely ── */
    [data-testid="stSidebar"], section[data-testid="stSidebar"] {
        display: none !important;
    }
    [data-testid="collapsedControl"] {
        display: none !important;
    }

    /* ── Metrics Box ── */
    [data-testid="stMetricValue"] {
        color: #e0e7ff;
        font-weight: 800;
        font-size: 1.8rem;
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8;
        font-weight: 600;
    }

    /* ── History Card ── */
    .hist-card {
        background: linear-gradient(145deg, #18192a, #1f2038);
        border-radius: 12px;
        border-left: 4px solid #8b5cf6;
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.85rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .hist-card:hover {
        transform: translateX(3px);
        box-shadow: 0 6px 22px rgba(139, 92, 246, 0.22);
    }

    /* ── Scrollbars ── */
    ::-webkit-scrollbar { width: 7px; height: 7px; }
    ::-webkit-scrollbar-track { background: #0f101d; }
    ::-webkit-scrollbar-thumb { background: #3b3d5c; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #565985; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants & Colors
# ─────────────────────────────────────────────────────────────────────────────

CATEGORY_COLORS = {
    "technology": "#6366f1",   # Indigo
    "sports": "#10b981",       # Emerald
    "politics": "#ef4444",     # Red
    "entertainment": "#f59e0b",# Amber
    "science": "#06b6d4",      # Cyan
    "health": "#ec4899",       # Pink
    "finance": "#3b82f6",      # Blue
    "world": "#a855f7",        # Purple
}

CATEGORY_EMOJIS = {
    "technology": "💻",
    "sports": "⚽",
    "politics": "🏛️",
    "entertainment": "🎬",
    "science": "🔬",
    "health": "🏥",
    "finance": "📈",
    "world": "🌍",
}

CATEGORY_DESCRIPTIONS = {
    "technology": "AI, Hardware, Software, Chips, Cybersecurity, Cloud",
    "sports": "Championships, Football, Basketball, Tournaments, Transfers",
    "politics": "Elections, Parliament, Legislation, Foreign Policy, Governance",
    "entertainment": "Cinema, Music Releases, Awards, Celebrities, Box Office",
    "science": "Astronomy, Quantum Physics, Biology, Climate, Space Missions",
    "health": "Medicine, Clinical Trials, Diseases, Nutrition, Public Health",
    "finance": "Stock Markets, Crypto, Interest Rates, Earnings, M&A",
    "world": "Geopolitics, Natural Disasters, Humanitarian Aid, Peace Treaties",
}

SUPPORTED_FORMATS = {
    "pdf":  {"label": "PDF",   "emoji": "📕", "color": "#ef4444"},
    "docx": {"label": "Word",  "emoji": "📘", "color": "#3b82f6"},
    "doc":  {"label": "Word",  "emoji": "📘", "color": "#3b82f6"},
    "xlsx": {"label": "Excel", "emoji": "📗", "color": "#10b981"},
    "xls":  {"label": "Excel", "emoji": "📗", "color": "#10b981"},
    "pptx": {"label": "PPT",   "emoji": "📙", "color": "#f97316"},
    "ppt":  {"label": "PPT",   "emoji": "📙", "color": "#f97316"},
    "txt":  {"label": "Text",  "emoji": "📄", "color": "#94a3b8"},
    "csv":  {"label": "CSV",   "emoji": "📊", "color": "#a855f7"},
    "json": {"label": "JSON",  "emoji": "🗂️",  "color": "#fbbf24"},
    "rtf":  {"label": "RTF",   "emoji": "📝", "color": "#64748b"},
    "md":   {"label": "Markdown", "emoji": "📋", "color": "#06b6d4"},
}

SAMPLE_DOCUMENTS = {
    "Technology — Apple M4 & AI Neural Engine": (
        "Apple unveiled its next-generation MacBook Pro powered by the revolutionary M4 Max chip, "
        "featuring a 16-core CPU and an advanced 40-core GPU. The built-in Neural Engine accelerates "
        "on-device machine learning models, allowing developers to run open-source LLMs locally at "
        "unprecedented speed without cloud connectivity. Industry analysts predict this hardware "
        "leap will redefine mobile workstation computing across the software engineering landscape."
    ),
    "Sports — Champions League Final Hat-Trick": (
        "Manchester City defeated Real Madrid 3-1 in a thrilling UEFA Champions League final at Wembley Stadium. "
        "Erling Haaland delivered a career-defining performance by scoring a sensational hat-trick in the "
        "second half, becoming only the second player in history to score three goals in a European final. "
        "The victory secured the treble for Pep Guardiola's squad amidst roaring celebration from 88,000 fans."
    ),
    "Politics — Climate Emergency Act Passes Senate": (
        "The Senate voted 312 to 128 to pass the Landmark Clean Energy and Infrastructure Act today, "
        "allocating over $400 billion toward nationwide renewable energy grids and decarbonization targets. "
        "Opposition lawmakers questioned the aggressive timeline and deficit impacts, but the President "
        "hailed the bipartisan vote as an unprecedented turning point in international climate diplomacy."
    ),
    "Entertainment — Record-Breaking Album Release": (
        "Beyoncé shattered all existing global streaming benchmarks with the surprise launch of her 24-track "
        "album 'Renaissance Act II', accumulating over 220 million Spotify streams within the first 24 hours. "
        "Music critics praised the seamless blending of country, Afrobeats, and house music, calling it an "
        "uncompromising artistic triumph that is practically guaranteed to dominate the upcoming Grammy Awards."
    ),
    "Science — James Webb Detects Europa Ocean Plumes": (
        "NASA's James Webb Space Telescope has captured high-resolution infrared spectrographic data confirming "
        "active water vapor plumes erupting from Jupiter's icy moon Europa. Astrobiologists at the Jet Propulsion "
        "Laboratory detected organic carbon signatures and amino compounds within the plumes, providing the most "
        "compelling evidence to date that Europa's subterranean ocean possesses conditions capable of supporting life."
    ),
    "Health — Clinical Trial Reverses Heart Disease Risk": (
        "A 12-year longitudinal clinical study published in The New England Journal of Medicine demonstrates that "
        "a combined Mediterranean diet and daily 35-minute aerobic regimen reduces the incidence of coronary artery "
        "disease by 48% in patients over 50. Researchers identified a key reduction in arterial inflammation markers "
        "and improved endothelial elasticity among the 40,000 international participants."
    ),
    "Finance — Federal Reserve Cuts Rates Amid Market Rally": (
        "Global financial markets surged following the Federal Reserve's announcement to reduce the benchmark "
        "federal funds rate by 50 basis points. The S&P 500 gained 2.8% to reach a fresh all-time high, while the "
        "Nasdaq rallied 3.6% led by semiconductor heavyweights. Bitcoin rallied past $88,000 as institutional "
        "traders anticipated accelerated liquidity expansion across digital asset markets."
    ),
    "World — International Relief for Coastal Earthquake": (
        "A magnitude 7.3 earthquake struck the coast of central Japan early Tuesday morning, causing structural "
        "damage across four prefectures and triggering immediate coastal tsunami advisories. The United Nations "
        "Humanitarian Office coordinated with rescue teams from Australia, South Korea, and the Red Cross to "
        "deliver emergency medical supplies and establish temporary shelter for over 15,000 displaced residents."
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# Document Parsers
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF using PyMuPDF."""
    try:
        import pymupdf
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
        pages = [page.get_text("text") for page in doc]
        return "\n\n".join(pages)
    except Exception as e:
        return f"[Error reading PDF: {e}]"


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from .docx Word document."""
    try:
        from docx import Document
        doc = Document(io.BytesIO(file_bytes))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    paragraphs.append(row_text)
        return "\n".join(paragraphs)
    except Exception as e:
        return f"[Error reading DOCX: {e}]"


def extract_text_from_excel(file_bytes: bytes, ext: str) -> str:
    """Extract text from Excel worksheets."""
    try:
        engine = "openpyxl" if ext == "xlsx" else "xlrd"
        xls = pd.ExcelFile(io.BytesIO(file_bytes), engine=engine)
        texts = []
        for sheet in xls.sheet_names:
            df = xls.parse(sheet)
            texts.append(f"[Sheet: {sheet}]")
            texts.append(df.to_string(index=False, na_rep=""))
        return "\n\n".join(texts)
    except Exception as e:
        return f"[Error reading Excel: {e}]"


def extract_text_from_pptx(file_bytes: bytes) -> str:
    """Extract text from PowerPoint slides."""
    try:
        from pptx import Presentation
        prs = Presentation(io.BytesIO(file_bytes))
        slides = []
        for i, slide in enumerate(prs.slides, 1):
            parts = [f"── Slide {i} ──"]
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    parts.append(shape.text.strip())
            slides.append("\n".join(parts))
        return "\n\n".join(slides)
    except Exception as e:
        return f"[Error reading PPTX: {e}]"


def extract_text_from_rtf(file_bytes: bytes) -> str:
    """Extract text from RTF."""
    try:
        from striprtf.striprtf import rtf_to_text
        return rtf_to_text(file_bytes.decode("latin-1", errors="replace"))
    except Exception as e:
        return f"[Error reading RTF: {e}]"


def extract_text_from_json(file_bytes: bytes) -> str:
    """Flatten JSON content to readable text."""
    try:
        data = json.loads(file_bytes.decode("utf-8", errors="replace"))
        def _flatten(obj, prefix=""):
            lines = []
            if isinstance(obj, dict):
                for k, v in obj.items():
                    lines.extend(_flatten(v, f"{prefix}{k}: "))
            elif isinstance(obj, list):
                for item in obj:
                    lines.extend(_flatten(item, prefix))
            else:
                lines.append(f"{prefix}{obj}")
            return lines
        return "\n".join(_flatten(data))
    except Exception as e:
        return f"[Error reading JSON: {e}]"


def parse_uploaded_file(uploaded_file) -> Tuple[str, str]:
    """Parse any uploaded document file."""
    if uploaded_file is None:
        return "", "No file uploaded."

    file_bytes = uploaded_file.read()
    name = uploaded_file.name
    ext = Path(name).suffix.lstrip(".").lower()

    if ext == "pdf":
        text = extract_text_from_pdf(file_bytes)
    elif ext in ("docx", "doc"):
        text = extract_text_from_docx(file_bytes)
    elif ext == "xlsx":
        text = extract_text_from_excel(file_bytes, "xlsx")
    elif ext == "xls":
        text = extract_text_from_excel(file_bytes, "xls")
    elif ext in ("pptx", "ppt"):
        text = extract_text_from_pptx(file_bytes)
    elif ext in ("txt", "md"):
        text = file_bytes.decode("utf-8", errors="replace")
    elif ext == "csv":
        try:
            df = pd.read_csv(io.BytesIO(file_bytes))
            text = df.to_string(index=False, na_rep="")
        except Exception as e:
            return "", f"Error reading CSV: {e}"
    elif ext == "json":
        text = extract_text_from_json(file_bytes)
    elif ext == "rtf":
        text = extract_text_from_rtf(file_bytes)
    else:
        return "", f"Unsupported file format: .{ext}"

    if text.startswith("[Error"):
        return "", text
    return text.strip(), ""


# ─────────────────────────────────────────────────────────────────────────────
# Model & Asset Loading
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    """Load all available trained models (cached across reruns)."""
    from src.data.preprocessor import TextPreprocessor
    from src.models.traditional import TraditionalClassifier
    from src.utils.config import load_config

    config = load_config("configs/config.yaml")
    artifacts_dir = Path(config.get("training", {}).get("artifacts_dir", "artifacts"))

    preprocessor = TextPreprocessor(
        lowercase=True,
        remove_punctuation=True,
        remove_stopwords=True,
        lemmatization=True,
        min_word_length=2,
    )

    models = {}
    categories = []
    model_names = ["logistic_regression", "svm", "naive_bayes", "random_forest"]

    for name in model_names:
        model_path = artifacts_dir / "models" / name / f"{name}_pipeline.joblib"
        if model_path.exists():
            try:
                model = TraditionalClassifier.load(str(model_path))
                models[name] = model
                if not categories and model.categories:
                    categories = model.categories
            except Exception as e:
                st.warning(f"Could not load {name}: {e}")

    return models, preprocessor, categories


def load_evaluation_report():
    report_path = Path("artifacts/evaluation_report.json")
    if report_path.exists():
        with open(report_path) as f:
            return json.load(f)
    return {}


def load_comparison_csv():
    csv_path = Path("artifacts/model_comparison.csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# NLP Explainability Helper
# ─────────────────────────────────────────────────────────────────────────────

def extract_key_salient_terms(model, clean_text: str, top_n: int = 7) -> List[Tuple[str, float]]:
    """Extract the most influential TF-IDF terms from the document."""
    try:
        tfidf = model.pipeline.named_steps["tfidf"]
        feature_names = tfidf.get_feature_names_out()
        tfidf_vec = tfidf.transform([clean_text])

        # Get non-zero indices & values
        indices = tfidf_vec.indices
        data = tfidf_vec.data

        if len(data) == 0:
            return []

        scored_terms = []
        for idx, score in zip(indices, data):
            term = feature_names[idx]
            scored_terms.append((term, float(score)))

        # Sort descending by TF-IDF weight
        scored_terms.sort(key=lambda x: x[1], reverse=True)
        return scored_terms[:top_n]
    except Exception:
        # Fallback to simple word frequency if vectorizer not accessible
        tokens = [w for w in clean_text.split() if len(w) > 3]
        counts = pd.Series(tokens).value_counts().head(top_n)
        return [(word, float(cnt)) for word, cnt in counts.items()]


# ─────────────────────────────────────────────────────────────────────────────
# Activity History
# ─────────────────────────────────────────────────────────────────────────────

def _load_history_from_disk() -> List[Dict]:
    try:
        if HISTORY_FILE.exists():
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return []


def _save_history_to_disk(history: List[Dict]) -> None:
    try:
        HISTORY_FILE.write_text(
            json.dumps(history[-500:], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def init_history() -> None:
    if "activity_history" not in st.session_state:
        st.session_state.activity_history = _load_history_from_disk()


def add_history_entry(
    *,
    entry_type: str,
    input_source: str,
    text_preview: str,
    word_count: int,
    model: str,
    predicted_label: str,
    confidence: float,
    elapsed_ms: float,
    probabilities: Dict[str, float],
    batch_count: int = 0,
) -> None:
    init_history()
    entry = {
        "id": str(uuid.uuid4())[:8],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": entry_type,
        "input_source": input_source,
        "text_preview": text_preview[:220],
        "word_count": word_count,
        "model": model,
        "predicted_label": predicted_label,
        "confidence": round(confidence, 4),
        "elapsed_ms": round(elapsed_ms, 2),
        "probabilities": {k: round(v, 4) for k, v in probabilities.items()},
        "batch_count": batch_count,
    }
    st.session_state.activity_history.insert(0, entry)
    _save_history_to_disk(st.session_state.activity_history)


def clear_history() -> None:
    st.session_state.activity_history = []
    _save_history_to_disk([])


# ─────────────────────────────────────────────────────────────────────────────
# Model & History Initialization
# ─────────────────────────────────────────────────────────────────────────────

init_history()
_hist_count = len(st.session_state.activity_history)
models_dict, preprocessor, categories = load_models()

# ─────────────────────────────────────────────────────────────────────────────
# Top Hero Banner
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-title">
            <span>🧠</span> NLP Document Classification System
        </div>
        <div class="hero-subtitle">
            Classify raw text or entire files across 8 major domains with multi-algorithm consensus, instant explanations, and deep analytics.
        </div>
        <div class="hero-badges">
            <span class="hero-badge active">● System Live</span>
            <span class="hero-badge">⚡ 4 ML Models Active</span>
            <span class="hero-badge">📁 10+ File Formats Supported</span>
            <span class="hero-badge">🎯 Sub-millisecond Inference</span>
            <span class="hero-badge">🔍 Salient Word Explanations</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Main Navigation Tabs
# ─────────────────────────────────────────────────────────────────────────────

tab_classify, tab_models, tab_history = st.tabs(
    [
        "🔍 Classify & Explain",
        "🏆 Model Leaderboard",
        f"🕐 Activity Log ({_hist_count})",
    ]
)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Classify & Explain
# ═════════════════════════════════════════════════════════════════════════════

with tab_classify:

    # ── Model Selection ──
    if models_dict:
        selected_model = st.selectbox(
            "🤖 Classification Model",
            options=list(models_dict.keys()),
            format_func=lambda x: {
                "logistic_regression": "⚡ Logistic Regression (Fast)",
                "svm": "🎯 Linear SVM (High Margin)",
                "naive_bayes": "⚡ Multinomial Naive Bayes",
                "random_forest": "🌲 Random Forest (Ensemble)",
            }.get(x, x.replace("_", " ").title()),
            index=0,
        )
    else:
        selected_model = None
        st.error("⚠️ No models detected. Run python train.py first.")

    top_k = 3

    st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)

    col_input, col_output = st.columns([1.1, 0.9], gap="large")

    with col_input:
        st.markdown(
            '<div class="card-title">📝 Document Input Mode</div>',
            unsafe_allow_html=True,
        )

        input_mode = st.radio(
            "Choose your input method:",
            ["📁 Drag & Drop Document File", "✍️ Write / Paste Text & Quick Samples"],
            horizontal=True,
            label_visibility="collapsed",
        )

        user_text = ""
        active_source_label = "direct text"

        # ── Mode A: Drag & Drop Document File ──
        if input_mode == "📁 Drag & Drop Document File":
            uploaded_file = st.file_uploader(
                "📥 Drag & drop your document here, or click to browse:",
                type=list(SUPPORTED_FORMATS.keys()),
                help="Supported formats: PDF, DOCX, XLSX, PPTX, TXT, CSV, JSON, RTF, Markdown",
                label_visibility="visible",
            )
            st.caption("Supports: **PDF** · **Word (DOCX)** · **Excel** · **PowerPoint** · **TXT** · **CSV** · **JSON** · **Markdown**")

            if uploaded_file is not None:
                active_source_label = uploaded_file.name
                ext = Path(uploaded_file.name).suffix.lstrip(".").lower()
                fmt_info = SUPPORTED_FORMATS.get(ext, {"emoji": "📄", "label": ext.upper(), "color": "#818cf8"})

                with st.spinner(f"Extracting content from {fmt_info['label']} file…"):
                    extracted_text, error_msg = parse_uploaded_file(uploaded_file)

                if error_msg:
                    st.error(f"❌ {error_msg}")
                elif not extracted_text:
                    st.warning("⚠️ No readable text content found in this file.")
                else:
                    user_text = extracted_text
                    wc = len(user_text.split())
                    cc = len(user_text)

                    st.markdown(
                        f"""
                        <div class="file-summary-card">
                            <span style="font-size: 2rem;">{fmt_info['emoji']}</span>
                            <div style="flex: 1;">
                                <div style="font-weight: 700; color: #f1f5f9; font-size: 0.95rem;">
                                    {uploaded_file.name}
                                </div>
                                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 0.2rem;">
                                    {fmt_info['label']} Format · {uploaded_file.size / 1024:.1f} KB · <strong>{wc:,} words</strong> ({cc:,} chars)
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    with st.expander("👁️ Inspect & Edit Extracted Content", expanded=True):
                        edited_text = st.text_area(
                            "Extracted Text",
                            value=user_text,
                            height=170,
                            label_visibility="collapsed",
                            key="extracted_doc_text",
                        )
                        if edited_text:
                            user_text = edited_text

        # ── Mode B: Direct Text & Quick Samples ──
        else:
            st.markdown(
                '<div style="font-size:0.85rem; color:#94a3b8; margin: 0.6rem 0 0.4rem;">'
                '🚀 <strong>One-Click Benchmark Samples:</strong>'
                '</div>',
                unsafe_allow_html=True,
            )

            # Interactive category sample selectbox
            sample_options = ["-- None / Write My Own Text --"] + list(SAMPLE_DOCUMENTS.keys())
            chosen_sample = st.selectbox(
                "Choose a sample article:",
                options=sample_options,
                label_visibility="collapsed",
            )

            if chosen_sample != "-- None / Write My Own Text --":
                preset_text = SAMPLE_DOCUMENTS[chosen_sample]
                active_source_label = chosen_sample.split(" — ")[0]
            else:
                preset_text = ""
                active_source_label = "manual text input"

            # Quick action buttons for samples
            sample_cols = st.columns(4)
            quick_keys = list(SAMPLE_DOCUMENTS.keys())
            with sample_cols[0]:
                if st.button("💻 Tech Sample", use_container_width=True):
                    st.session_state["live_user_input"] = SAMPLE_DOCUMENTS[quick_keys[0]]
                    st.rerun()
            with sample_cols[1]:
                if st.button("⚽ Sports Sample", use_container_width=True):
                    st.session_state["live_user_input"] = SAMPLE_DOCUMENTS[quick_keys[1]]
                    st.rerun()
            with sample_cols[2]:
                if st.button("🔬 Science Sample", use_container_width=True):
                    st.session_state["live_user_input"] = SAMPLE_DOCUMENTS[quick_keys[4]]
                    st.rerun()
            with sample_cols[3]:
                if st.button("📈 Finance Sample", use_container_width=True):
                    st.session_state["live_user_input"] = SAMPLE_DOCUMENTS[quick_keys[6]]
                    st.rerun()

            # Main Text Input Area
            initial_val = st.session_state.get("live_user_input", preset_text)
            user_text = st.text_area(
                "Document Text",
                value=initial_val,
                height=220,
                placeholder="Paste any article, news story, report, or paragraph here…",
                label_visibility="collapsed",
                key="text_box_input",
            )
            st.session_state["live_user_input"] = user_text

            # Live text metrics counter
            if user_text.strip():
                wc = len(user_text.split())
                cc = len(user_text)
                sc = len(re.split(r'[.!?]+', user_text)) - 1
                sc = max(sc, 1)
                read_time = max(round(wc / 200, 1), 0.1)

                st.markdown(
                    f"""
                    <div style="display:flex; gap:1rem; margin-top:0.4rem; padding:0.45rem 0.8rem; background:rgba(30,32,54,0.6); border-radius:10px; border:1px solid rgba(75,85,99,0.25); font-size:0.8rem; color:#94a3b8;">
                        <span>📝 <strong>{wc:,}</strong> words</span>
                        <span>🔤 <strong>{cc:,}</strong> characters</span>
                        <span>📄 <strong>{sc}</strong> sentences</span>
                        <span>⏱️ <strong>~{read_time} min</strong> read</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # Primary Action Buttons
        btn_col1, btn_col2 = st.columns([1.5, 1])
        with btn_col1:
            classify_btn = st.button(
                "🚀 Classify Document",
                use_container_width=True,
                disabled=not models_dict,
                type="primary",
            )
        with btn_col2:
            consensus_btn = st.button(
                "⚡ Test All 4 Models",
                use_container_width=True,
                disabled=not models_dict,
                help="Runs Logistic Regression, SVM, Naive Bayes, and Random Forest simultaneously",
            )

    # ── Right Column: Results & Explainability Panel ──
    with col_output:
        st.markdown(
            '<div class="card-title">🎯 Classification Output & Insights</div>',
            unsafe_allow_html=True,
        )

        should_run = (classify_btn or consensus_btn) and user_text.strip() and selected_model

        if should_run:
            active_model = models_dict[selected_model]
            clean_text = preprocessor.preprocess(user_text)

            with st.spinner("Analyzing semantics & calculating class probabilities…"):
                t_start = time.time()
                result = active_model.predict_single(clean_text)
                elapsed_ms = (time.time() - t_start) * 1000

            pred_label = result["predicted_label"]
            confidence = result["confidence"]
            theme_color = CATEGORY_COLORS.get(pred_label, "#6366f1")
            theme_emoji = CATEGORY_EMOJIS.get(pred_label, "📌")
            probs = result["probabilities"]

            # Save to activity history
            add_history_entry(
                entry_type="file" if input_mode != "✍️ Direct Text & Quick Samples" else "single",
                input_source=active_source_label,
                text_preview=user_text[:200],
                word_count=len(user_text.split()),
                model=selected_model,
                predicted_label=pred_label,
                confidence=confidence,
                elapsed_ms=elapsed_ms,
                probabilities=probs,
            )

            # ── 1. Hero Classification Card ──
            st.markdown(
                f"""
                <div class="result-hero" style="background: linear-gradient(135deg, {theme_color}25 0%, rgba(20,21,38,0.95) 100%); border: 2px solid {theme_color}77;">
                    <div style="font-size:3.2rem; margin-bottom:0.1rem; filter: drop-shadow(0 0 14px {theme_color}88);">{theme_emoji}</div>
                    <div class="result-category" style="color: {theme_color};">
                        {pred_label}
                    </div>
                    <div style="font-size: 1.15rem; color: #f1f5f9; font-weight:700;">
                        Confidence: <span style="color:{theme_color}; font-size:1.35rem;">{confidence * 100:.1f}%</span>
                    </div>
                    <div class="result-meta">
                        <span>🤖 <strong>{selected_model.replace('_', ' ').title()}</strong></span>
                        <span>⚡ <strong>{elapsed_ms:.1f} ms</strong> latency</span>
                        <span>📄 <strong>{len(user_text.split()):,}</strong> words</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ── 2. Confidence Gauge & Probability Distribution ──
            g_col, b_col = st.columns([1, 1.4])
            with g_col:
                fig_gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=confidence * 100,
                        number={"suffix": "%", "font": {"size": 28, "color": theme_color, "family": "Plus Jakarta Sans"}},
                        gauge={
                            "axis": {"range": [0, 100], "tickcolor": "#475569"},
                            "bar": {"color": theme_color, "thickness": 0.4},
                            "bgcolor": "#131424",
                            "borderwidth": 0,
                            "steps": [
                                {"range": [0, 40], "color": "#18192d"},
                                {"range": [40, 75], "color": "#232644"},
                                {"range": [75, 100], "color": "#2e325c"},
                            ],
                        },
                    )
                )
                fig_gauge.update_layout(
                    height=180,
                    margin=dict(l=15, r=15, t=10, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#f1f5f9"),
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            with b_col:
                # Sorted Top-K probability horizontal bar chart
                sorted_p = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:top_k]
                p_labels = [p[0].capitalize() for p in sorted_p]
                p_vals = [round(p[1] * 100, 2) for p in sorted_p]
                p_colors = [CATEGORY_COLORS.get(p[0], "#6366f1") for p in sorted_p]

                fig_bars = go.Figure(
                    go.Bar(
                        x=p_vals,
                        y=p_labels,
                        orientation="h",
                        marker=dict(color=p_colors, line=dict(width=0)),
                        text=[f"{v:.1f}%" for v in p_vals],
                        textposition="outside",
                    )
                )
                fig_bars.update_layout(
                    title=dict(text="Top Category Probabilities", font=dict(size=12, color="#94a3b8")),
                    xaxis_range=[0, 118],
                    height=180,
                    margin=dict(l=10, r=40, t=25, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#f8fafc", size=11),
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig_bars, use_container_width=True)

            # ── 3. Salient Word Explanations (NLP Interpretability) ──
            salient_terms = extract_key_salient_terms(active_model, clean_text, top_n=8)
            if salient_terms:
                st.markdown(
                    '<div style="font-size:0.85rem; font-weight:700; color:#cbd5e1; margin-top:0.6rem; margin-bottom:0.3rem;">'
                    '🔍 <strong>Predictive Keywords in Document:</strong>'
                    '</div>',
                    unsafe_allow_html=True,
                )
                kw_html = '<div style="display:flex; flex-wrap:wrap;">'
                for term, weight in salient_terms:
                    kw_html += f'<span class="keyword-badge">#{term} <small style="opacity:0.65">({weight:.2f})</small></span>'
                kw_html += '</div>'
                st.markdown(kw_html, unsafe_allow_html=True)

            # ── 4. Multi-Model Consensus Breakdown (if requested) ──
            if consensus_btn:
                st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
                st.markdown(
                    '<div style="font-size:0.92rem; font-weight:700; color:#f8fafc; margin-top:0.6rem;">'
                    '⚡ Multi-Model Consensus Agreement'
                    '</div>',
                    unsafe_allow_html=True,
                )

                consensus_results = {}
                for m_name, m_inst in models_dict.items():
                    m_pred = m_inst.predict_single(clean_text)
                    consensus_results[m_name] = m_pred

                # Consensus score
                all_preds = [v["predicted_label"] for v in consensus_results.values()]
                most_common = pd.Series(all_preds).value_counts()
                consensus_rate = (most_common.iloc[0] / len(all_preds)) * 100

                cons_cols = st.columns(len(consensus_results))
                for i, (m_name, m_pred) in enumerate(consensus_results.items()):
                    c_col = CATEGORY_COLORS.get(m_pred["predicted_label"], "#818cf8")
                    c_emoji = CATEGORY_EMOJIS.get(m_pred["predicted_label"], "📌")
                    with cons_cols[i]:
                        st.markdown(
                            f"""
                            <div class="consensus-item" style="border-top: 3px solid {c_col};">
                                <div style="font-size:0.75rem; color:#94a3b8; font-weight:600;">
                                    {m_name.replace('_', ' ').title()}
                                </div>
                                <div style="font-size:1.4rem; margin:0.2rem 0;">{c_emoji}</div>
                                <div style="font-weight:700; font-size:0.9rem; color:{c_col};">
                                    {m_pred["predicted_label"].capitalize()}
                                </div>
                                <div style="font-size:0.75rem; color:#cbd5e1;">
                                    {m_pred["confidence"]*100:.1f}%
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                agreement_badge = (
                    "🟢 100% Unanimous Consensus" if consensus_rate == 100
                    else f"🟡 {consensus_rate:.0f}% Model Majority Consensus"
                )
                st.info(f"**{agreement_badge}** across all 4 machine learning pipelines.")

        elif (classify_btn or consensus_btn) and not user_text.strip():
            st.warning("⚠️ Please provide or paste document text, or choose a benchmark sample.")
        else:
            # Standby Placeholder
            st.markdown(
                """
                <div style="border: 2px dashed rgba(99, 102, 241, 0.25); border-radius: 18px; padding: 3.5rem 1.5rem; text-align: center; background: rgba(18, 19, 34, 0.4);">
                    <div style="font-size: 3.2rem; margin-bottom: 0.8rem; filter: drop-shadow(0 0 16px rgba(129, 140, 248, 0.3));">✨</div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #f8fafc;">
                        Classifier Ready for Input
                    </div>
                    <div style="font-size: 0.88rem; color: #94a3b8; max-width: 380px; margin: 0.5rem auto 1.2rem;">
                        Type text, select a preset sample, or drop a document on the left, then click <strong>Classify Document</strong>.
                    </div>
                    <div style="display:inline-flex; gap:0.5rem; font-size:0.75rem; color:#818cf8; background:rgba(99,102,241,0.1); border:1px solid rgba(99,102,241,0.25); border-radius:20px; padding:0.35rem 0.9rem;">
                        ⚡ 8 Categories Supported · Sub-millisecond Response
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Batch Classification Section ──
    st.divider()
    with st.expander("🗂️ Batch Document Processing (CSV File Upload)", expanded=False):
        st.markdown(
            "Classify an entire dataset at once. Upload a CSV file that contains a `text` column with your documents."
        )

        b_left, b_right = st.columns([2, 1])
        with b_left:
            batch_csv_file = st.file_uploader(
                "Upload Batch CSV",
                type=["csv"],
                key="batch_file_uploader_v2",
                label_visibility="collapsed",
            )
        with b_right:
            # Sample CSV download button for instant testing
            sample_batch_df = pd.DataFrame(
                [{"id": i + 1, "text": text} for i, text in enumerate(SAMPLE_DOCUMENTS.values())]
            )
            st.download_button(
                "⬇️ Download Sample Batch CSV",
                sample_batch_df.to_csv(index=False).encode("utf-8"),
                "sample_nlp_batch.csv",
                "text/csv",
                use_container_width=True,
                help="Download a ready-to-test CSV with 8 pre-filled documents.",
            )

        if batch_csv_file and models_dict:
            try:
                batch_df = pd.read_csv(batch_csv_file)
                if "text" not in batch_df.columns:
                    st.error("❌ The CSV file must contain a column named `text`.")
                else:
                    st.info(f"📋 Loaded **{len(batch_df):,} documents** from `{batch_csv_file.name}`")
                    if st.button("🚀 Process Batch Classification", key="execute_batch_btn"):
                        batch_model = models_dict[selected_model]
                        batch_results = []
                        prog_bar = st.progress(0, text="Classifying batch…")

                        for idx, row in batch_df.iterrows():
                            c_text = preprocessor.preprocess(str(row["text"]))
                            res = batch_model.predict_single(c_text)
                            batch_results.append(
                                {
                                    "Document Snippet": str(row["text"])[:85] + "…",
                                    "Predicted Category": res["predicted_label"],
                                    "Confidence": f"{res['confidence']*100:.1f}%",
                                    "Raw Confidence": res["confidence"],
                                }
                            )
                            prog_bar.progress((idx + 1) / len(batch_df))

                        out_df = pd.DataFrame(batch_results)
                        st.dataframe(
                            out_df[["Document Snippet", "Predicted Category", "Confidence"]],
                            use_container_width=True,
                        )

                        # Category breakdown chart
                        counts_df = out_df["Predicted Category"].value_counts().reset_index()
                        counts_df.columns = ["Category", "Count"]
                        fig_b_dist = px.bar(
                            counts_df,
                            x="Category",
                            y="Count",
                            color="Category",
                            color_discrete_map=CATEGORY_COLORS,
                            title="Batch Predictions Distribution",
                        )
                        fig_b_dist.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#f8fafc"),
                            height=280,
                        )
                        st.plotly_chart(fig_b_dist, use_container_width=True)

                        # Download results button
                        st.download_button(
                            "⬇️ Export Classified CSV",
                            out_df.to_csv(index=False).encode("utf-8"),
                            "classification_batch_results.csv",
                            "text/csv",
                            use_container_width=True,
                        )

                        # Record to history
                        top_pred = out_df["Predicted Category"].value_counts().index[0]
                        add_history_entry(
                            entry_type="batch",
                            input_source=batch_csv_file.name,
                            text_preview=f"Batch of {len(batch_df)} documents from {batch_csv_file.name}",
                            word_count=0,
                            model=selected_model,
                            predicted_label=top_pred,
                            confidence=0.0,
                            elapsed_ms=0.0,
                            probabilities={},
                            batch_count=len(batch_df),
                        )
            except Exception as e:
                st.error(f"Error processing CSV: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Model Leaderboard
# ═════════════════════════════════════════════════════════════════════════════

with tab_models:
    st.markdown('<div class="card-title">🏆 Model Benchmark Leaderboard</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="background:rgba(99,102,241,0.08); border:1px solid rgba(99,102,241,0.22); border-radius:12px; padding:0.85rem 1.2rem; margin-bottom:1.1rem; font-size:0.87rem; color:#cbd5e1;">
            🎯 <strong>Rigorous Test Evaluation:</strong> Benchmarked on a stratified held-out test split of <strong>600 unseen documents</strong> (75 per class).
            Captures real-world generalizability with natural cross-domain semantic overlap.
        </div>
        """,
        unsafe_allow_html=True,
    )

    comp_df = load_comparison_csv()
    if not comp_df.empty:
        # Styled leaderboard table
        st.dataframe(
            comp_df.style.background_gradient(
                subset=["f1_macro", "accuracy"], cmap="Purples"
            ).format({
                "accuracy": "{:.2%}",
                "f1_macro": "{:.4f}",
                "f1_weighted": "{:.4f}",
                "precision_macro": "{:.4f}",
                "recall_macro": "{:.4f}",
            }),
            use_container_width=True,
            height=200,
        )

        col_l1, col_l2 = st.columns(2)

        with col_l1:
            # Comparative metrics chart
            available_metrics = [m for m in ["f1_macro", "accuracy", "precision_macro", "recall_macro"] if m in comp_df.columns]
            melted_df = comp_df[["model"] + available_metrics].melt(
                id_vars="model", var_name="Metric", value_name="Score"
            )
            fig_bench = px.bar(
                melted_df,
                x="model",
                y="Score",
                color="Metric",
                barmode="group",
                title="Performance Across Evaluation Metrics",
                color_discrete_sequence=["#6366f1", "#10b981", "#f59e0b", "#ec4899"],
            )
            fig_bench.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f8fafc"),
                yaxis_range=[0, 1.1],
                height=350,
                xaxis=dict(tickangle=15),
            )
            st.plotly_chart(fig_bench, use_container_width=True)

        with col_l2:
            # Latency benchmark
            if "inference_ms_per_sample" in comp_df.columns:
                fig_speed = px.bar(
                    comp_df.sort_values("inference_ms_per_sample"),
                    x="model",
                    y="inference_ms_per_sample",
                    title="⚡ Inference Speed (ms / document) — Lower is Faster",
                    color="inference_ms_per_sample",
                    color_continuous_scale="Viridis",
                )
                fig_speed.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#f8fafc"),
                    height=350,
                    xaxis=dict(tickangle=15),
                    yaxis_title="Milliseconds per sample",
                )
                st.plotly_chart(fig_speed, use_container_width=True)
    else:
        st.info("📂 No benchmark data found. Run `python train.py` first.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — Activity Log
# ═════════════════════════════════════════════════════════════════════════════

with tab_history:
    init_history()
    log_history = st.session_state.activity_history

    h_head1, h_head2, h_head3 = st.columns([3, 1, 1])
    with h_head1:
        st.markdown('<div class="card-title">🕐 Session Classification Audit Log</div>', unsafe_allow_html=True)
    with h_head2:
        if log_history:
            st.download_button(
                "⬇️ Export History JSON",
                json.dumps(log_history, ensure_ascii=False, indent=2).encode("utf-8"),
                "classification_history.json",
                "application/json",
                use_container_width=True,
            )
    with h_head3:
        if log_history and st.button("🗑️ Clear Log", use_container_width=True):
            clear_history()
            st.rerun()

    if not log_history:
        st.markdown(
            """
            <div style="border: 2px dashed rgba(75, 85, 99, 0.4); border-radius: 16px; padding: 3.5rem 1rem; text-align: center; margin-top: 1rem;">
                <div style="font-size: 2.8rem; margin-bottom: 0.6rem;">📭</div>
                <div style="font-weight: 700; color: #94a3b8; font-size: 1.05rem;">No Classification History Yet</div>
                <div style="font-size: 0.85rem; color: #64748b; margin-top: 0.3rem;">
                    Run a classification in the <strong>Classify & Explain</strong> tab to start recording activity.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Metrics summary
        total_runs = len(log_history)
        single_runs = sum(1 for e in log_history if e["type"] in ("single", "file"))
        batch_runs = sum(1 for e in log_history if e["type"] == "batch")
        avg_confidence = (
            sum(e["confidence"] for e in log_history if e["type"] != "batch") / max(single_runs, 1)
        ) * 100

        s_col1, s_col2, s_col3, s_col4 = st.columns(4)
        with s_col1:
            st.metric("Total Inferences", total_runs)
        with s_col2:
            st.metric("Single Inferences", single_runs)
        with s_col3:
            st.metric("Batch Inferences", batch_runs)
        with s_col4:
            st.metric("Average Confidence", f"{avg_confidence:.1f}%")

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        for entry in log_history[:25]:
            cat_c = CATEGORY_COLORS.get(entry["predicted_label"], "#818cf8")
            cat_e = CATEGORY_EMOJIS.get(entry["predicted_label"], "📌")
            ts = entry.get("timestamp", "")
            try:
                ts_fmt = datetime.fromisoformat(ts).strftime("%d %b %Y · %H:%M:%S UTC")
            except Exception:
                ts_fmt = ts

            st.markdown(
                f"""
                <div class="hist-card" style="border-left-color: {cat_c};">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <span style="background:{cat_c}22; color:{cat_c}; border:1px solid {cat_c}55; padding:0.25rem 0.75rem; border-radius:12px; font-weight:700; font-size:0.82rem; text-transform:uppercase;">
                                {cat_e} {entry['predicted_label']}
                            </span>
                            &nbsp;
                            <span style="font-size:0.75rem; color:#94a3b8; background:rgba(255,255,255,0.05); padding:0.2rem 0.5rem; border-radius:6px;">
                                {entry['type'].upper()}
                            </span>
                        </div>
                        <div style="font-size:0.75rem; color:#64748b;">
                            🕐 {ts_fmt}
                        </div>
                    </div>
                    <div style="margin: 0.6rem 0 0.4rem; font-size:0.86rem; color:#cbd5e1; line-height:1.5;">
                        {entry['text_preview']}…
                    </div>
                    <div style="font-size:0.78rem; color:#94a3b8; display:flex; gap:1.2rem; flex-wrap:wrap;">
                        <span>🤖 Model: <strong>{entry['model'].replace('_', ' ').title()}</strong></span>
                        <span>🎯 Confidence: <strong>{entry['confidence']*100:.1f}%</strong></span>
                        <span>⚡ Latency: <strong>{entry.get('elapsed_ms', 0):.1f} ms</strong></span>
                        <span>📄 Source: <strong>{entry.get('input_source', 'text')}</strong></span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
