# NLP Document Classification System

> A production-ready, end-to-end NLP pipeline for multi-class document classification with traditional ML and transformer-based deep learning models.

---

## 🏗️ Architecture

```
NLP Project/
├── src/
│   ├── data/
│   │   ├── generator.py      # Synthetic data generator (8 categories × 500 docs)
│   │   ├── loader.py         # Flexible data loader (CSV/JSON/JSONL/TXT)
│   │   └── preprocessor.py   # Text cleaning, tokenization, lemmatization
│   ├── models/
│   │   ├── traditional.py    # TF-IDF + LR / SVM / NB / RF / GB classifiers
│   │   ├── transformer.py    # DistilBERT fine-tuning classifier
│   │   └── evaluator.py      # Metrics, confusion matrix, model comparison
│   ├── utils/
│   │   ├── visualizer.py     # Matplotlib/Seaborn plots & word clouds
│   │   └── config.py         # YAML config loader
│   ├── api/
│   │   └── app.py            # FastAPI REST API (single + batch inference)
│   └── ui/
│       └── app.py            # Streamlit interactive dashboard
├── configs/
│   └── config.yaml           # Master system configuration
├── data/
│   └── sample/               # Generated train/val/test splits
├── artifacts/
│   ├── models/               # Saved model files
│   └── plots/                # Generated visualizations
├── tests/
│   ├── test_preprocessor.py  # 15 unit tests for preprocessing
│   └── test_classifier.py    # 12 unit tests for classifiers & evaluator
├── train.py                  # Main training pipeline
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train All Models

```bash
python train.py
```

This will:
- Generate 4,000 synthetic documents (500 per category × 8 categories)
- Preprocess text (clean, tokenize, lemmatize)
- Train 4 models: Logistic Regression, SVM, Naive Bayes, Random Forest
- Generate evaluation plots and reports in `artifacts/`

### 3. Start the REST API

```bash
python -m uvicorn src.api.app:app --reload --port 8000
```

API docs: http://localhost:8000/docs

### 4. Launch the UI Dashboard

```bash
streamlit run src/ui/app.py
```

Dashboard: http://localhost:8501

### 5. Run Tests

```bash
pytest tests/ -v --tb=short
```

---

## 🐳 Docker

```bash
# Train models
docker compose --profile train up trainer

# Start API + UI
docker compose up api ui
```

---

## 📊 Classification Categories

| Category | Emoji | Examples |
|---|---|---|
| Technology | 💻 | AI, software, hardware, cybersecurity |
| Sports | ⚽ | Football, basketball, tennis, Olympics |
| Politics | 🏛️ | Elections, legislation, diplomacy |
| Entertainment | 🎬 | Movies, music, TV, celebrities |
| Science | 🔬 | Research, space, climate, biology |
| Health | 🏥 | Medicine, disease, wellness, drugs |
| Finance | 📈 | Stocks, crypto, earnings, banking |
| World | 🌍 | Disasters, geopolitics, conflicts |

---

## 🤖 Models

| Model | Type | Strengths |
|---|---|---|
| **Logistic Regression** | TF-IDF + LR | Fast, interpretable, great baseline |
| **SVM** | TF-IDF + LinearSVC | High accuracy, robust to high dims |
| **Naive Bayes** | TF-IDF + MNB | Fastest inference, probabilistic |
| **Random Forest** | TF-IDF + RF | Non-linear, feature importance |
| **DistilBERT** | Transformer | Highest accuracy, context-aware |

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health + loaded models |
| `GET` | `/models` | List available models |
| `GET` | `/categories` | List classification categories |
| `POST` | `/classify` | Classify a single document |
| `POST` | `/classify/batch` | Classify up to 100 documents |

### Example Request

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Scientists at NASA discovered evidence of water on Mars using the James Webb telescope.",
    "model": "logistic_regression",
    "top_k": 3
  }'
```

### Example Response

```json
{
  "predicted_label": "science",
  "confidence": 0.8912,
  "top_predictions": [
    {"label": "science", "confidence": 0.8912},
    {"label": "technology", "confidence": 0.0631},
    {"label": "world", "confidence": 0.0214}
  ],
  "model_used": "logistic_regression",
  "inference_time_ms": 2.3
}
```

---

## ⚙️ Configuration

Edit [`configs/config.yaml`](configs/config.yaml) to customize:

- **Categories** — Add/remove classification labels
- **Preprocessing** — Enable/disable stopword removal, lemmatization, etc.
- **TF-IDF** — Adjust `max_features`, `ngram_range`
- **Model hyperparameters** — Tune C, max_iter, n_estimators
- **Deep Learning** — Switch transformer model, epochs, learning rate

---

## 🔬 Using Your Own Data

Drop a CSV file with `text` and `label` columns into `data/raw/` and update `config.yaml`:

```yaml
data:
  raw_dir: "data/raw"
  # ... your dataset will be auto-loaded
```

Or upload directly through the UI's batch classification tab.
