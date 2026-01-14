# Insight Extractor

An NLP pipeline that automatically extracts key insights from YouTube subtitles.

## Features

- **Subtitle Collection**: Automatically download English subtitles from YouTube playlists
- **Text Preprocessing**: NLTK/LLM-based sentence segmentation and cleaning
- **Semantic Clustering**: Group sentences using UMAP + HDBSCAN
- **Theme Analysis**: Extract cluster themes using LLM
- **Visualization**: Interactive UMAP-based visualizations

## Tech Stack

| Category | Technologies |
|----------|-------------|
| Embedding | Sentence-Transformers, spaCy |
| Clustering | HDBSCAN, UMAP |
| LLM | Transformers (Gemma, LLaMA) |
| Visualization | Matplotlib, Plotly |
| API | FastAPI |
| Experiment Tracking | MLflow, Hydra |

## Installation

```bash
# Clone repository
git clone https://github.com/baek0203/insight_extractor.git
cd insight_extractor

# Set up virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

## How to Start

### 1. Collect Subtitles

```bash
python -m src.collector.youtube "https://youtube.com/playlist?list=..." --domain interview
```

### 2. Run Clustering

```bash
python -m src.mining.clustering \
    --input data/split/interview \
    --output outputs/interview \
    --domain interview
```

## Project Structure

```
insight_extractor/
├── src/
│   ├── collector/        # YouTube subtitle collection
│   │   └── youtube.py
│   ├── preprocessor/     # Text preprocessing
│   │   ├── nltk_splitter.py
│   │   └── llm_splitter.py
│   ├── mining/           # Clustering & analysis
│   │   ├── clustering.py
│   │   └── summarizer.py
│   ├── extractor/        # Video extraction utils
│   ├── api/              # FastAPI server
│   └── utils/            # Common utilities
│       ├── text.py       # Text processing
│       ├── time.py       # Timestamp handling
│       └── gpu.py        # GPU management
├── configs/
│   └── config.yaml       # Configuration file
├── data/                 # Data (What you download in youtube)
├── outputs/              # Output results
└── docs/
```

## Configuration

Manage global settings in `configs/config.yaml`:

## License

MIT License

## Contact

if you have any question of my project. please contact me through email "sungback6475@gmail.com".

thx!