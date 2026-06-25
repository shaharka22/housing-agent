# HouseIQ – AI Real Estate Agent

An AI-powered housing market analysis agent that combines machine learning with Claude AI to provide data-driven real estate recommendations.

## Features

- **K-Means Clustering** – Segments the market into 5 distinct property categories
- **Isolation Forest** – Detects anomalous listings (unusually priced properties)
- **Cosine Similarity** – Finds comparable properties
- **Claude AI Agent** – Natural language analysis and recommendations
- **Flask Web App** – Clean, interactive UI deployable on Render

## Local Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
python app.py
```

Then open: http://localhost:5000

## Deploy to Render

1. Push this folder to a GitHub repository
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your GitHub repo
4. Set environment variable: `ANTHROPIC_API_KEY = your_key`
5. Build command: `pip install -r requirements.txt`
6. Start command: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`

## Project Structure

```
housing-agent/
├── app.py              # Flask server + routes
├── ml_models.py        # K-Means, Isolation Forest, Similarity
├── agent.py            # Claude AI agent logic
├── housing_data.csv    # Dataset (2000 listings)
├── requirements.txt    # Python dependencies
├── render.yaml         # Render deployment config
└── templates/
    └── index.html      # Frontend UI
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Web Framework | Flask + Gunicorn |
| ML Models | scikit-learn (KMeans, IsolationForest) |
| NLP/Agent | Claude claude-sonnet-4-6 via Anthropic API |
| Data Processing | pandas, numpy |
| Deployment | Render.com |
| Frontend | Vanilla HTML/CSS/JS |

## Dataset

Synthetic housing dataset with 2,000 listings across 10 US cities, featuring:
- Price, bedrooms, rooms, sqft
- City, neighborhood, property type
- Property descriptions (for NLP analysis)
- Geographic coordinates

## AI Agent Flow

1. User enters a natural language query + optional filters
2. System retrieves matching properties + cluster/anomaly data
3. Full context is sent to Claude claude-sonnet-4-6
4. Agent returns a structured, data-driven recommendation
5. Results displayed alongside interactive visualizations
