# SaaSquatch-AI-Leads

SaaSquatch-AI-Leads is a modular, production-ready, AI-powered lead generation and analysis tool built with Streamlit. It leverages Crunchbase startup data and advanced machine learning to help SaaS businesses and sales teams discover, score, and segment high-potential leads.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Customization](#customization)
- [Model Training & Updating](#model-training--updating)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features

### 1. AI-Powered Lead Scoring
- Predicts the likelihood of a company converting based on features such as market, funding, and employee count.
- Uses a LightGBM/XGBoost model trained on a synthetic target (`converted = funding_total_usd > 1,000,000`).
- Outputs a lead score (0–100) for each company and displays ranked results.

### 2. Company Similarity Recommender
- Recommends the top 5 most similar companies for any selected company.
- Uses TF-IDF or Sentence Transformers on company descriptions and categories.
- Computes cosine similarity for recommendations.

### 3. Intent Signal Prediction
- Detects buying signals in company descriptions using a synthetic label (intent = 1 if description contains keywords like "hiring", "launch", "AI", "funding").
- Trains a BERT-based or TF-IDF + Logistic Regression model to classify intent.

### 4. Smart Clustering for Segmentation
- Clusters leads into segments using KMeans on TF-IDF features.
- Displays cluster label, company name, industry, and visualizes clusters with PCA.

## Project Structure

```
leads-tool-ai/
│
├── app.py                  # Streamlit app entry point
├── requirements.txt        # Python dependencies
├── data/                   # Input and processed datasets
│   ├── leads_cleaned.csv
│   ├── lead_scoring.csv
│   ├── leads_recommender.csv
│   ├── leads_intent.csv
│   └── ...
├── models/                 # Trained models and vectorizers (.pkl)
│   ├── lead_xgb_model.pkl
│   ├── lead_ohe.pkl
│   ├── lead_tfidf.pkl
│   ├── intent_model.pkl
│   ├── intent_tfidf.pkl
│   └── ...
├── notebooks/              # Jupyter notebooks for each feature
│   ├── 1_lead_scoring.ipynb
│   ├── 2_similarity_recommender.ipynb
│   ├── 3_leads_market_segmentation.ipynb
│   ├── 4_leads_intent_signals.ipynb
│   └── ...
├── src/                    # Modular Python scripts for each feature
│   ├── lead_scoring.py
│   ├── similarity_recommender.py
│   ├── market_segmentation.py
│   └── intent_signal.py
└── ...
```

## Setup & Installation

1. **Clone the repository**
   ```
   git clone https://github.com/veydantkatyal/leads-tool-ai.git
   cd leads-tool-ai
   ```

2. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

3. **Prepare data and models**
   - Place your Crunchbase or compatible dataset in the `data/` directory.
   - Run the notebooks in `notebooks/` to preprocess data and train models, or use the provided scripts in `src/` to retrain models as needed.
   - Ensure all `.pkl` model and vectorizer files are in the `models/` directory.

4. **Run the Streamlit app**
   ```
   streamlit run app.py
   ```

## Usage

- **Lead Scoring:** Upload a CSV or use the sample data to score and rank leads.
- **Company Recommender:** Select a company to view the most similar companies.
- **Market Segmentation:** Cluster companies and visualize segments.
- **Intent Prediction:** Enter a company description or keywords to detect buying intent.

## Customization

- To adjust model parameters, feature engineering, or intent keywords, edit the relevant scripts in `src/` or the corresponding notebooks.
- To use a different dataset, ensure it matches the expected schema or adapt the preprocessing steps.

## Model Training & Updating

- All model training and feature engineering steps are available as Jupyter notebooks in the `notebooks/` directory.
- For production, use the modular scripts in `src/` to retrain and save models.
- After retraining, ensure the new `.pkl` files are placed in the `models/` directory and committed to the repository.

## Troubleshooting

- **Memory Errors:** If you encounter memory errors, reduce the number of rows or TF-IDF features, or sample a subset of your data for testing.
- **Model Compatibility:** Ensure that the scikit-learn and numpy versions used for training and deployment match to avoid pickle loading errors.
- **Missing Files:** Make sure all required data and model files are present in the correct directories.

## License

This project is licensed under the MIT License.

## Acknowledgments

- Crunchbase for the startup dataset
- scikit-learn, XGBoost, and Streamlit for the core technology stack 