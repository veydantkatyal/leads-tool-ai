import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
st.set_page_config(page_title="SaaSquatch-AI-Leads", layout="wide")

@st.cache_resource
def load_artifacts():
    lead_model = joblib.load("models/lead_xgb_model.pkl")
    lead_ohe = joblib.load("models/lead_ohe.pkl")
    lead_tfidf = joblib.load("models/lead_tfidf.pkl")
    intent_model = joblib.load("models/intent_model.pkl")
    intent_tfidf = joblib.load("models/intent_tfidf.pkl")
    return lead_model, lead_ohe, lead_tfidf, intent_model, intent_tfidf

lead_model, lead_ohe, lead_tfidf, intent_model, intent_tfidf = load_artifacts()

st.title("🦄 SaaSquatch-AI-Leads")

tabs = st.tabs([
    "🔮 Lead Scoring",
    "🧬 Company Recommender",
    "🧩 Market Segmentation",
    "🚨 Intent Prediction"
])

# --- Tab 1: Lead Scoring ---
with tabs[0]:
    st.header("🔮 Lead Scoring")
    uploaded = st.file_uploader("Upload a CSV file (preprocessed or raw Crunchbase format)", type=["csv"], key="leadscoring")
    if uploaded:
        df = pd.read_csv(uploaded, dtype=str, low_memory=False)
    else:
        st.info("Using sample data from data/lead_scoring.csv")
        df = pd.read_csv("data/lead_scoring.csv", dtype=str, low_memory=False)
    if len(df) > 2000:
        df = df.sample(2000, random_state=42)
    cat_features = ['market', 'category_list', 'status', 'country_code', 'state_code', 'region', 'city']
    num_features = [col for col in df.columns if col not in cat_features and not col.startswith("tfidf_") and col != 'converted']
    for col in cat_features:
        if col not in df.columns:
            df[col] = ""
    for col in num_features:
        if col not in df.columns:
            df[col] = 0
    tfidf_features = [col for col in df.columns if col.startswith("tfidf_")]
    if tfidf_features:
        X_tfidf = df[tfidf_features].values
    else:
        if 'text_features' not in df.columns:
            df['text_features'] = df['category_list'].fillna('') + ' ' + df['market'].fillna('')
        X_tfidf = lead_tfidf.transform(df["text_features"].astype(str)).toarray()
    X_cat = lead_ohe.transform(df[cat_features].astype(str))
    X_num = df[num_features].astype(float).values
    X_all = np.hstack([X_cat, X_num, X_tfidf])
    probs = lead_model.predict_proba(X_all)[:, 1]
    lead_scores = (probs * 100).round(2)
    df["Lead Score (0-100)"] = lead_scores
    st.dataframe(df[cat_features + num_features + ["Lead Score (0-100)"]].sort_values("Lead Score (0-100)", ascending=False).reset_index(drop=True))
    st.download_button("Download Scored Leads as CSV", df.to_csv(index=False), file_name="scored_leads.csv")

# --- Tab 2: Company Recommender ---
with tabs[1]:
    st.header("🧬 Company Similarity Recommender")
    df = pd.read_csv("data/leads_recommender.csv", dtype=str, low_memory=False)
    # Sample 2000 rows for memory efficiency
    if len(df) > 2000:
        df = df.sample(2000, random_state=42)
    if 'text_features' not in df.columns:
        df['text_features'] = df['name'].fillna('') + ' ' + df['market'].fillna('') + ' ' + df['category_list'].fillna('')
    tfidf_features = [col for col in df.columns if col.startswith("tfidf_")]
    if tfidf_features:
        tfidf_matrix = df[tfidf_features].values.astype(float)
    else:
        tfidf_matrix = lead_tfidf.transform(df['text_features'].astype(str)).toarray()
    company_list = df['name'].tolist()
    selected_company = st.selectbox("Select a company", company_list)
    idx = company_list.index(selected_company)
    sims = cosine_similarity([tfidf_matrix[idx]], tfidf_matrix)[0]
    top_indices = np.argsort(sims)[::-1][1:6]
    similar_companies = df.iloc[top_indices]
    st.write(f"**Selected company:** {selected_company}")
    st.write("**Top 5 most similar companies:**")
    display_cols = [col for col in ['name', 'market', 'city', 'country_code'] if col in df.columns]
    st.dataframe(similar_companies[display_cols].reset_index(drop=True))

# --- Tab 3: Market Segmentation ---
with tabs[2]:
    st.header("🧩 Market Segmentation (Clustering)")
    df = pd.read_csv("data/leads_recommender.csv", dtype=str, low_memory=False)
    # Sample 2000 rows for memory efficiency
    if len(df) > 2000:
        df = df.sample(2000, random_state=42)
    if 'text_features' not in df.columns:
        df['text_features'] = df['name'].fillna('') + ' ' + df['market'].fillna('') + ' ' + df['category_list'].fillna('')
    tfidf_features = [col for col in df.columns if col.startswith("tfidf_")]
    if tfidf_features:
        tfidf_matrix = df[tfidf_features].values.astype(float)
    else:
        tfidf_matrix = lead_tfidf.transform(df['text_features'].astype(str)).toarray()
    n_clusters = st.slider("Number of clusters", 2, 15, 10)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(tfidf_matrix)
    df['cluster'] = clusters
    st.dataframe(df[['cluster', 'name', 'market']].sort_values('cluster').reset_index(drop=True))
    # PCA plot
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(tfidf_matrix)
    st.write("### Cluster Visualization (PCA)")
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(8,6))
    scatter = ax.scatter(X_pca[:,0], X_pca[:,1], c=clusters, cmap='tab10', alpha=0.6)
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_title('Company Clusters (PCA)')
    plt.colorbar(scatter, ax=ax, label='Cluster')
    st.pyplot(fig)

# --- Tab 4: Intent Prediction ---
with tabs[3]:
    st.header("🚨 Intent Signal Prediction")
    input_text = st.text_area("Enter company description, market, or keywords:")
    if st.button("Predict Intent"):
        X_input = intent_tfidf.transform([input_text])
        intent_pred = intent_model.predict(X_input)[0]
        intent_proba = intent_model.predict_proba(X_input)[0,1]
        st.write(f"**Intent Prediction:** {'Intent' if intent_pred else 'No Intent'} (probability: {intent_proba:.2f})")
