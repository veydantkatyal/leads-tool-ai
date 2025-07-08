import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

class SimilarityRecommender:
    def __init__(self, data_csv, tfidf_path):
        self.df = pd.read_csv(data_csv, dtype=str, low_memory=False)
        if 'name' in self.df.columns:
            self.df['name'] = self.df['name'].str.replace('/organization/', '', regex=False)
        self.tfidf = joblib.load(tfidf_path)
        self._prepare_matrix()

    def _prepare_matrix(self):
        if any(col.startswith('tfidf_') for col in self.df.columns):
            tfidf_features = [col for col in self.df.columns if col.startswith('tfidf_')]
            self.tfidf_matrix = self.df[tfidf_features].values.astype(float)
        else:
            self.tfidf_matrix = self.tfidf.transform(self.df['text_features'].astype(str)).toarray()

    def get_similar(self, company, top_n=5):
        if isinstance(company, int):
            idx = company
        else:
            if 'name' not in self.df.columns:
                raise ValueError("No 'name' column in dataframe.")
            idx = self.df[self.df['name'] == company].index[0]
        sims = cosine_similarity([self.tfidf_matrix[idx]], self.tfidf_matrix)[0]
        top_indices = np.argsort(sims)[::-1][1:top_n+1]
        similar_companies = self.df.iloc[top_indices].copy()
        display_cols = [col for col in ['name', 'market', 'city', 'country_code'] if col in self.df.columns]
        return similar_companies[display_cols].reset_index(drop=True)

if __name__ == '__main__':
    data_csv = 'data/leads_recommender.csv' 
    tfidf_path = 'models/lead_tfidf.pkl'
    recommender = SimilarityRecommender(data_csv, tfidf_path)
    company_name = recommender.df.iloc[5]['name']
    print(f"Selected company: {company_name}")
    print("Top 5 most similar companies:")
    print(recommender.get_similar(company_name, top_n=5).to_string(index=False)) 