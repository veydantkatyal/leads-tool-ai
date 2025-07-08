import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

class MarketSegmentation:
    def __init__(self, data_csv, tfidf_path):
        self.df = pd.read_csv(data_csv, dtype=str, low_memory=False)
        self.tfidf = joblib.load(tfidf_path)
        self._prepare_matrix()

    def _prepare_matrix(self):
        if any(col.startswith('tfidf_') for col in self.df.columns):
            tfidf_features = [col for col in self.df.columns if col.startswith('tfidf_')]
            tfidf_matrix = self.df[tfidf_features].values
        else:
            tfidf_matrix = self.tfidf.transform(self.df['text_features'].astype(str)).toarray()
        imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
        tfidf_matrix_imputed = imputer.fit_transform(tfidf_matrix)
        self.tfidf_matrix = np.asarray(tfidf_matrix_imputed, dtype=np.float64)

    def cluster(self, n_clusters=10):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.df['cluster'] = kmeans.fit_predict(self.tfidf_matrix)
        self.kmeans = kmeans
        return self.df[['cluster', 'name', 'market']]

    def plot_pca(self):
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.tfidf_matrix)
        plt.figure(figsize=(8,6))
        scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=self.df['cluster'].astype(int), cmap='tab10', alpha=0.6)
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.title('Company Clusters (PCA)')
        plt.colorbar(scatter, label='Cluster')
        plt.show()

if __name__ == '__main__':
    data_csv = 'data/leads_recommender.csv'
    tfidf_path = 'models/lead_tfidf.pkl'
    segmenter = MarketSegmentation(data_csv, tfidf_path)
    print(segmenter.cluster(n_clusters=10).head())
    segmenter.plot_pca() 