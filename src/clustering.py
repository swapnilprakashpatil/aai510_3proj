import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from kneed import KneeLocator
from joblib import Parallel, delayed
from src.config import RANDOM_STATE, N_JOBS

class ClusteringAnalysis:
    def __init__(self, df, numeric_features=None):
        self.df = df
        if numeric_features is None:
            self.numeric_df = df.select_dtypes(include=[np.number]).copy()
        else:
            self.numeric_df = df[numeric_features].copy()
        self.scaler = StandardScaler()
        self.pca = None
        self.X_scaled = None
        self.X_pca = None
        self.pca_df = None
        self.top_n = None
        self.X_reduced = None
        self.kmeans = None
        self.gmm = None
        self.kmeans_labels = None
        self.gmm_labels = None
        self.kmeans_scores = None
        self.gmm_scores = None

    def add_emotional_distress(self, cols=['stress', 'fear', 'anxiety']):
        self.numeric_df['Emotional_Distress'] = self.numeric_df[cols].mean(axis=1)

    def standardize(self):
        self.X_scaled = self.scaler.fit_transform(self.numeric_df)
        return self.X_scaled

    def run_pca(self, n_components=None):
        self.pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
        self.X_pca = self.pca.fit_transform(self.X_scaled)
        self.pca_df = pd.DataFrame(self.X_pca, columns=[f'PC{i+1}' for i in range(self.X_pca.shape[1])])
        return self.pca_df

    def explained_variance(self):
        explained_variance = self.pca.explained_variance_ratio_
        explained_df = pd.DataFrame({
            'Principal Component': [f'PC{i+1}' for i in range(len(explained_variance))],
            'Explained Variance Ratio': explained_variance,
            'Cumulative Variance': np.cumsum(explained_variance)
        })
        return explained_df

    def select_top_n_components(self, target_variance=0.90):
        cumulative_variance = np.cumsum(self.pca.explained_variance_ratio_)
        self.top_n = np.argmax(cumulative_variance >= target_variance) + 1
        self.X_reduced = self.pca_df.iloc[:, :self.top_n]
        return self.top_n, self.X_reduced

    def get_loadings(self):
        loadings = pd.DataFrame(self.pca.components_.T,
                               columns=[f'PC{i+1}' for i in range(self.pca.n_components_)],
                               index=self.numeric_df.columns)
        return loadings

    def get_feature_contributions(self):
        feature_contribution = pd.DataFrame(
            self.pca.components_,
            columns=[f'PC{i+1}' for i in range(self.pca.components_.shape[0])],
            index=self.numeric_df.columns
        )
        return feature_contribution
    
    def elbow_method(self, k_range=range(1, 15)):
        wcss = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init='auto')
            kmeans.fit(self.X_reduced)
            wcss.append(kmeans.inertia_)
        knee_locator = KneeLocator(list(k_range), wcss, curve="convex", direction="decreasing")
        optimal_k = knee_locator.knee
        return wcss, optimal_k


    def evaluate_clustering_performance(self, k_values):
        X = self.X_reduced.astype(np.float32) if hasattr(self.X_reduced, 'astype') else self.X_reduced
        def compute_metrics(k):
            print(f"Evaluating clustering performance for k={k}")
            kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init='auto')
            kmeans_labels = kmeans.fit_predict(X)
            gmm = GaussianMixture(n_components=k, random_state=RANDOM_STATE)
            gmm_labels = gmm.fit_predict(X)
            return {
                'k': k,
                'kmeans_labels': kmeans_labels,
                'gmm_labels': gmm_labels,
                'kmeans_silhouette': silhouette_score(X, kmeans_labels),
                'kmeans_davies': davies_bouldin_score(X, kmeans_labels),
                'kmeans_calinski': calinski_harabasz_score(X, kmeans_labels),
                'gmm_silhouette': silhouette_score(X, gmm_labels),
                'gmm_davies': davies_bouldin_score(X, gmm_labels),
                'gmm_calinski': calinski_harabasz_score(X, gmm_labels)
            }
        results = Parallel(n_jobs=-1, backend='loky')(delayed(compute_metrics)(k) for k in k_values)
        # Aggregate results
        kmeans_silhouette_scores = [r['kmeans_silhouette'] for r in results]
        kmeans_davies_bouldin_scores = [r['kmeans_davies'] for r in results]
        kmeans_calinski_harabasz_scores = [r['kmeans_calinski'] for r in results]
        gmm_silhouette_scores = [r['gmm_silhouette'] for r in results]
        gmm_davies_bouldin_scores = [r['gmm_davies'] for r in results]
        gmm_calinski_harabasz_scores = [r['gmm_calinski'] for r in results]
        kmeans_labels_list = [r['kmeans_labels'] for r in results]
        gmm_labels_list = [r['gmm_labels'] for r in results]
        self.kmeans_scores = pd.DataFrame({
            'k': k_values,
            'Silhouette Score': kmeans_silhouette_scores,
            'Davies-Bouldin Score': kmeans_davies_bouldin_scores,
            'Calinski-Harabasz Score': kmeans_calinski_harabasz_scores
        })
        self.gmm_scores = pd.DataFrame({
            'k': k_values,
            'Silhouette Score': gmm_silhouette_scores,
            'Davies-Bouldin Score': gmm_davies_bouldin_scores,
            'Calinski-Harabasz Score': gmm_calinski_harabasz_scores
        })
        return self.kmeans_scores, self.gmm_scores, kmeans_labels_list, gmm_labels_list
