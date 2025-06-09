from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import pandas as pd
import numpy as np

class PatientClustering:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.kmeans_model = None
        self.gmm_model = None

    def fit_kmeans(self, data):
        self.kmeans_model = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.kmeans_model.fit(data)
        return self.kmeans_model.labels_

    def fit_gmm(self, data):
        self.gmm_model = GaussianMixture(n_components=self.n_clusters, random_state=42)
        self.gmm_model.fit(data)
        return self.gmm_model.predict(data)

    def reduce_dimensions(self, data, n_components=2):
        pca = PCA(n_components=n_components)
        return pca.fit_transform(data)

    def get_cluster_centers(self):
        if self.kmeans_model:
            return self.kmeans_model.cluster_centers_
        else:
            raise ValueError("KMeans model has not been fitted yet.")

    def get_gmm_means(self):
        if self.gmm_model:
            return self.gmm_model.means_
        else:
            raise ValueError("GMM model has not been fitted yet.")