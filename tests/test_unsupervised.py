import pytest
from src.unsupervised import KMeansClustering, PCAAnalysis

def test_kmeans_clustering():
    # Sample data for testing
    sample_data = [[1, 2], [1, 4], [1, 0],
                   [4, 2], [4, 4], [4, 0]]
    
    # Initialize KMeansClustering
    kmeans = KMeansClustering(n_clusters=2)
    kmeans.fit(sample_data)
    
    # Test if the number of clusters is correct
    assert len(kmeans.labels_) == len(sample_data)
    assert len(set(kmeans.labels_)) == 2

def test_pca_analysis():
    # Sample data for testing
    sample_data = [[2.5, 2.4], [0.5, 0.7], [2.2, 2.9],
                   [1.9, 2.2], [3.1, 3.0], [2.3, 2.7],
                   [3.0, 3.3], [3.0, 3.0], [3.0, 3.5]]
    
    # Initialize PCAAnalysis
    pca = PCAAnalysis(n_components=2)
    transformed_data = pca.fit_transform(sample_data)
    
    # Test if the transformed data has the correct shape
    assert transformed_data.shape[1] == 2
    assert transformed_data.shape[0] == len(sample_data)