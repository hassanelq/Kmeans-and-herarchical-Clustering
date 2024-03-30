import numpy as np

def euclidean_distance(data_point, centroids):
    return np.sqrt(np.sum((data_point - centroids) ** 2, axis=1))

def manhattan_distance(data_point, centroids):
    return np.sum(np.abs(data_point - centroids), axis=1)

def cosine_distance(data_point, centroids):
    dot_product = np.dot(data_point, centroids.T)
    norm_product = np.linalg.norm(data_point) * np.linalg.norm(centroids, axis=1)
    norm_product = np.where(norm_product == 0, 1, norm_product)
    cosine_similarity = dot_product / norm_product
    cosine_distance = 1 - cosine_similarity
    return cosine_distance