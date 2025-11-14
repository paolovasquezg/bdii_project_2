import numpy as np
from numpy.linalg import norm
from backend.multimedia.extractors import extract_features
from backend.multimedia.codebook import compute_histogram


def cosine_similarity(a, b):
    if norm(a) == 0 or norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (norm(a) * norm(b)))


def compute_query_histogram(path, kmeans, k, method="sift"):
    desc = extract_features(path, method=method)
    hist = compute_histogram(desc, kmeans, k)
    return hist


def knn_secuencial(query_hist, all_hists, top_k=5):
    resultados = []

    for img_id, hist in all_hists.items():
        sim = cosine_similarity(query_hist, hist)
        resultados.append((img_id, sim))

    resultados.sort(key=lambda x: x[1], reverse=True)

    return resultados[:top_k]
