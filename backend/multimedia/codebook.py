import numpy as np
from sklearn.cluster import KMeans
from backend.multimedia.extractors import extract_features

def collect_descriptors(image_paths, method="sift"):
    all_desc = []
    per_image_desc = []

    for path in image_paths:
        desc = extract_features(path, method=method)

        per_image_desc.append(desc)

        if desc.shape[0] > 0:
            all_desc.append(desc)

    if len(all_desc) == 0:
        raise RuntimeError("No se encontraron descriptores en ninguna imagen.")

    all_desc = np.vstack(all_desc)
    return per_image_desc, all_desc


def build_codebook(all_descriptors, k=64):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(all_descriptors)

    codebook = kmeans.cluster_centers_
    return codebook, kmeans


def compute_histogram(descriptors, kmeans, k):
    if descriptors.shape[0] == 0:
        return np.zeros(k)

    labels = kmeans.predict(descriptors)
    hist, _ = np.histogram(labels, bins=np.arange(k+1))
    return hist.astype(float)
