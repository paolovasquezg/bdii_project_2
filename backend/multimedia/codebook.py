import numpy as np
import pickle
from backend.multimedia.extractors import extract_features




class KMeansManual:
    def __init__(self, n_clusters=16, max_iter=20):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None

    def fit(self, X):
        idx = np.random.choice(len(X), self.n_clusters, replace=False)
        self.centroids = X[idx].astype(float)

        for _ in range(self.max_iter):
            labels = self._assign_clusters(X)
            new_centroids = self._recompute_centroids(X, labels)

            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

    def _assign_clusters(self, X):
        distances = np.linalg.norm(X[:, None, :] - self.centroids[None, :, :], axis=2)
        return np.argmin(distances, axis=1)

    def _recompute_centroids(self, X, labels):
        new_centroids = np.zeros_like(self.centroids)
        for c in range(self.n_clusters):
            points = X[labels == c]
            if len(points) > 0:
                new_centroids[c] = points.mean(axis=0)
            else:
                new_centroids[c] = self.centroids[c]
        return new_centroids

    def predict(self, X):
        distances = np.linalg.norm(X[:, None, :] - self.centroids[None, :, :], axis=2)
        return np.argmin(distances, axis=1)




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




def build_codebook(all_descriptors, k=16):
    kmeans = KMeansManual(n_clusters=k, max_iter=20)
    kmeans.fit(all_descriptors)
    codebook = kmeans.centroids
    return codebook, kmeans




def compute_histogram(descriptors, kmeans, k):
    if descriptors.shape[0] == 0:
        return np.zeros(k)
    labels = kmeans.predict(descriptors)
    hist, _ = np.histogram(labels, bins=np.arange(k+1))
    return hist.astype(float)




def compute_df(all_hists):
    k = len(next(iter(all_hists.values())))
    df = np.zeros(k)
    for hist in all_hists.values():
        df += (hist > 0).astype(int)
    return df


def compute_idf(df, N):
    idf = np.log((N + 1) / (df + 1))
    return idf


def histogram_to_tfidf(hist, idf):
    if hist.sum() == 0:
        return np.zeros_like(hist)
    tf = hist / hist.sum()
    tfidf = tf * idf
    return tfidf




def build_inverted_index(tfidf_hists):
    inverted = {}
    for img, vec in tfidf_hists.items():
        for cluster_id, weight in enumerate(vec):
            if weight > 0:
                if cluster_id not in inverted:
                    inverted[cluster_id] = []
                inverted[cluster_id].append((img, weight))
    return inverted



def save_binary(obj, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)


def load_binary(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)




def prepare_and_save_all(image_paths, kmeans, k, save_dir="multimedia_bin"):
    all_hists = {}
    for path in image_paths:
        desc = extract_features(path)
        hist = compute_histogram(desc, kmeans, k)
        all_hists[path] = hist
    df = compute_df(all_hists)
    idf = compute_idf(df, N=len(all_hists))
    tfidf_hists = {img: histogram_to_tfidf(h, idf) for img, h in all_hists.items()}
    inverted = build_inverted_index(tfidf_hists)

    import os
    os.makedirs(save_dir, exist_ok=True)

    save_binary(tfidf_hists, f"{save_dir}/tfidf_hists.bin")
    save_binary(df, f"{save_dir}/df.bin")
    save_binary(idf, f"{save_dir}/idf.bin")
    save_binary(inverted, f"{save_dir}/inverted_index.bin")

    print("✔ Datos guardados en:", save_dir)
