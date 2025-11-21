import numpy as np
import pickle
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
