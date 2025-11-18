import glob
from backend.multimedia.extractors import extract_features
from backend.multimedia.codebook import collect_descriptors, build_codebook, compute_histogram
from backend.multimedia.knn import compute_query_histogram, knn_secuencial

image_paths = glob.glob("images/*.png")


print(f"Encontradas {len(image_paths)} imágenes.")

per_image_desc, all_desc = collect_descriptors(image_paths, method="sift")
print("Descriptores extraídos para todas las imágenes.")
print("Shape descriptores globales:", all_desc.shape)
k = 32
codebook, kmeans = build_codebook(all_desc, k=k)
print("Codebook generado con K =", k)
print("Shape codebook:", codebook.shape)

all_hists = {}

for path, desc in zip(image_paths, per_image_desc):
    hist = compute_histogram(desc, kmeans, k)
    all_hists[path] = hist

print("Histogramas generados.")


query_path = image_paths[0]   # escogemos la primera imagen del dataset
query_hist = compute_query_histogram(query_path, kmeans, k)

print("Histograma del query generado.")


results = knn_secuencial(query_hist, all_hists, top_k=5)

print("\nResultados del KNN:")
for img, score in results:
    print(img, " → similitud:", score)
