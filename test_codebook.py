import glob
from backend.multimedia.extractors import extract_features
from backend.multimedia.codebook import (
    collect_descriptors,
    build_codebook,
    compute_histogram,
    compute_df,
    compute_idf,
    histogram_to_tfidf,
    build_inverted_index,
    save_binary,
    load_binary,
    prepare_and_save_all
)

# ===============================
# 1. Obtener lista de imágenes
# ===============================

image_paths = glob.glob("images/*.png") + glob.glob("images/*.jpg")
print(f"Imagenes encontradas: {len(image_paths)}")

if len(image_paths) == 0:
    raise RuntimeError("No se encontraron imágenes para procesar.")


# ===============================
# 2. Extraer descriptores
# ===============================

per_img_desc, all_desc = collect_descriptors(image_paths)
print("Descriptores extraídos.")
print("Total descriptores:", all_desc.shape)


# ===============================
# 3. Entrenar codebook
# ===============================

k = 16
codebook, kmeans = build_codebook(all_desc, k=k)
print("Codebook entrenado.")
print("Shape del codebook:", codebook.shape)


# ===============================
# 4. Crear histogramas por imagen
# ===============================

all_hists = {}
for path, desc in zip(image_paths, per_img_desc):
    hist = compute_histogram(desc, kmeans, k)
    all_hists[path] = hist

print("Histogramas generados.")
print("Ejemplo histograma:", next(iter(all_hists.values())))


# ===============================
# 5. Calcular DF e IDF
# ===============================

df = compute_df(all_hists)
idf = compute_idf(df, N=len(all_hists))

print("DF:", df)
print("IDF:", idf)


# ===============================
# 6. Generar TF–IDF por imagen
# ===============================

tfidf_hists = {
    path: histogram_to_tfidf(hist, idf)
    for path, hist in all_hists.items()
}

print("TF-IDF generado.")
print("Ejemplo TF-IDF:", next(iter(tfidf_hists.values())))


# ===============================
# 7. Construir índice invertido
# ===============================

inverted = build_inverted_index(tfidf_hists)

print("Índice invertido construido.")
print("Clusters indexados:", list(inverted.keys())[:5])


# ===============================
# 8. Guardar en binarios
# ===============================

save_binary(tfidf_hists, "tfidf_hists.bin")
save_binary(df, "df.bin")
save_binary(idf, "idf.bin")
save_binary(inverted, "inverted_index.bin")

print("Binarios guardados correctamente.")


# ===============================
# 9. Cargar binarios
# ===============================

tfidf_loaded = load_binary("tfidf_hists.bin")
df_loaded = load_binary("df.bin")
idf_loaded = load_binary("idf.bin")
inv_loaded = load_binary("inverted_index.bin")

print("Binarios cargados correctamente.")

print("Comparación TF-IDF (primera imagen):")
first_img = list(tfidf_hists.keys())[0]
print("Original:", tfidf_hists[first_img])
print("Cargado :", tfidf_loaded[first_img])


print("\n✔ TEST COMPLETO: Todos los pasos del codebook funcionan correctamente.")
