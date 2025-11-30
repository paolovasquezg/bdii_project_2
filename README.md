# Backend – Índice Invertido para Descriptores Locales (Multimedia)

Este módulo implementa un sistema de búsqueda de imágenes basado en descriptores locales usando Bag of Visual Words (BoVW), TF-IDF y un índice invertido.  
El objetivo es permitir recuperar imágenes similares de forma eficiente sin usar librerías externas como sklearn.

---

# Flujo General del Sistema

[Diagrama de flujo: Imagen → Extracción de Descriptores SIFT/ORB → Codebook Clustering → Histogramas TF-IDF → Índice Invertido → KNN Similitud Coseno]

---

# Extracción de Características

Para cada imagen se extraen descriptores locales (SIFT u ORB).  
Cada descriptor representa una pequeña parte de la imagen en forma de vector numérico.

- Una imagen → muchos descriptores
- Todos se guardan para construir el codebook
- Proceso obligatorio para generar el sistema

**Este paso incluye:**
- Leer cada imagen del dataset
- Extraer descriptores locales  
- Guardarlos para el clustering
- Acumular todos los descriptores en una sola matriz

---

# Construcción del Codebook

Se agrupan todos los descriptores usando un algoritmo propio de K-Means (sin sklearn).  
Cada cluster representa una visual word.

[Diagrama de flujo: Descriptores de todas las imágenes → Agrupación en k clusters → Centroides → Codebook (k visual words)]

**Proceso:**
- Combinar descriptores en una matriz
- Aplicar K-Means manual
- Cada cluster = visual word
- Centroides = codebook
- Cada imagen → histograma de visual words

---

# Técnica de Indexación

Se aplica el modelo TF-IDF a los histogramas:

- TF: frecuencia del visual word en la imagen
- DF: en cuántas imágenes aparece  
- IDF: reduce peso de visual words comunes

Con los vectores TF-IDF se construye un índice invertido:

visual_word_id → [ (imagen_id, peso_tfidf), ... ]

---

# Índice Invertido

Se construye un índice donde cada visual word apunta a las imágenes donde aparece, igual que en motores de búsqueda de texto.

[Diagrama de flujo: Visual Word ID → Lista de imágenes donde aparece → Peso TF-IDF por imagen]

**Ejemplo visual:**
Word 0 → img_01 (0.12), img_03 (0.15)
Word 1 → img_02 (0.08), img_03
Word 2 → img_01

---

# Flujo de Construcción del Índice Invertido

[Diagrama de flujo: Histogramas TF-IDF → Calcular DF e IDF → Generar pesos TF-IDF → Crear listas por visual word → Guardar inverted_index.bin]

---

# Búsqueda KNN

Para buscar imágenes similares:

1. Se extraen descriptores de la imagen de consulta
2. Se genera su histograma usando el codebook
3. Se obtiene su vector TF-IDF
4. Se compara contra el índice invertido
5. Se calcula la similitud de coseno
6. Se usan heaps para obtener los K más parecidos

[Diagrama de flujo: Imagen de consulta → Histograma TF-IDF → Comparación contra índice invertido → Similitud de coseno → Heap con K resultados → Top K imágenes similares]

**Versiones implementadas:**
- KNN secuencial
- KNN usando índice invertido (más eficiente)

---

# Maldición de la Dimensionalidad

Los histogramas tienen muchas dimensiones y esto afecta la precisión.  
Para mitigar este problema se aplicó:

- TF-IDF para reducir ruido
- Normalización de vectores
- Escoger un valor "k" moderado en el clustering

Estas decisiones mejoran la precisión y estabilidad del sistema.

---

## Distribución de Visual Words

[Gráfico de torta: Word 0: 12%, Word 1: 8%, Word 2: 20%, Word 3: 14%, Word 4: 10%, Otros: 36%]

---

# Archivos Generados

Los datos del sistema se guardan en archivos binarios para evitar recalcular todo:

- codebook.bin → centroides
- df.bin → Document Frequency
- idf.bin → IDF
- histograms.bin → histogramas TF-IDF
- inverted_index.bin → índice invertido

Esto hace que el sistema pueda cargarse rápido sin recomputar los descriptores.

---


# Esquema Completo

[Diagrama de flujo completo: Conjunto de Imágenes → Extracción de Descriptores → Clustering (Codebook) → Histogramas TF-IDF → Índice Invertido → Consulta KNN → Ranking de Similares]

---

# Resumen General

- Se extraen descriptores locales por imagen
- Se construye un codebook (visual words)
- Se crean histogramas TF-IDF para cada imagen
- Se construye un índice invertido
- Se implementa KNN con similitud de coseno
- Todo se guarda en binarios para reutilizarlo
