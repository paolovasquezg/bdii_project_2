# Indexación de Descriptores Locales (Multimedia Database)

En esta parte del proyecto se implementó un sistema de búsqueda para imágenes usando descriptores locales.  
El objetivo es representar cada imagen como un conjunto de “visual words” y luego buscar imágenes similares usando técnicas como TF-IDF, índice invertido y KNN.

---

# Flujo general del sistema

```mermaid
flowchart LR
    A[Imagen] --> B[Extraccion de Descriptores SIFT ORB]
    B --> C[Codebook Clustering]
    C --> D[Histogramas TF-IDF]
    D --> E[Indice Invertido]
    E --> F[KNN Similitud Coseno]


---

#  Extracción de Características

Para cada imagen se extraen **descriptores locales** (como SIFT u ORB).  
Cada descriptor representa una pequeña parte de la imagen en forma de vector numérico.

- Una imagen → muchos descriptores.  
- Todos se guardan para construir el codebook.

---

#  Construcción del Codebook

Se agrupan todos los descriptores usando un algoritmo propio de K-Means (sin sklearn).  
Cada cluster representa una **visual word**.

```mermaid
flowchart LR
    A[Descriptores de todas las imágenes] --> B[Agrupación en k clusters]
    B --> C[Centroides]
    C --> D[Codebook<br>k visual words]
```

Cada imagen luego se convierte en un histograma que indica cuántas visual words contiene.


---

# Índice Invertido

Se construye un índice donde **cada visual word apunta a las imágenes donde aparece**, igual que en motores de búsqueda de texto.

```mermaid
flowchart LR
    A[Visual Word ID] --> B[Lista de imágenes donde aparece]
    B --> C[Peso TF-IDF por imagen]
```

Ejemplo visual:

```mermaid
graph TD
    W0["Word 0"] --> I1["img_01 (0.12)"]
    W0 --> I3["img_03 (0.15)"]
    W1["Word 1"] --> I2["img_02 (0.08)"]
    W1 --> I3
    W2["Word 2"] --> I1
```

---

# Búsqueda KNN 

Para buscar imágenes similares:

1. Se extraen descriptores de la imagen de consulta.  
2. Se genera su histograma TF-IDF.  
3. Se compara contra el índice invertido.  
4. Se calcula la similitud de coseno.  
5. Se usan heaps para obtener los K más parecidos.

```mermaid
flowchart TD
    A[Imagen de consulta] --> B[Histograma TF-IDF]
    B --> C[Comparación contra índice invertido]
    C --> D[Similitud de coseno]
    D --> E[Heap con K resultados]
    E --> F[Top K imágenes similares]
```

---


##  Distribución de visual words

```mermaid
pie showData
    title Distribución de palabras visuales
    "Word 0" : 12
    "Word 1" : 8
    "Word 2" : 20
    "Word 3" : 14
    "Word 4" : 10
    "Otros" : 36
```

---

#  Archivos generados

Los datos del sistema se guardan en archivos binarios para evitar recalcular todo:

- `codebook.bin` → centroides  
- `df.bin` → Document Frequency  
- `idf.bin` → IDF  
- `histograms.bin` → histogramas TF-IDF  
- `inverted_index.bin` → índice invertido  

Esto hace que el sistema pueda cargarse rápido sin recomputar los descriptores.

---

#  Resumen general

- Se extraen descriptores locales por imagen.  
- Se construye un codebook (visual words).  
- Se crean histogramas TF-IDF para cada imagen.  
- Se construye un índice invertido.  
- Se implementa KNN con similitud de coseno.  
- Todo se guarda en binarios para reutilizarlo.

