# Indexación de Descriptores Locales – Bag of Visual Words

## 1. Construcción del Bag of Visual Words

Primero se extraen descriptores locales (SIFT u ORB) de cada imagen.  
Todos los descriptores del dataset se combinan en una sola matriz y se aplica un algoritmo K-Means implementado manualmente.  
Cada cluster generado representa un *visual word*, y los centroides forman el **codebook**.  

Luego, cada imagen se convierte en un **histograma**, donde cada posición indica cuántos de sus descriptores pertenecen a cada visual word. Esto permite representar todas las imágenes con vectores de igual tamaño.

flowchart LR
    A[Imágenes] --> B[Extracción de Descriptores<br/>(SIFT u ORB)]
    B --> C[Clustering Manual<br/>(K-Means)]
    C --> D[Codebook<br/>(Visual Words)]
    D --> E[Histogramas BoVW<br/>(TF)]


---

## 2. Técnica de Indexación Usada

A cada histograma se le aplica **TF-IDF**, donde:

- **TF** mide la frecuencia del visual word dentro de la imagen  
- **DF** indica en cuántas imágenes aparece  
- **IDF** disminuye el peso de visual words demasiado comunes  

Con estos vectores se construye un **índice invertido**, cuya estructura es:

visual_word_id → lista de (imagen_id, peso_tfidf)

Este índice se guarda en archivos binarios (`df.bin`, `idf.bin`, `inverted_index.bin`) para cargarlo rápidamente cuando se necesita.

---

## 3. Búsqueda KNN sobre los Histogramas

Se implementaron dos métodos de búsqueda:

### a) KNN Secuencial  
1. Se extraen descriptores del query.  
2. Se genera su histograma y se aplica TF-IDF.  
3. Se calcula la similitud del coseno entre el query y cada imagen del dataset.  
4. Se usa un heap para quedarse solo con los K resultados más similares.

### b) KNN usando Índice Invertido  
En lugar de comparar con todas las imágenes, solo se evalúan aquellas que contienen visual words presentes en el query.  
Esto reduce la cantidad de comparaciones y mejora el tiempo de búsqueda.

flowchart TD
    A[Imagen de Consulta] --> B[Descriptores + Histograma TF-IDF]
    B --> C[Índice Invertido<br/>(visual_word → lista de imágenes)]
    C --> D[Filtrado de Candidatos Relevantes]
    D --> E[Cálculo de Similitud de Coseno]
    E --> F[Heap K Resultados<br/>(KNN)]


---

## 4. Impacto de la Maldición de la Dimensionalidad

Los histogramas BoVW pueden tener muchas dimensiones (según k del clustering).  
Esto afecta la precisión y eficiencia de la similitud, porque:

- Distancias se vuelven menos discriminativas  
- Vectores muy grandes generan más ruido  
- El cálculo de similitud se vuelve más costoso  

### Estrategias aplicadas para mitigar el problema:

- **TF-IDF** para reducir impacto de palabras muy frecuentes  
- **Normalización** de vectores antes de la similitud  
- **Elección moderada de k** en el clustering  
- Uso de similitud del coseno, que funciona mejor en alta dimensionalidad  

Estas decisiones permiten que el sistema funcione de forma más estable y con mejores resultados.

