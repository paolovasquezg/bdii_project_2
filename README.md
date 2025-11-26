### Extracción de características

Se extraen descriptores SIFT de cada imagen. Estos descriptores representan puntos importantes como esquinas y bordes. Cada imagen genera varios descriptores y todos juntos se usan para entrenar el modelo BoVW. Los descriptores se almacenan por imagen y también en un arreglo global para el clustering.

---

### Construcción del diccionario visual (K-Means manual)

Para formar las “palabras visuales” se usa un K-Means implementado manualmente, ya que no está permitido usar sklearn. El algoritmo inicializa centroides aleatorios, asigna cada descriptor al centroide más cercano, recalcula promedios y repite varias veces hasta que los centroides dejan de cambiar. El resultado es un diccionario visual con *k* centroides que representan patrones visuales frecuentes en las imágenes.

---

### Generación de histogramas

Cada imagen se convierte en un histograma que cuenta cuántos de sus descriptores pertenecen a cada palabra visual. Este vector numérico es la representación fija de cada imagen en el modelo BoVW.

---

### Aplicación de TF-IDF

Los histogramas se ajustan usando TF-IDF para mejorar su capacidad de diferenciación.  
**df** indica en cuántas imágenes aparece cada palabra visual.  
**idf** se calcula como log(N / df).  
**tf-idf** es el histograma normalizado multiplicado por idf.  
Esto reduce el peso de palabras que aparecen en todas las imágenes y resalta patrones más específicos.

---

### Índice invertido visual

Con los histogramas TF-IDF se construye un índice invertido visual donde:

visual_word_id → [(imagen, peso_tfidf), ...]

Este índice permite acelerar la búsqueda, ya que no revisa toda la colección, sino solo las imágenes relacionadas con las palabras visuales del query.

---

### Persistencia en archivos binarios

Para evitar recalcular todo, se guardan en binarios:
- los histogramas TF-IDF  
- el df  
- el idf  
- el índice invertido  

Esto permite cargar rápidamente toda la información y usarla en pruebas de rendimiento, como comparar la búsqueda secuencial con la búsqueda usando el índice invertido.
