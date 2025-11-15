
import string
import nltk
import pandas as pd
import json
import os
import math
import glob
import heapq
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer


def obtener_lista_parada(ruta_archivo="stoplist-1.txt"):
    try:
        with open(ruta_archivo, 'r', encoding='utf-8') as f:
            palabras_parada = {linea.strip() for linea in f}
        return palabras_parada
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de lista de parada en: {ruta_archivo}")
        return set()


def preprocesar_texto(texto, lematizador, palabras_parada):
    if not isinstance(texto, str):
        return []

    tokens = word_tokenize(texto.lower(), language='spanish')

    tokens_procesados = []
    for token in tokens:
        if not token.isalpha():
            continue
        if token in palabras_parada:
            continue
        raiz = lematizador.stem(token)
        tokens_procesados.append(raiz)

    return tokens_procesados


def construir_indice_spimi(ruta_dataset, ruta_lista_parada, nombre_columna_texto, tamano_bloque=1000):
    print("Iniciando Fase 2.A: Construcción de Bloques SPIMI...")

    lematizador = SnowballStemmer('spanish')
    palabras_parada = obtener_lista_parada(ruta_lista_parada)

    directorio_temporal = "bloques_temporales"
    if not os.path.exists(directorio_temporal):
        os.makedirs(directorio_temporal)

    num_bloque = 0
    contador_id_doc = 0

    for bloque_datos in pd.read_csv(ruta_dataset, chunksize=tamano_bloque):
        indice_invertido_temporal = {}
        print(f"Procesando bloque {num_bloque}...")

        for indice, fila in bloque_datos.iterrows():
            id_doc = contador_id_doc
            texto_doc = fila.get(nombre_columna_texto, '')

            tokens = preprocesar_texto(texto_doc, lematizador, palabras_parada)

            conteo_tf = {}
            for token in tokens:
                conteo_tf[token] = conteo_tf.get(token, 0) + 1

            for token, tf in conteo_tf.items():
                if token not in indice_invertido_temporal:
                    indice_invertido_temporal[token] = {}
                indice_invertido_temporal[token][id_doc] = tf

            contador_id_doc += 1

        terminos_ordenados = sorted(indice_invertido_temporal.keys())
        nombre_archivo_bloque = os.path.join(directorio_temporal, f"bloque_temporal_{num_bloque}.jsonl")

        with open(nombre_archivo_bloque, 'w', encoding='utf-8') as f:
            for termino in terminos_ordenados:
                lista_posteo = indice_invertido_temporal[termino]
                objeto_linea = {"termino": termino, "lista_posteo": lista_posteo}
                json.dump(objeto_linea, f)
                f.write('\n')

        num_bloque += 1

    total_documentos = contador_id_doc
    print(f"Fase 2.A completada. Se procesaron {total_documentos} documentos.")
    print(f"Se crearon {num_bloque} bloques temporales en formato .jsonl.")

    return directorio_temporal, total_documentos, num_bloque


def fusionar_bloques_spimi(directorio_temporal, archivo_indice_salida, archivo_metadatos_salida, total_documentos):
    print("\nIniciando Fase 2.B: Fusión de bloques SPIMI (con 'B buffers')...")

    rutas_archivos_bloques = glob.glob(os.path.join(directorio_temporal, "bloque_temporal_*.jsonl"))
    archivos_abiertos = [open(f, 'r', encoding='utf-8') for f in rutas_archivos_bloques]
    heap_minimos = []

    for i, f_abierto in enumerate(archivos_abiertos):
        linea = f_abierto.readline()
        if linea:
            datos = json.loads(linea)
            heapq.heappush(heap_minimos, (datos['termino'], i, datos['lista_posteo']))

    with open(archivo_indice_salida, 'w', encoding='utf-8') as archivo_indice_final:
        termino_actual = None
        postings_actuales = {}

        while heap_minimos:
            termino, indice_archivo, lista_posteo = heapq.heappop(heap_minimos)

            if termino != termino_actual:
                if termino_actual is not None:
                    objeto_linea = {"termino": termino_actual, "lista_posteo": postings_actuales}
                    json.dump(objeto_linea, archivo_indice_final)
                    archivo_indice_final.write('\n')

                termino_actual = termino
                postings_actuales = lista_posteo
            else:
                postings_actuales.update(lista_posteo)

            siguiente_linea = archivos_abiertos[indice_archivo].readline()
            if siguiente_linea:
                siguientes_datos = json.loads(siguiente_linea)
                heapq.heappush(heap_minimos,
                               (siguientes_datos['termino'], indice_archivo, siguientes_datos['lista_posteo']))

        if termino_actual is not None:
            objeto_linea = {"termino": termino_actual, "lista_posteo": postings_actuales}
            json.dump(objeto_linea, archivo_indice_final)
            archivo_indice_final.write('\n')

    for f_abierto in archivos_abiertos:
        f_abierto.close()

    metadatos = {"total_documentos": total_documentos}
    with open(archivo_metadatos_salida, 'w', encoding='utf-8') as f_metadatos:
        json.dump(metadatos, f_metadatos)

    print("Fase 2.B completada.")
    print(f"Índice final (.jsonl) guardado en: {archivo_indice_salida}")
    print(f"Metadatos guardados en: {archivo_metadatos_salida}")


def calcular_pesos(archivo_indice, archivo_metadatos, archivo_idf, archivo_normas):
    print("\nIniciando Fase 3: Cálculo de Pesos (IDF y Normas)...")

    with open(archivo_metadatos, 'r', encoding='utf-8') as f:
        metadatos = json.load(f)
    N = metadatos['total_documentos']

    if N == 0:
        print("Error: No se encontraron documentos (N=0).")
        return
    print(f"Número total de documentos (N): {N}")

    print("Iniciando Pasada 1: Calculando puntajes IDF...")
    diccionario_idf = {}

    with open(archivo_indice, 'r', encoding='utf-8') as f_indice, \
            open(archivo_idf, 'w', encoding='utf-8') as f_idf:

        for linea in f_indice:
            datos = json.loads(linea)
            termino = datos['termino']
            lista_posteo = datos['lista_posteo']

            df = len(lista_posteo)
            idf = math.log10(N / df)

            diccionario_idf[termino] = idf

            objeto_linea = {"termino": termino, "idf": idf}
            json.dump(objeto_linea, f_idf)
            f_idf.write('\n')

    print(f"Pasada 1 completada. IDFs guardados en {archivo_idf}")

    print("Iniciando Pasada 2: Calculando normas de documentos...")
    normas_doc_cuadrado = {}

    with open(archivo_indice, 'r', encoding='utf-8') as f_indice:
        for linea in f_indice:
            datos = json.loads(linea)
            termino = datos['termino']
            lista_posteo = datos['lista_posteo']

            puntaje_idf = diccionario_idf.get(termino, 0.0)
            if puntaje_idf == 0.0:
                continue

            for id_doc_str, tf in lista_posteo.items():
                peso_td = tf * puntaje_idf
                id_doc = int(id_doc_str)
                normas_doc_cuadrado[id_doc] = normas_doc_cuadrado.get(id_doc, 0.0) + (peso_td ** 2)

    print("Finalizando cálculo de normas (aplicando raíz cuadrada)...")
    normas_doc = {}
    for id_doc, suma_cuadrados in normas_doc_cuadrado.items():
        normas_doc[id_doc] = math.sqrt(suma_cuadrados)

    with open(archivo_normas, 'w', encoding='utf-8') as f_normas:
        json.dump(normas_doc, f_normas)

    print(f"Pasada 2 completada. Normas guardadas en {archivo_normas}")
    print("Fase 3 completada.")


def cargar_idfs(archivo_idf):
    print(f"Cargando puntajes IDF desde {archivo_idf}...")
    idfs = {}
    with open(archivo_idf, 'r', encoding='utf-8') as f:
        for linea in f:
            datos = json.loads(linea)
            idfs[datos['termino']] = datos['idf']
    print("IDFs cargados.")
    return idfs


def cargar_normas(archivo_normas):
    print(f"Cargando normas de documentos desde {archivo_normas}...")
    with open(archivo_normas, 'r', encoding='utf-8') as f:
        normas_doc = json.load(f)
    normas_doc = {int(k): v for k, v in normas_doc.items()}
    print("Normas cargadas.")
    return normas_doc


def cargar_indice_invertido(archivo_indice):
    print(f"Cargando índice invertido desde {archivo_indice}...")
    indice_invertido = {}
    with open(archivo_indice, 'r', encoding='utf-8') as f:
        for linea in f:
            datos = json.loads(linea)
            postings_llaves_int = {int(k): v for k, v in datos['lista_posteo'].items()}
            indice_invertido[datos['termino']] = postings_llaves_int
    print("Índice invertido cargado.")
    return indice_invertido


def rankear_consulta(consulta_str, lematizador, palabras_parada, indice_invertido, puntajes_idf, normas_doc, top_k=10):
    tokens_consulta = preprocesar_texto(consulta_str, lematizador, palabras_parada)
    if not tokens_consulta:
        return []

    tf_consulta = {}
    for token in tokens_consulta:
        tf_consulta[token] = tf_consulta.get(token, 0) + 1

    vector_consulta = {}
    norma_consulta_cuadrado = 0.0

    for token, tf in tf_consulta.items():
        if token in puntajes_idf:
            idf = puntajes_idf[token]
            peso_tq = tf * idf
            vector_consulta[token] = peso_tq
            norma_consulta_cuadrado += peso_tq ** 2

    norma_consulta = math.sqrt(norma_consulta_cuadrado)
    if norma_consulta == 0.0:
        return []

    puntajes_doc = {}

    for termino_consulta, peso_consulta in vector_consulta.items():
        if termino_consulta in indice_invertido:
            lista_posteo = indice_invertido[termino_consulta]
            for id_doc, tf_doc in lista_posteo.items():
                peso_td = tf_doc * puntajes_idf[termino_consulta]
                puntajes_doc[id_doc] = puntajes_doc.get(id_doc, 0.0) + (peso_consulta * peso_td)

    puntajes_finales = []
    for id_doc, producto_punto in puntajes_doc.items():
        if id_doc in normas_doc:
            norma_doc = normas_doc[id_doc]
            denominador = (norma_consulta * norma_doc)
            if denominador > 0:
                similitud_coseno = producto_punto / denominador
                puntajes_finales.append((id_doc, similitud_coseno))

    puntajes_finales.sort(key=lambda item: item[1], reverse=True)
    return puntajes_finales[:top_k]


if __name__ == "__main__":

    ARCHIVO_DATOS = "news_es-2.csv"
    ARCHIVO_LISTA_PARADA = "stoplist-1.txt"
    COLUMNA_TEXTO = "contenido"

    ARCHIVO_INDICE_FINAL = "final_inverted_index.jsonl"
    ARCHIVO_METADATOS = "metadata.json"
    ARCHIVO_IDF = "idf_scores.jsonl"
    ARCHIVO_NORMAS = "doc_norms.json"

    print("--- INICIANDO PIPELINE DE INDEXACIÓN ---")
    directorio_temp, N_docs, num_bloques = construir_indice_spimi(
        ARCHIVO_DATOS,
        ARCHIVO_LISTA_PARADA,
        COLUMNA_TEXTO,
        tamano_bloque=1000
    )
    fusionar_bloques_spimi(
        directorio_temp,
        ARCHIVO_INDICE_FINAL,
        ARCHIVO_METADATOS,
        N_docs
    )
    calcular_pesos(
        ARCHIVO_INDICE_FINAL,
        ARCHIVO_METADATOS,
        ARCHIVO_IDF,
        ARCHIVO_NORMAS
    )
    print("--- PIPELINE DE INDEXACIÓN COMPLETADO ---")

    # print("\n--- INICIANDO SERVICIO DE CONSULTAS ---")
    #
    # lematizador_consulta = SnowballStemmer('spanish')
    # stopwords_consulta = obtener_lista_parada(ARCHIVO_LISTA_PARADA)
    #
    # try:
    #     puntajes_idf_memoria = cargar_idfs(ARCHIVO_IDF)
    #     normas_doc_memoria = cargar_normas(ARCHIVO_NORMAS)
    #     indice_invertido_memoria = cargar_indice_invertido(ARCHIVO_INDICE_FINAL)
    # except FileNotFoundError:
    #     print("\nError: Archivos de índice no encontrados.")
    #     print("Por favor, descomenta el 'PASO 1' y ejecuta el script para crear los índices primero.")
    #     exit()
    #
    # print("\n--- ¡Sistema listo para recibir consultas! ---")
    #
    # lista_consultas = [
    #     "presidente de colombia",
    #     "sostenibilidad y finanzas",
    #     "tecnología futurista y moda"
    # ]
    #
    # for consulta in lista_consultas:
    #     print(f"\nProcesando consulta: '{consulta}'")
    #
    #     resultados = rankear_consulta(
    #         consulta,
    #         lematizador_consulta,
    #         stopwords_consulta,
    #         indice_invertido_memoria,
    #         puntajes_idf_memoria,
    #         normas_doc_memoria,
    #         top_k=5
    #     )
    #
    #     if not resultados:
    #         print("No se encontraron resultados relevantes.")
    #     else:
    #         print(f"Top {len(resultados)} resultados relevantes:")
    #         for id_doc, score in resultados:
    #             print(f"  DocID: {id_doc} (Score: {score:.4f})")