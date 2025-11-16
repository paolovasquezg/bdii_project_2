import os
import json
import math
import heapq
import glob
from procesador_texto import cargar_lista_parada, preprocesar_texto

CONTADOR_BLOQUES = {}


def createInvertedIndex(nombre_tabla_indice, lote_de_registros, columna_texto, columna_id):
    global CONTADOR_BLOQUES

    directorio_temporal = f"{nombre_tabla_indice}"
    CONTADOR_BLOQUES[nombre_tabla_indice] = 0

    if os.path.exists(directorio_temporal):
        for f in glob.glob(os.path.join(directorio_temporal, "*.jsonl")):
            os.remove(f)
    else:
        os.makedirs(directorio_temporal)

    addInvertedIndex(nombre_tabla_indice, lote_de_registros, columna_texto, columna_id)


def addInvertedIndex(nombre_tabla_indice, lote_de_registros, columna_texto, columna_id):
    global CONTADOR_BLOQUES

    directorio_temporal = f"{nombre_tabla_indice}"
    num_bloque = CONTADOR_BLOQUES.get(nombre_tabla_indice, 0)

    indice_invertido_temporal = {}

    for registro in lote_de_registros:
        id_doc = registro.get(columna_id)
        texto_doc = registro.get(columna_texto, '')

        if id_doc is None:
            continue

        tokens = preprocesar_texto(texto_doc)

        conteo_tf = {}
        for token in tokens:
            conteo_tf[token] = conteo_tf.get(token, 0) + 1

        for token, tf in conteo_tf.items():
            if token not in indice_invertido_temporal:
                indice_invertido_temporal[token] = {}
            indice_invertido_temporal[token][id_doc] = tf

    terminos_ordenados = sorted(indice_invertido_temporal.keys())
    nombre_archivo_bloque = os.path.join(directorio_temporal, f"bloque_temporal_{num_bloque}.jsonl")

    with open(nombre_archivo_bloque, 'w', encoding='utf-8') as f:
        for termino in terminos_ordenados:
            lista_posteo = indice_invertido_temporal[termino]
            objeto_linea = {"termino": termino, "lista_posteo": lista_posteo}
            json.dump(objeto_linea, f)
            f.write('\n')

    CONTADOR_BLOQUES[nombre_tabla_indice] = num_bloque + 1


def finalizar_indice_spimi(nombre_tabla_indice):
    directorio_temporal = f"{nombre_tabla_indice}"

    archivo_indice_final = os.path.join(directorio_temporal, "indice_final.jsonl")
    archivo_metadatos = os.path.join(directorio_temporal, "metadatos.json")

    total_documentos = _contar_documentos_en_lotes(directorio_temporal)
    if total_documentos == 0:
        return

    _fusionar_bloques_spimi(directorio_temporal, archivo_indice_final, archivo_metadatos, total_documentos)

    archivo_lexicon = os.path.join(directorio_temporal, "lexicon.json")
    archivo_idf = os.path.join(directorio_temporal, "idf.json")
    archivo_normas = os.path.join(directorio_temporal, "normas.json")

    _calcular_pesos_y_normas(archivo_indice_final, archivo_metadatos, archivo_idf, archivo_normas, archivo_lexicon)


def _contar_documentos_en_lotes(directorio_temporal):
    docs_vistos = set()
    for f_path in glob.glob(os.path.join(directorio_temporal, "*.jsonl")):
        with open(f_path, 'r', encoding='utf-8') as f:
            for linea in f:
                try:
                    datos = json.loads(linea)
                    for doc_id in datos['lista_posteo'].keys():
                        docs_vistos.add(int(doc_id))
                except json.JSONDecodeError:
                    continue
    return len(docs_vistos)


def _fusionar_bloques_spimi(directorio_temporal, archivo_indice_salida, archivo_metadatos_salida, total_documentos):
    rutas_archivos_bloques = glob.glob(os.path.join(directorio_temporal, "bloque_temporal_*.jsonl"))
    if not rutas_archivos_bloques:
        return

    archivos_abiertos = [open(f, 'r', encoding='utf-8') for f in rutas_archivos_bloques]
    heap_minimos = []

    for i, f_abierto in enumerate(archivos_abiertos):
        linea = f_abierto.readline()
        if linea:
            try:
                datos = json.loads(linea)
                heapq.heappush(heap_minimos, (datos['termino'], i, datos['lista_posteo']))
            except json.JSONDecodeError:
                continue

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
                try:
                    siguientes_datos = json.loads(siguiente_linea)
                    heapq.heappush(heap_minimos,
                                   (siguientes_datos['termino'], indice_archivo, siguientes_datos['lista_posteo']))
                except json.JSONDecodeError:
                    continue

        if termino_actual is not None:
            objeto_linea = {"termino": termino_actual, "lista_posteo": postings_actuales}
            json.dump(objeto_linea, archivo_indice_final)
            archivo_indice_final.write('\n')

    for f_abierto in archivos_abiertos:
        f_abierto.close()

    metadatos = {"total_documentos": total_documentos}
    with open(archivo_metadatos_salida, 'w', encoding='utf-8') as f_metadatos:
        json.dump(metadatos, f_metadatos)


def _calcular_pesos_y_normas(archivo_indice, archivo_metadatos, archivo_idf_salida, archivo_normas_salida,
                             archivo_lexicon_salida):
    with open(archivo_metadatos, 'r', encoding='utf-8') as f:
        metadatos = json.load(f)
    N = metadatos['total_documentos']

    if N == 0:
        return

    diccionario_idf = {}
    lexicon = {}

    with open(archivo_indice, 'r', encoding='utf-8') as f_indice, \
            open(archivo_idf_salida, 'w', encoding='utf-8') as f_idf:

        while True:
            posicion_linea = f_indice.tell()
            linea = f_indice.readline()
            if not linea:
                break

            try:
                datos = json.loads(linea)
            except json.JSONDecodeError:
                continue

            termino = datos['termino']
            lista_posteo = datos['lista_posteo']

            lexicon[termino] = posicion_linea

            df = len(lista_posteo)
            idf = 0.0
            if df > 0:
                idf = math.log10(N / df)

            diccionario_idf[termino] = idf

            objeto_linea = {"termino": termino, "idf": idf}
            json.dump(objeto_linea, f_idf)
            f_idf.write('\n')

    with open(archivo_lexicon_salida, 'w', encoding='utf-8') as f_lexicon:
        json.dump(lexicon, f_lexicon)

    normas_doc_cuadrado = {}

    with open(archivo_indice, 'r', encoding='utf-8') as f_indice:
        for linea in f_indice:
            try:
                datos = json.loads(linea)
            except json.JSONDecodeError:
                continue

            termino = datos['termino']
            lista_posteo = datos['lista_posteo']

            puntaje_idf = diccionario_idf.get(termino, 0.0)
            if puntaje_idf == 0.0:
                continue

            for id_doc_str, tf in lista_posteo.items():
                peso_td = tf * puntaje_idf
                id_doc = int(id_doc_str)
                normas_doc_cuadrado[id_doc] = normas_doc_cuadrado.get(id_doc, 0.0) + (peso_td ** 2)

    normas_doc = {}
    for id_doc, suma_cuadrados in normas_doc_cuadrado.items():
        norma = 0.0
        if suma_cuadrados > 0:
            norma = math.sqrt(suma_cuadrados)
        normas_doc[id_doc] = norma

    with open(archivo_normas_salida, 'w', encoding='utf-8') as f_normas:
        json.dump(normas_doc, f_normas)

