import json
import math
import os
from procesador_texto import cargar_lista_parada, preprocesar_texto


class BuscadorSPIMI:

    def __init__(self, nombre_tabla_indice):
        self.nombre_indice = nombre_tabla_indice

        self.archivo_indice_final = os.path.join(self.nombre_indice, "indice_final.jsonl")
        self.archivo_lexicon = os.path.join(self.nombre_indice, "lexicon.json")
        self.archivo_idf = os.path.join(self.nombre_indice, "idf.json")
        self.archivo_normas = os.path.join(self.nombre_indice, "normas.json")

        self.puntero_archivo_indice = None
        self.diccionario_idf = {}
        self.normas_documentos = {}
        self.lexicon = {}

        self._cargar_recursos_en_ram()

    def _cargar_recursos_en_ram(self):
        try:
            self.diccionario_idf = {}
            with open(self.archivo_idf, 'r', encoding='utf-8') as f:
                for linea in f:
                    try:
                        datos = json.loads(linea)
                        self.diccionario_idf[datos['termino']] = datos['idf']
                    except json.JSONDecodeError:
                        continue

            with open(self.archivo_normas, 'r', encoding='utf-8') as f:
                self.normas_documentos = json.load(f)
                self.normas_documentos = {int(k): v for k, v in self.normas_documentos.items()}

            with open(self.archivo_lexicon, 'r', encoding='utf-8') as f:
                self.lexicon = json.load(f)

            self.puntero_archivo_indice = open(self.archivo_indice_final, 'r', encoding='utf-8')

        except FileNotFoundError as e:
            print(f"Error: Archivos de índice no encontrados para '{self.nombre_indice}'. {e.filename}")
            self.puntero_archivo_indice = None

    def _obtener_postings_de_disco(self, termino):
        if not self.puntero_archivo_indice or termino not in self.lexicon:
            return None

        posicion = self.lexicon[termino]

        try:
            self.puntero_archivo_indice.seek(posicion)
            linea = self.puntero_archivo_indice.readline()
            datos = json.loads(linea)
            return datos['lista_posteo']
        except Exception:
            return None

    def rankear_consulta(self, consulta_str, top_k=10):
        if not self.puntero_archivo_indice:
            return []

        tokens_consulta = preprocesar_texto(consulta_str)
        if not tokens_consulta:
            return []

        tf_consulta = {}
        for token in tokens_consulta:
            tf_consulta[token] = tf_consulta.get(token, 0) + 1

        vector_consulta = {}
        norma_consulta_cuadrado = 0.0

        for token, tf in tf_consulta.items():
            if token in self.diccionario_idf:
                idf = self.diccionario_idf[token]
                peso_tq = tf * idf
                vector_consulta[token] = peso_tq
                norma_consulta_cuadrado += peso_tq ** 2

        norma_consulta = 0.0
        if norma_consulta_cuadrado > 0:
            norma_consulta = math.sqrt(norma_consulta_cuadrado)

        if norma_consulta == 0.0:
            return []

        puntajes_doc = {}

        for termino_consulta, peso_consulta in vector_consulta.items():

            lista_posteo = self._obtener_postings_de_disco(termino_consulta)

            if lista_posteo:
                for id_doc_str, tf_doc in lista_posteo.items():
                    id_doc = int(id_doc_str)

                    idf_termino = self.diccionario_idf.get(termino_consulta, 0.0)
                    if idf_termino == 0.0:
                        continue

                    peso_td = tf_doc * idf_termino
                    puntajes_doc[id_doc] = puntajes_doc.get(id_doc, 0.0) + (peso_consulta * peso_td)

        puntajes_finales = []
        for id_doc, producto_punto in puntajes_doc.items():
            if id_doc in self.normas_documentos:
                norma_doc = self.normas_documentos[id_doc]
                denominador = (norma_consulta * norma_doc)
                if denominador > 0:
                    similitud_coseno = producto_punto / denominador
                    puntajes_finales.append((id_doc, similitud_coseno))

        puntajes_finales.sort(key=lambda item: item[1], reverse=True)
        return puntajes_finales[:top_k]

    def cerrar(self):
        if self.puntero_archivo_indice:
            self.puntero_archivo_indice.close()

