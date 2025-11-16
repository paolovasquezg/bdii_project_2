import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

LEMATIZADOR_ES = SnowballStemmer('spanish')
LISTA_PARADA = set()


def cargar_lista_parada(ruta_archivo="stoplist-1.txt"):
    global LISTA_PARADA
    try:
        with open(ruta_archivo, 'r', encoding='utf-8') as f:
            LISTA_PARADA = {linea.strip() for linea in f}
        print(f"Lista de parada cargada con {len(LISTA_PARADA)} palabras.")
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de lista de parada en: {ruta_archivo}")
        LISTA_PARADA = set()


def preprocesar_texto(texto_sucio):
    if not isinstance(texto_sucio, str):
        return []

    tokens = word_tokenize(texto_sucio.lower(), language='spanish')

    tokens_limpios = []
    for token in tokens:
        if not token.isalpha():
            continue
        if token in LISTA_PARADA:
            continue
        raiz = LEMATIZADOR_ES.stem(token)
        tokens_limpios.append(raiz)

    return tokens_limpios