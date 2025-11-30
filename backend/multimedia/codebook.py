import numpy as np
from .kmeans_custom import KMeansCustom
from .demo_config import DEMO_CONFIG, PROD_CONFIG
from sklearn.feature_extraction.text import TfidfTransformer
from tqdm import tqdm
from typing import List
import sys
from pathlib import Path
# Agregar scripts.py al path para los imports de utils
sys.path.append(str(Path(__file__).parent.parent / "scripts.py"))
from utils import (
    save_pickle, load_pickle,
    Timer, PROCESSED_DIR, INDEX_DIR
)


class AcousticCodebook:    
    def __init__(self, n_clusters: int = 500, demo_mode: bool = False):
        if demo_mode:
            config = DEMO_CONFIG
            self.n_clusters = n_clusters  # Usar el parámetro pasado, no el del config
            self.max_samples = config["max_samples"]
            self.max_iter = config["max_iter"]
            print(f" MODO DEMO: Configuración optimizada para velocidad (K={n_clusters})")
        else:
            config = PROD_CONFIG
            self.n_clusters = n_clusters
            self.max_samples = None
            self.max_iter = 300
        
        self.kmeans = None
        self.codebook = None  # Centroides (palabras acústicas)
        self.tfidf_transformer = None
        self.demo_mode = demo_mode
    
    def build_codebook(self, descriptors: np.ndarray) -> None:
        print(f"\n Construyendo codebook con K={self.n_clusters}")
        
        # En modo demo, usar subset de datos
        if self.demo_mode and self.max_samples and len(descriptors) > self.max_samples:
            print(f" MODO DEMO: Usando subset de {self.max_samples:,} descriptores (de {len(descriptors):,})")
            indices = np.random.choice(len(descriptors), self.max_samples, replace=False)
            descriptors = descriptors[indices]
        
        print(f"� Descriptores de entrada: {descriptors.shape}")
        
        with Timer("K-Means clustering"):
            self.kmeans = KMeansCustom(
                n_clusters=self.n_clusters,
                max_iter=self.max_iter,
                tol=1e-4  # Tolerancia por defecto
            )
            self.kmeans.fit(descriptors)
        
        # Los centroides son las "acoustic words"
        self.codebook = self.kmeans.cluster_centers_
        
        print(f"Codebook creado: {self.codebook.shape}")
        print(f"   Cada audio será representado por {self.n_clusters} palabras acústicas")
        
        if self.demo_mode:
            print(" DEMO: Entrenamiento completado con configuración rápida")
    
    def audio_to_histogram(self, descriptors: np.ndarray) -> np.ndarray:
        if len(descriptors) == 0:
            return np.zeros(self.n_clusters)
        
        # Asignar cada descriptor al codeword más cercano
        labels = self.kmeans.predict(descriptors)
        
        # Crear histograma: contar frecuencia de cada palabra acústica
        histogram = np.zeros(self.n_clusters)
        for label in labels:
            histogram[label] += 1
        
        # Normalizar (TF - Term Frequency)
        if histogram.sum() > 0:
            histogram = histogram / histogram.sum()
        
        return histogram
    
    def convert_dataset_to_histograms(
        self, 
        all_descriptors: List[np.ndarray]
    ) -> np.ndarray:
        print(f"\n Convirtiendo {len(all_descriptors)} audios a histogramas")
        
        histograms = []
        for descriptors in tqdm(all_descriptors, desc="Generando histogramas"):
            hist = self.audio_to_histogram(descriptors)
            histograms.append(hist)
        
        histograms = np.array(histograms)
        print(f" Histogramas generados: {histograms.shape}")
        print(f"   Cada fila = 1 audio representado por {self.n_clusters} frecuencias")
        
        return histograms
    
    def apply_tfidf(self, histograms: np.ndarray) -> np.ndarray:
        print(f"\n🔢 Aplicando TF-IDF")
        print(f"   TF: Frecuencia de cada palabra en el audio")
        print(f"   IDF: Importancia de la palabra en toda la colección")
        
        with Timer("TF-IDF transformation"):
            self.tfidf_transformer = TfidfTransformer()
            tfidf_matrix = self.tfidf_transformer.fit_transform(histograms)
            tfidf_dense = tfidf_matrix.toarray()
        
        print(f"TF-IDF aplicado: {tfidf_dense.shape}")
        print(f"Min: {tfidf_dense.min():.4f}, Max: {tfidf_dense.max():.4f}")
        
        return tfidf_dense
    
    def save(self, filepath: str) -> None:
        codebook_data = {
            'kmeans': self.kmeans,
            'codebook': self.codebook,
            'tfidf_transformer': self.tfidf_transformer,
            'n_clusters': self.n_clusters
        }
        save_pickle(codebook_data, filepath)
    
    def load(self, filepath: str) -> None:
        codebook_data = load_pickle(filepath)
        self.kmeans = codebook_data['kmeans']
        self.codebook = codebook_data['codebook']
        self.tfidf_transformer = codebook_data['tfidf_transformer']
        self.n_clusters = codebook_data['n_clusters']


def main():    
    print("="*60)
    print("🎵 FASE 2: CONSTRUCCIÓN DEL CODEBOOK")
    print("="*60)
    
    # 1. Cargar descriptores de la FASE 1
    print("\n Cargando datos de la FASE 1...")
    descriptors_flat = np.load(PROCESSED_DIR / "descriptors_flat.npy")
    all_descriptors = load_pickle(PROCESSED_DIR / "all_descriptors.pkl")
    
    print(f"  Descriptores cargados: {descriptors_flat.shape}")
    print(f"   Total audios: {len(all_descriptors)}")
    print(f"   Descriptores por audio: ~{len(descriptors_flat) // len(all_descriptors)}")
    
    # 2. Construir codebook con K-Means
    K = 500  # Número de palabras acústicas
    print(f"\n K = {K} palabras acústicas")
    
    codebook_builder = AcousticCodebook(n_clusters=K)
    codebook_builder.build_codebook(descriptors_flat)
    
    # 3. Convertir audios a histogramas
    histograms = codebook_builder.convert_dataset_to_histograms(all_descriptors)
    
    # Guardar histogramas raw
    histograms_path = PROCESSED_DIR / "histograms.npy"
    np.save(histograms_path, histograms)
    print(f"\n Guardado: {histograms_path}")
    
    # 4. Aplicar TF-IDF
    tfidf_matrix = codebook_builder.apply_tfidf(histograms)
    
    # Guardar TF-IDF
    tfidf_path = PROCESSED_DIR / "tfidf_matrix.npy"
    np.save(tfidf_path, tfidf_matrix)
    print(f" Guardado: {tfidf_path}")
    
    # 5. Guardar codebook completo
    codebook_path = INDEX_DIR / "codebook.pkl"
    codebook_builder.save(codebook_path)
    print(f" Guardado: {codebook_path}")
    
    # Estadísticas finales
    print("\n" + "="*60)
    print(" FASE 2 COMPLETADA")
    print("="*60)
    print(f"  Archivos generados:")
    print(f"   - Histogramas: {histograms_path}")
    print(f"   - TF-IDF: {tfidf_path}")
    print(f"   - Codebook: {codebook_path}")
    print(f"\n Resumen:")
    print(f"   - Audios procesados: {len(all_descriptors)}")
    print(f"   - Palabras acústicas: {K}")
    print(f"   - Dimensión TF-IDF: {tfidf_matrix.shape}")
    print(f"\n Siguiente: Construcción del Índice Invertido (FASE 3)")


if __name__ == "__main__":
    main()