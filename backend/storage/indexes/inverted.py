"""
Índice Invertido para Búsqueda de Audio
backend/storage/indexes/inverted.py

Integrado con el sistema de índices del Proyecto 1
"""

import numpy as np
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import pickle


class InvertedIndex:
    """
    Índice Invertido Flat para búsqueda de audio por similitud
    
    Estructura:
        {
            word_id: [(audio_id, tfidf_score), ...]
        }
    
    Uso:
        index = InvertedIndex()
        index.build(tfidf_matrix)
        candidates = index.get_candidates(query_vector)
    """
    
    def __init__(self):
        self.index: Dict[int, List[Tuple[int, float]]] = {}
        self.audio_norms: Dict[int, float] = {}
        self.n_words: int = 0
        self.n_audios: int = 0
    
    def build(self, tfidf_matrix: np.ndarray, min_score: float = 0.0) -> None:
        """
        Construye el índice invertido desde una matriz TF-IDF
        
        Args:
            tfidf_matrix: Matriz (n_audios, n_words) con scores TF-IDF
            min_score: Score mínimo para incluir en posting list
        """
        self.n_audios, self.n_words = tfidf_matrix.shape
        self.index = defaultdict(list)
        
        print(f"🔨 Construyendo índice invertido...")
        print(f"   Matriz: {tfidf_matrix.shape}")
        
        # Construir posting lists
        for word_id in range(self.n_words):
            for audio_id in range(self.n_audios):
                score = tfidf_matrix[audio_id, word_id]
                
                if score > min_score:
                    self.index[word_id].append((audio_id, score))
            
            # Ordenar por score (opcional, para eficiencia)
            if len(self.index[word_id]) > 0:
                self.index[word_id].sort(key=lambda x: x[1], reverse=True)
        
        # Calcular normas para similitud coseno
        for audio_id in range(self.n_audios):
            self.audio_norms[audio_id] = np.linalg.norm(tfidf_matrix[audio_id])
        
        # Convertir a dict normal
        self.index = dict(self.index)
        
        # Stats
        avg_postings = np.mean([len(p) for p in self.index.values()])
        print(f"✅ Índice construido:")
        print(f"   - {self.n_words} palabras acústicas")
        print(f"   - {len(self.index)} palabras con postings")
        print(f"   - {avg_postings:.1f} postings promedio")
    
    def get_candidates(self, query_vector: np.ndarray) -> Set[int]:
        """
        Obtiene candidatos usando el índice
        
        Args:
            query_vector: Vector TF-IDF del query
            
        Returns:
            Set de audio_ids candidatos
        """
        candidates = set()
        
        # Solo mirar palabras que aparecen en el query
        for word_id in range(len(query_vector)):
            if query_vector[word_id] > 0 and word_id in self.index:
                for audio_id, _ in self.index[word_id]:
                    candidates.add(audio_id)
        
        return candidates
    
    def search_knn(
        self, 
        query_vector: np.ndarray,
        tfidf_matrix: np.ndarray,
        k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Búsqueda KNN usando el índice
        
        Args:
            query_vector: Vector TF-IDF del query
            tfidf_matrix: Matriz completa de vectores
            k: Top-K resultados
            
        Returns:
            Lista de (audio_id, similarity) ordenada
        """
        # 1. Obtener candidatos
        candidates = self.get_candidates(query_vector)
        
        if len(candidates) == 0:
            return []
        
        # 2. Calcular similitud solo con candidatos
        similarities = []
        query_norm = np.linalg.norm(query_vector)
        
        for audio_id in candidates:
            audio_vector = tfidf_matrix[audio_id]
            
            # Similitud coseno
            dot_product = np.dot(query_vector, audio_vector)
            audio_norm = self.audio_norms[audio_id]
            
            if query_norm > 0 and audio_norm > 0:
                sim = dot_product / (query_norm * audio_norm)
                similarities.append((audio_id, sim))
        
        # 3. Ordenar y retornar top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def save(self, filepath: str) -> None:
        """Guarda el índice en disco"""
        data = {
            'index': self.index,
            'audio_norms': self.audio_norms,
            'n_words': self.n_words,
            'n_audios': self.n_audios
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"✅ Índice guardado: {filepath}")
    
    def load(self, filepath: str) -> None:
        """Carga el índice desde disco"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.index = data['index']
        self.audio_norms = data['audio_norms']
        self.n_words = data['n_words']
        self.n_audios = data['n_audios']
        
        print(f"✅ Índice cargado: {filepath}")
    
    def get_stats(self) -> Dict:
        """Retorna estadísticas del índice"""
        postings_per_word = [len(p) for p in self.index.values()]
        
        return {
            'n_words': self.n_words,
            'n_audios': self.n_audios,
            'non_empty_words': len(self.index),
            'total_postings': sum(postings_per_word),
            'avg_postings': np.mean(postings_per_word) if postings_per_word else 0,
            'max_postings': max(postings_per_word) if postings_per_word else 0
        }


# Para compatibilidad con tu sistema de índices
class AudioInvertedIndex(InvertedIndex):
    """
    Wrapper para compatibilidad con el sistema de índices del Proyecto 1
    """
    
    def __init__(self, data_path: str = None):
        super().__init__()
        self.data_path = data_path
    
    def create_index(self, tfidf_matrix: np.ndarray) -> None:
        """Método compatible con interfaz de índices P1"""
        self.build(tfidf_matrix)
    
    def query(self, query_vector: np.ndarray, k: int = 10) -> List[int]:
        """Método compatible con interfaz de índices P1"""
        candidates = self.get_candidates(query_vector)
        return list(candidates)[:k]