"""
Módulo de Audio para integración con Proyecto 1
backend/storage/audio.py

Usa Sequential File del P1 (con catálogo) + Índice Invertido del P2
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import time
import heapq

# Importar Sequential File del P1
from backend.storage.indexes.sequential import SeqFile

# Importar índice invertido del P2
from backend.storage.indexes.inverted import InvertedIndex

# Importar catálogo para obtener rutas
from backend.catalog.catalog import get_filename


class AudioRecord:
    """Registro de audio para Sequential File"""
    
    def __init__(
        self,
        audio_id: int,
        file_name: str,
        file_path: str,
        tfidf_vector: np.ndarray
    ):
        self.audio_id = audio_id
        self.file_name = file_name
        self.file_path = file_path
        self.tfidf_vector = tfidf_vector
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario para Sequential File"""
        return {
            'audio_id': self.audio_id,
            'file_name': self.file_name[:50],  # Truncar a 50 caracteres
            'file_path': self.file_path[:200],  # Truncar a 200 caracteres
            'tfidf_vector': self.tfidf_vector,  # Array numpy directo - ahora soportado nativamente
            'deleted': False  # Campo requerido por Sequential File
        }
    
    @staticmethod
    def from_dict(data: Dict) -> 'AudioRecord':
        """Crea desde diccionario"""
        return AudioRecord(
            audio_id=data['audio_id'],
            file_name=data['file_name'],
            file_path=data['file_path'],
            tfidf_vector=data['tfidf_vector']  # Array numpy directo - ya no necesita deserialización
        )


class AudioStorage:
    """
    Sistema de almacenamiento de audio
    Integra Sequential File (P1) con Índice Invertido (P2)
    """
    
    def __init__(self, table_name: str = "audios"):
        """
        Inicializa el storage usando el catálogo del P1
        
        Args:
            table_name: Nombre de la tabla registrada en el catálogo
        """
        # Obtener ruta del archivo desde el catálogo
        self.data_file = Path(get_filename(table_name))
        
        # Verificar que el archivo existe
        if not self.data_file.exists():
            raise FileNotFoundError(
                f"Archivo no encontrado: {self.data_file}\n"
                f"   Ejecuta primero: python scripts.py/register_table_catalog.py"
            )
        
        # Inicializar Sequential File del P1
        # El schema se lee del catálogo automáticamente
        try:
            self.sequential_file = SeqFile(str(self.data_file))
            print(f"✅ Sequential File inicializado: {self.data_file}")
        except Exception as e:
            raise RuntimeError(
                f"Error al inicializar Sequential File: {e}\n"
                f"   Verifica que la tabla esté registrada en el catálogo\n"
                f"   Ejecuta: python scripts.py/register_table_catalog.py"
            )
        
        # Índice Invertido para búsqueda rápida
        self.inverted_index = InvertedIndex()
        
        # Caché en memoria para eficiencia
        self._vector_cache = {}
        self._metadata_cache = {}
    
    def clear(self) -> None:
        """Limpia el cache y fuerza recarga desde Sequential File"""
        self._vector_cache.clear()
        self._metadata_cache.clear()
        print("✅ Cache limpiado")
    
    def insert(self, audio_record: AudioRecord) -> bool:
        """
        Inserta un audio en el Sequential File
        
        Args:
            audio_record: Registro de audio
            
        Returns:
            True si se insertó correctamente
        """
        try:
            # Preparar registro para Sequential File
            record_dict = audio_record.to_dict()
            
            # Preparar parámetros adicionales para Sequential File
            additional = {
                "key": "audio_id",  # Campo clave principal
                "unique": ["audio_id"]  # Campos únicos 
            }
            
            # Insertar en Sequential File (P1)
            self.sequential_file.insert(record_dict, additional)
            
            # Actualizar caché
            self._vector_cache[audio_record.audio_id] = audio_record.tfidf_vector
            self._metadata_cache[audio_record.audio_id] = {
                'file_name': audio_record.file_name,
                'file_path': audio_record.file_path
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Error al insertar audio {audio_record.audio_id}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def search(self, audio_id: int) -> AudioRecord:
        """
        Busca un audio por ID en el Sequential File
        
        Args:
            audio_id: ID del audio
            
        Returns:
            AudioRecord o None
        """
        try:
            # Buscar en Sequential File (P1)
            data = self.sequential_file.search(audio_id)
            
            if data:
                return AudioRecord.from_dict(data)
            return None
            
        except Exception as e:
            print(f"❌ Error al buscar audio {audio_id}: {e}")
            return None
    
    def scan(self):
        """Itera sobre todos los registros del Sequential File"""
        # Usar el método get_all del Sequential File (P1)
        all_records = self.sequential_file.get_all()
        
        for record_dict in all_records:
            # Agregar el campo deleted si no existe (get_all lo elimina)
            record_dict['deleted'] = False
            audio_record = AudioRecord.from_dict(record_dict)
            yield audio_record
    
    def build_inverted_index(self) -> None:
        """
        Construye el índice invertido a partir del Sequential File
        """
        print("🔨 Construyendo índice invertido desde Sequential File...")
        
        # 1. Leer todos los vectores del Sequential File
        vectors = []
        audio_ids = []
        
        for record in self.scan():
            vectors.append(record.tfidf_vector)
            audio_ids.append(record.audio_id)
            
            # Actualizar caché
            self._vector_cache[record.audio_id] = record.tfidf_vector
            self._metadata_cache[record.audio_id] = {
                'file_name': record.file_name,
                'file_path': record.file_path
            }
        
        print(f"   Leídos {len(vectors)} vectores del Sequential File")
        
        # 2. Construir índice
        tfidf_matrix = np.array(vectors)
        self.inverted_index.build(tfidf_matrix)
        
        # 3. Guardar índice (en directorio backend/storage/data)
        index_path = Path(__file__).parent.parent.parent / "backend" / "storage" / "data" / "inverted_index.pkl"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        self.inverted_index.save(str(index_path))
        
        print(f"✅ Índice invertido guardado: {index_path}")
    
    def load_inverted_index(self) -> None:
        """Carga el índice invertido desde disco"""
        index_path = Path(__file__).parent.parent.parent / "backend" / "storage" / "data" / "inverted_index.pkl"
        
        if not index_path.exists():
            raise FileNotFoundError(f"Índice no encontrado: {index_path}")
        
        self.inverted_index.load(str(index_path))
        
        # Cargar caché desde Sequential File
        record_count = 0
        for record in self.scan():
            self._vector_cache[record.audio_id] = record.tfidf_vector
            self._metadata_cache[record.audio_id] = {
                'file_name': record.file_name,
                'file_path': record.file_path
            }
            record_count += 1
        
        print(f"✅ Índice cargado: {str(index_path)}")
        print(f"✅ Sistema cargado: {record_count} audios")
    
    def knn_sequential(
        self, 
        query_vector: np.ndarray, 
        k: int = 10
    ) -> Tuple[List[Dict], float]:
        """
        Búsqueda KNN Secuencial (Lee TODO el Sequential File)
        Optimizado con HEAP para eficiencia de memoria
        
        SQL Equivalente:
        SELECT * FROM Audios 
        WHERE audio_sim_sequential <-> query 
        LIMIT k;
        """
        start_time = time.time()
        
        # Usar heap para mantener top-k eficientemente
        heap = []
        
        # Leer TODOS los registros del Sequential File
        for record in self.scan():
            sim = self._cosine_similarity(query_vector, record.tfidf_vector)
            
            # Mantener top-k usando heap (min-heap)
            if len(heap) < k:
                heapq.heappush(heap, (sim, record.audio_id))
            elif sim > heap[0][0]:  # Si la similitud es mejor que la peor del heap
                heapq.heapreplace(heap, (sim, record.audio_id))
        
        elapsed = time.time() - start_time
        
        # Convertir heap a lista ordenada (descendente por similitud)
        top_k = [(audio_id, sim) for sim, audio_id in sorted(heap, reverse=True)]
        
        # Enriquecer con metadata
        results = []
        for audio_id, similarity in top_k:
            results.append({
                'audio_id': audio_id,
                'similarity': similarity,
                **self._metadata_cache[audio_id]
            })
        
        return results, elapsed
    
    def knn_indexed(
        self, 
        query_vector: np.ndarray, 
        k: int = 10
    ) -> Tuple[List[Dict], float]:
        """
        Búsqueda KNN Indexada (Usa Índice Invertido + Sequential File)
        Optimizado con HEAP para eficiencia de memoria
        
        SQL Equivalente:
        SELECT * FROM Audios 
        WHERE audio_sim_indexed <-> query 
        LIMIT k;
        """
        start_time = time.time()
        
        # 1. Obtener candidatos del índice invertido (P2)
        candidates = self.inverted_index.get_candidates(query_vector)
        
        # 2. Usar heap para mantener top-k eficientemente
        heap = []
        
        # 3. Calcular similitud solo con candidatos
        for audio_id in candidates:
            if audio_id in self._vector_cache:
                vector = self._vector_cache[audio_id]
                sim = self._cosine_similarity(query_vector, vector)
                
                # Mantener top-k usando heap (min-heap)
                if len(heap) < k:
                    heapq.heappush(heap, (sim, audio_id))
                elif sim > heap[0][0]:  # Si la similitud es mejor que la peor del heap
                    heapq.heapreplace(heap, (sim, audio_id))
        
        elapsed = time.time() - start_time
        
        # 4. Convertir heap a lista ordenada (descendente por similitud)
        top_k = [(audio_id, sim) for sim, audio_id in sorted(heap, reverse=True)]
        
        # 5. Enriquecer con metadata
        results = []
        for audio_id, similarity in top_k:
            results.append({
                'audio_id': audio_id,
                'similarity': similarity,
                **self._metadata_cache[audio_id]
            })
        
        return results, elapsed
    
    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calcula similitud coseno"""
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot / (norm1 * norm2)
    
    def get_stats(self) -> Dict:
        """Obtiene estadísticas del sistema"""
        # Contar audios desde Sequential File si cache está vacío
        n_audios = len(self._vector_cache)
        vector_dimension = None
        
        if self._vector_cache:
            # Si hay cache, usar la dimensión de los vectores cached
            vector_dimension = len(next(iter(self._vector_cache.values())))
        else:
            # Si no hay cache, leer un registro para obtener la dimensión real
            try:
                first_record = next(self.scan())
                vector_dimension = len(first_record.tfidf_vector)
                n_audios = sum(1 for _ in self.scan()) + 1  # +1 por el primer registro
            except StopIteration:
                vector_dimension = 0
                n_audios = 0
        
        return {
            'n_audios': n_audios,
            'vector_dimension': vector_dimension,
            'index_stats': self.inverted_index.get_stats() if hasattr(self.inverted_index, 'index') else {}
        }