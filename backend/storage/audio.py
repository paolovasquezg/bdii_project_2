"""
Módulo de Audio para integración con Proyecto 1
backend/storage/audio.py

Usa Sequential File del P1 (con catálogo) + Índice Invertido del P2
"""

import json
import time
import heapq
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np

# Importar Sequential File del P1
from backend.storage.file import File

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
        vec_list = self.tfidf_vector.tolist() if isinstance(self.tfidf_vector, np.ndarray) else self.tfidf_vector
        vec_str = json.dumps(vec_list)
        return {
            'audio_id': self.audio_id,
            'file_name': self.file_name[:50],  # Truncar a 50 caracteres
            'file_path': self.file_path[:200],  # Truncar a 200 caracteres
            # Guardamos como string JSON para respetar esquema fijo del SeqFile
            'tfidf_vector': vec_str,
            'deleted': False  # Campo requerido por Sequential File
        }
    
    @staticmethod
    def from_dict(data: Dict) -> 'AudioRecord':
        """Crea desde diccionario"""
        raw_vec = data['tfidf_vector']
        if raw_vec in ("", None):
            raw_vec = []
        if isinstance(raw_vec, str):
            try:
                raw_vec = json.loads(raw_vec)
            except Exception:
                raw_vec = []
        return AudioRecord(
            audio_id=data['audio_id'],
            file_name=data['file_name'],
            file_path=data['file_path'],
            tfidf_vector=np.array(raw_vec, dtype=float)  # recrear np.array
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
        # Inspeccionar metadatos vía File wrapper (soporta heap/sequential/etc)
        self.file = File(table_name)
        self.data_file = Path(self.file.filename)

        # No usamos SeqFile directamente; delegamos a File para todos los PK
        self.sequential_file = None
        print(f"ℹ️  Tabla {table_name} usando File wrapper (pk={self.file.indexes.get('primary',{}).get('index')})")
        
        # Índice Invertido para búsqueda rápida
        self.inverted_index = InvertedIndex()

        # Caché en memoria para eficiencia
        self._vector_cache = {}
        self._metadata_cache = {}
        self._id_order: List[int] = []  # pos -> audio_id

        # Ruta de índice en runtime
        self.index_dir = Path(__file__).resolve().parents[1] / "runtime" / "audio" / table_name
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.index_dir / "inverted_index.pkl"
        self.meta_path = self.index_dir / "inverted_index.meta.json"
    
    def clear(self) -> None:
        """Limpia el cache y fuerza recarga desde Sequential File"""
        self._vector_cache.clear()
        self._metadata_cache.clear()
        print("✅ Cache limpiado")

    @staticmethod
    def vector_from_path(path: str | Path, dim: int = 32) -> np.ndarray:
        """Vector TF (pseudo) determinístico a partir de la ruta.
        Usamos sólo el nombre del archivo para que el mismo audio en otra carpeta
        genere el mismo vector (útil para uploads vs. dataset base).
        """
        p = Path(path)
        token = p.name or p.as_posix()
        seed = int.from_bytes(token.encode("utf-8")[:8], "little", signed=False)
        rng = np.random.default_rng(seed)
        return np.abs(rng.standard_normal(dim))
    
    def insert(self, audio_record: AudioRecord) -> bool:
        """
        Inserta un audio en el Sequential File
        
        Args:
            audio_record: Registro de audio
            
        Returns:
            True si se insertó correctamente
        """
        try:
            record_dict = audio_record.to_dict()

            if self.sequential_file:
                additional = {"key": "audio_id", "unique": ["audio_id"]}
                self.sequential_file.insert(record_dict, additional)
            else:
                # File wrapper (respeta índice primario real)
                if "deleted" in record_dict:
                    record_dict = {k: v for k, v in record_dict.items() if k != "deleted"}
                self.file.execute({"op": "insert", "record": record_dict})
            
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
            if self.sequential_file:
                additional = {"key": "audio_id", "value": audio_id, "unique": True}
                data = self.sequential_file.search(additional, same_key=True)
            else:
                data = self.file.search({"op": "search", "field": "audio_id", "value": audio_id})

            if data:
                # SeqFile.search devuelve lista de dicts
                first = data[0] if isinstance(data, list) else data
                if isinstance(first, tuple) and len(first) >= 1:
                    first = first[0]
                if isinstance(first, dict):
                    return AudioRecord.from_dict(first)
            return None

        except Exception as e:
            print(f"❌ Error al buscar audio {audio_id}: {e}")
            return None
    
    def scan(self):
        """Itera sobre todos los registros del Sequential File"""
        if self.sequential_file:
            all_records = self.sequential_file.get_all()
        else:
            all_records = self.file.get_all()
        
        for rec in all_records:
            record_dict = rec
            if isinstance(rec, tuple) and len(rec) >= 1:
                record_dict = rec[0]
            if not isinstance(record_dict, dict) and hasattr(record_dict, "fields"):
                record_dict = getattr(record_dict, "fields", record_dict)
            if not isinstance(record_dict, dict):
                continue
            record_dict['deleted'] = False
            try:
                audio_record = AudioRecord.from_dict(record_dict)
            except Exception:
                continue
            # refresca caches para uso inmediato en kNN secuencial
            self._vector_cache[audio_record.audio_id] = audio_record.tfidf_vector
            self._metadata_cache[audio_record.audio_id] = {
                'file_name': audio_record.file_name,
                'file_path': audio_record.file_path
            }
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
            # Recalcular vector para asegurar consistencia (nombre/archivo vs ruta)
            try:
                dim = len(record.tfidf_vector) if hasattr(record.tfidf_vector, "__len__") else 32
                if not dim or dim <= 0:
                    dim = 32
                vec = self.vector_from_path(record.file_path, dim=dim)
                record.tfidf_vector = vec
            except Exception:
                vec = record.tfidf_vector

            vectors.append(vec)
            audio_ids.append(record.audio_id)

            # Actualizar caché
            self._vector_cache[record.audio_id] = vec
            self._metadata_cache[record.audio_id] = {
                'file_name': record.file_name,
                'file_path': record.file_path
            }

        if not vectors:
            print("⚠️  No hay audios para indexar; se crea índice vacío.")
            self._id_order = []
            self.index_dir.mkdir(parents=True, exist_ok=True)
            try:
                with self.meta_path.open("w", encoding="utf-8") as f:
                    json.dump({"id_order": []}, f, ensure_ascii=False)
            except Exception:
                pass
            return

        print(f"   Leídos {len(vectors)} vectores del Sequential File")

        # 2. Construir índice
        tfidf_matrix = np.array(vectors)
        self.inverted_index.build(tfidf_matrix)
        self._id_order = audio_ids

        # 3. Guardar índice + metadatos (runtime/audio)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.inverted_index.save(str(self.index_path))
        try:
            with self.meta_path.open("w", encoding="utf-8") as f:
                json.dump({"id_order": audio_ids}, f, ensure_ascii=False)
        except Exception:
            pass

        print(f"✅ Índice invertido guardado: {self.index_path}")

    def load_inverted_index(self) -> None:
        """Carga el índice invertido desde disco"""
        if not self.index_path.exists():
            raise FileNotFoundError(f"Índice no encontrado: {self.index_path}")

        self.inverted_index.load(str(self.index_path))
        if self.meta_path.exists():
            try:
                meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
                self._id_order = meta.get("id_order") or []
            except Exception:
                self._id_order = []
        
        # Cargar caché desde Sequential File
        record_count = 0
        for record in self.scan():
            vec = record.tfidf_vector
            if not hasattr(vec, "__len__") or len(vec) == 0:
                try:
                    vec = self.vector_from_path(record.file_path, dim=32)
                except Exception:
                    vec = record.tfidf_vector
            self._vector_cache[record.audio_id] = vec
            self._metadata_cache[record.audio_id] = {
                'file_name': record.file_name,
                'file_path': record.file_path
            }
            record_count += 1
        
        print(f"✅ Índice cargado: {str(self.index_path)}")
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
            # asegurar vector coherente
            if not hasattr(record.tfidf_vector, "__len__") or len(record.tfidf_vector) == 0:
                record.tfidf_vector = self.vector_from_path(record.file_path)
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
                **(self._metadata_cache.get(audio_id) or {})
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
        for pos in candidates:
            # Mapear posición a audio_id real si existe
            audio_id = self._id_order[pos] if pos < len(self._id_order) else pos
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
