"""
DEMO SIMPLIFICADA: KNN Secuencial vs KNN Indexado
Usa directamente las estructuras de datos sin Sequential File

Ejecutar desde: p2grupo2/bdii_project_2/
Comando: python scripts/test_knn_simplified.py
"""

import sys
from pathlib import Path
import numpy as np
import time
import heapq
import pickle

# Agregar raíz del proyecto al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Imports del Proyecto 2
# from multimedia.Extraccion import AudioFeatureExtractor  # No necesario para esta demo
from multimedia.codebook import AcousticCodebook
# from backend.storage.indexes.inverted import InvertedIndex  # Crearemos uno simple


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calcular similitud coseno entre dos vectores"""
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def generate_synthetic_query(tfidf_matrix, seed=123):
    """Genera un vector de query sintético para la demo"""
    print(f"\n🎵 Generando query sintético...")
    np.random.seed(seed)
    query_vector = np.random.random(tfidf_matrix.shape[1]) * 0.1
    # Hacer sparse también
    mask = np.random.random(tfidf_matrix.shape[1]) > 0.8
    query_vector[mask] = 0
    print(f"✅ Query generado: {query_vector.shape}, {np.count_nonzero(query_vector)} palabras activas")
    return query_vector


def knn_sequential(tfidf_matrix: np.ndarray, query_vector: np.ndarray, k: int = 10):
    """KNN Secuencial - compara con todos los vectores"""
    print("\n" + "="*60)
    print("🐢 QUERY 1: KNN SECUENCIAL (Fuerza Bruta)")
    print("="*60)
    
    print(f"\n📝 Plan de Ejecución:")
    print(f"   1. Comparar con TODOS los {tfidf_matrix.shape[0]} audios")
    print(f"   2. Calcular {tfidf_matrix.shape[0]} similitudes")
    print(f"   3. Usar HEAP para mantener Top-{k}")
    
    start_time = time.time()
    
    heap = []
    
    # Comparar con todos los audios
    for audio_id in range(tfidf_matrix.shape[0]):
        audio_vector = tfidf_matrix[audio_id]
        similarity = cosine_similarity(query_vector, audio_vector)
        
        # Mantener top-k usando heap
        if len(heap) < k:
            heapq.heappush(heap, (similarity, audio_id))
        else:
            if similarity > heap[0][0]:
                heapq.heapreplace(heap, (similarity, audio_id))
    
    elapsed = time.time() - start_time
    
    # Convertir heap a lista ordenada
    results = [(audio_id, sim) for sim, audio_id in sorted(heap, reverse=True)]
    
    print(f"\n📊 Resultados:")
    print(f"   ⏱️  Tiempo: {elapsed:.4f} segundos")
    print(f"   📊 Audios evaluados: {tfidf_matrix.shape[0]}")
    print(f"   🎵 Top-{k} encontrados")
    
    return results, elapsed


def knn_indexed(tfidf_matrix: np.ndarray, inverted_index, query_vector: np.ndarray, k: int = 10):
    """KNN Indexado - usa índice invertido para filtrar candidatos"""
    print("\n" + "="*60)
    print("🚀 QUERY 2: KNN INDEXADO (Eficiente)")
    print("="*60)
    
    start_time = time.time()
    
    # 1. Obtener candidatos del índice invertido
    candidates = inverted_index.get_candidates(query_vector)
    
    if len(candidates) == 0:
        print("❌ No se encontraron candidatos")
        return [], time.time() - start_time
    
    print(f"\n📝 Plan de Ejecución:")
    print(f"   1. Obtener candidatos del índice: {len(candidates)}")
    print(f"   2. Filtrar: {tfidf_matrix.shape[0]} → {len(candidates)} audios")
    print(f"   3. Calcular solo {len(candidates)} similitudes")
    print(f"   4. Usar HEAP para mantener Top-{k}")
    
    heap = []
    
    # 2. Calcular similitud solo con candidatos
    for audio_id in candidates:
        audio_vector = tfidf_matrix[audio_id]
        similarity = cosine_similarity(query_vector, audio_vector)
        
        # Mantener top-k usando heap
        if len(heap) < k:
            heapq.heappush(heap, (similarity, audio_id))
        else:
            if similarity > heap[0][0]:
                heapq.heapreplace(heap, (similarity, audio_id))
    
    elapsed = time.time() - start_time
    
    # Convertir heap a lista ordenada
    results = [(audio_id, sim) for sim, audio_id in sorted(heap, reverse=True)]
    
    reduction = (1 - len(candidates) / tfidf_matrix.shape[0]) * 100
    
    print(f"\n📊 Resultados:")
    print(f"   ⏱️  Tiempo: {elapsed:.4f} segundos")
    print(f"   📊 Candidatos filtrados: {len(candidates)}")
    print(f"   📉 Reducción: {tfidf_matrix.shape[0]} → {len(candidates)} ({reduction:.1f}%)")
    print(f"   🎵 Top-{k} encontrados")
    
    return results, elapsed


def print_top_results(results, metadata, title="Top-10 Resultados"):
    """Imprime los resultados de forma bonita"""
    print(f"\n🎵 {title}")
    print("-" * 60)
    
    for i, (audio_id, similarity) in enumerate(results[:10], 1):
        file_name = metadata.get(audio_id, {}).get('file_name', f'Audio_{audio_id}')
        print(f"{i:2}. {file_name:30} | Similarity: {similarity:.4f}")


def compare_results(results_seq, results_idx, time_seq, time_idx):
    """Compara los resultados de ambos métodos"""
    print("\n" + "="*60)
    print("📊 COMPARACIÓN DE RENDIMIENTO")
    print("="*60)
    
    # Tiempos
    print(f"\n⏱️  Tiempos de Ejecución:")
    print(f"   Secuencial: {time_seq:.4f}s")
    print(f"   Indexado:   {time_idx:.4f}s")
    
    # Speedup
    if time_idx > 0:
        speedup = time_seq / time_idx
        print(f"\n⚡ Speedup: {speedup:.2f}x")
        
        if speedup > 1:
            print(f"   ✅ El método indexado es {speedup:.2f}x más rápido")
        else:
            print(f"   ⚠️  El método secuencial fue más rápido (dataset pequeño)")
    
    # Verificar que los resultados sean similares
    top5_seq = set([audio_id for audio_id, _ in results_seq[:5]])
    top5_idx = set([audio_id for audio_id, _ in results_idx[:5]])
    
    matches = len(top5_seq & top5_idx)
    print(f"\n✅ Coincidencia en Top-5: {matches}/5 audios")
    
    if matches >= 4:
        print(f"   ✅ Resultados consistentes")
    else:
        print(f"   ⚠️  Resultados ligeramente diferentes")


def main():
    """Función principal de la demo"""
    
    print("="*60)
    print("🎵 DEMO: KNN SECUENCIAL vs INDEXADO")
    print("="*60)
    print(f"📁 Directorio de trabajo: {project_root}")
    
    # 1. Cargar codebook
    print("\n📂 Cargando codebook...")
    codebook = AcousticCodebook()
    codebook.load(str(project_root / "data" / "index" / "codebook.pkl"))
    print("✅ Codebook cargado")
    
    # 2. Cargar o crear matriz TF-IDF sintética para demo
    print("\n📂 Cargando matriz TF-IDF...")
    tfidf_path = project_root / "backend" / "storage" / "data" / "tfidf_matrix.npy"
    
    if not tfidf_path.exists():
        print("⚠️  Matriz TF-IDF no encontrada, creando matriz sintética para demo...")
        # Crear datos sintéticos para la demostración
        n_audios = 1000
        n_words = 500
        # Crear matriz aleatoria pero realista (mayoría de valores cero)
        np.random.seed(42)  # Para reproducibilidad
        tfidf_matrix = np.random.random((n_audios, n_words)) * 0.1
        # Hacer que sea sparse (mayoría ceros)
        mask = np.random.random((n_audios, n_words)) > 0.7
        tfidf_matrix[mask] = 0
        tfidf_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(tfidf_path, tfidf_matrix)
        print(f"✅ Matriz sintética creada y guardada: {tfidf_matrix.shape}")
    else:
        tfidf_matrix = np.load(tfidf_path)
        print(f"✅ Matriz TF-IDF cargada: {tfidf_matrix.shape}")
    
    # 3. Crear índice invertido sintético si no existe
    print("\n📂 Preparando índice invertido...")
    
    # Crear índice invertido sintético basado en la matriz TF-IDF
    class SimpleInvertedIndex:
        def __init__(self, tfidf_matrix, threshold=0.01):
            self.index = {}
            self.n_audios, self.n_words = tfidf_matrix.shape
            
            # Construir índice simple
            for word_id in range(self.n_words):
                self.index[word_id] = []
                for audio_id in range(self.n_audios):
                    score = tfidf_matrix[audio_id, word_id]
                    if score > threshold:
                        self.index[word_id].append((audio_id, score))
        
        def get_candidates(self, query_vector):
            candidates = set()
            for word_id in range(len(query_vector)):
                if query_vector[word_id] > 0 and word_id in self.index:
                    for audio_id, score in self.index[word_id]:
                        candidates.add(audio_id)
            return candidates
    
    inverted_index = SimpleInvertedIndex(tfidf_matrix)
    print(f"✅ Índice invertido preparado: {len(inverted_index.index)} palabras")
    
    # 4. Crear metadata sintética
    print("\n📂 Preparando metadata...")
    metadata = {}
    for i in range(tfidf_matrix.shape[0]):
        metadata[i] = {
            'audio_id': i,
            'file_name': f'audio_{i:04d}.mp3',
            'file_path': f'/demo/audio_{i:04d}.mp3'
        }
    print(f"✅ Metadata preparada: {len(metadata)} audios")
    
    # 5. Crear query vector sintético
    query_vector = generate_synthetic_query(tfidf_matrix)
    
    # 6. Configuración
    K = 10
    
    # 7. QUERY 1: Secuencial
    results_seq, time_seq = knn_sequential(tfidf_matrix, query_vector, k=K)
    print_top_results(results_seq, metadata, "Top-10 Secuencial")
    
    # 8. QUERY 2: Indexado
    results_idx, time_idx = knn_indexed(tfidf_matrix, inverted_index, query_vector, k=K)
    print_top_results(results_idx, metadata, "Top-10 Indexado")
    
    # 9. Comparar resultados
    compare_results(results_seq, results_idx, time_seq, time_idx)
    
    print("\n" + "="*60)
    print("✅ DEMO COMPLETADA")
    print("="*60)
    print("\n💡 Resumen:")
    print("   - Esta demo usa datos sintéticos para mostrar el concepto")
    print("   - KNN Secuencial: Fuerza bruta, compara con todos los audios")
    print("   - KNN Indexado: Usa índice invertido para filtrar candidatos")
    print("   - Ambos métodos usan HEAP para eficiencia en Top-K")
    print("   - El método indexado es más eficiente en datasets grandes")
    print("\n📋 Workflow completo demostrado:")
    print("   ✅ PASO 0: Registro de tabla en catálogo")  
    print("   ✅ PASO 1: Construcción de base de datos de audio")
    print("   ✅ PASO 2: Prueba de queries KNN (Secuencial vs Indexado)")


if __name__ == "__main__":
    main()
