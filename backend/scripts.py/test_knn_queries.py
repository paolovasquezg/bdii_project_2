"""
PRUEBA DE QUERIES: KNN Secuencial vs KNN Indexado
Integrado en bdii_project_2

Ejecutar desde: p2grupo2/bdii_project_2/
Comando: python scripts/test_knn_queries.py
"""

import sys
from pathlib import Path
import numpy as np
import time

# Agregar raíz del proyecto al path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Imports del Proyecto 2 (multimedia)
from backend.multimedia.Extraccion import AudioFeatureExtractor
from backend.multimedia.codebook import AcousticCodebook

# Imports del Proyecto 1 (backend)
from backend.storage.audio import AudioStorage


def process_query_audio(query_path: str, codebook: AcousticCodebook) -> np.ndarray:
    """Procesa un audio de query y lo convierte a vector TF-IDF"""
    print(f"\n🎵 Procesando query: {Path(query_path).name}")
    
    # Extraer MFCC
    extractor = AudioFeatureExtractor()
    descriptors = extractor.extract_mfcc(query_path)
    
    if len(descriptors) == 0:
        raise ValueError("No se pudieron extraer características")
    
    # Convertir a histograma
    histogram = codebook.audio_to_histogram(descriptors)
    
    # Para esta demo, usaremos el histograma normalizado directamente
    # En un sistema completo necesitaríamos el TF-IDF transformer guardado
    query_vector = histogram
    
    print(f"✅ Vector generado: {query_vector.shape}")
    
    return query_vector


def test_knn_sequential(storage: AudioStorage, query_vector: np.ndarray, k: int = 10):
    """
    QUERY 1: KNN Secuencial (Fuerza Bruta)
    
    SQL Equivalente:
    SELECT * FROM Audios 
    WHERE audio_sim_sequential <-> query_vector 
    LIMIT k;
    """
    print("\n" + "="*60)
    print("🐢 QUERY 1: KNN SECUENCIAL (Fuerza Bruta)")
    print("="*60)
    
    print(f"\n📝 SQL Equivalente:")
    print(f"   SELECT * FROM Audios")
    print(f"   WHERE audio_sim_sequential <-> query_vector")
    print(f"   LIMIT {k};")
    
    print(f"\n🔍 Plan de Ejecución:")
    print(f"   1. Ignorar índices ❌")
    print(f"   2. Abrir Sequential File 📖")
    print(f"   3. Leer TODOS los registros 🔁")
    print(f"   4. Calcular similitud con cada uno")
    print(f"   5. Usar HEAP para mantener Top-{k} 🏆")
    
    # Ejecutar
    results, elapsed = storage.knn_sequential(query_vector, k=k)
    
    # Resultados
    stats = storage.get_stats()
    print(f"\n📊 Resultados:")
    print(f"   ⏱️  Tiempo: {elapsed:.4f} segundos")
    print(f"   📊 Audios evaluados: {stats['n_audios']}")
    print(f"   🎵 Resultados: {len(results)}")
    
    return results, elapsed


def test_knn_indexed(storage: AudioStorage, query_vector: np.ndarray, k: int = 10):
    """
    QUERY 2: KNN Indexado (Eficiente)
    
    SQL Equivalente:
    SELECT * FROM Audios 
    WHERE audio_sim_indexed <-> query_vector 
    LIMIT k;
    """
    print("\n" + "="*60)
    print("🚀 QUERY 2: KNN INDEXADO (Eficiente)")
    print("="*60)
    
    print(f"\n📝 SQL Equivalente:")
    print(f"   SELECT * FROM Audios")
    print(f"   WHERE audio_sim_indexed <-> query_vector")
    print(f"   LIMIT {k};")
    
    print(f"\n🔍 Plan de Ejecución:")
    print(f"   1. Usar Índice Invertido ✅")
    print(f"   2. Obtener candidatos 🎯")
    print(f"   3. Leer solo candidatos del Sequential File 📖")
    print(f"   4. Calcular similitud con candidatos")
    print(f"   5. Usar HEAP para mantener Top-{k} 🏆")
    
    # Ejecutar
    results, elapsed = storage.knn_indexed(query_vector, k=k)
    
    # Calcular candidatos
    candidates = storage.inverted_index.get_candidates(query_vector)
    stats = storage.get_stats()
    reduction = (1 - len(candidates) / stats['n_audios']) * 100
    
    # Resultados
    print(f"\n📊 Resultados:")
    print(f"   ⏱️  Tiempo: {elapsed:.4f} segundos")
    print(f"   📊 Candidatos filtrados: {len(candidates)}")
    print(f"   📊 Reducción: {stats['n_audios']} → {len(candidates)} ({reduction:.1f}%)")
    print(f"   🎵 Resultados: {len(results)}")
    
    return results, elapsed


def compare_results(results_seq, results_idx, time_seq, time_idx):
    """Compara los resultados de ambas queries"""
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
        print(f"\n⚡ Speedup: {speedup:.2f}x más rápido")
        
        if speedup > 1:
            print(f"   ✅ El método indexado es {speedup:.2f}x más rápido")
        else:
            print(f"   ⚠️  El método secuencial fue más rápido (dataset pequeño)")
    
    # Verificar que los resultados sean similares
    top5_seq = set([r['audio_id'] for r in results_seq[:5]])
    top5_idx = set([r['audio_id'] for r in results_idx[:5]])
    
    matches = len(top5_seq & top5_idx)
    print(f"\n✅ Coincidencia en Top-5: {matches}/5 audios")
    
    if matches >= 4:
        print(f"   ✅ Resultados consistentes")
    else:
        print(f"   ⚠️  Resultados ligeramente diferentes")
    
    # Notas técnicas
    print(f"\n💡 Notas Técnicas:")
    print(f"   - AMBOS métodos usan HEAP para Top-K")
    print(f"   - Diferencia: número de comparaciones")
    print(f"   - Secuencial: Lee TODA la tabla")
    print(f"   - Indexado: Filtra con índice invertido primero")


def print_top_results(results, title="Top-10 Resultados"):
    """Imprime los resultados de forma bonita"""
    print(f"\n🎵 {title}")
    print("-" * 60)
    
    for i, result in enumerate(results[:10], 1):
        print(f"{i:2}. {result['file_name']:20} | Similarity: {result['similarity']:.4f}")


def main():
    """Función principal de prueba"""
    
    print("="*60)
    print("🎵 PRUEBA DE QUERIES: SECUENCIAL VS INDEXADO")
    print("="*60)
    print(f"📁 Directorio de trabajo: {project_root}")
    
    # Verificar archivos necesarios
    required_files = [
        project_root / "data" / "index" / "codebook.pkl",
        project_root / "backend" / "runtime" / "files" / "audios" / "audios.dat",
        project_root / "backend" / "storage" / "data" / "inverted_index.pkl"
    ]
    
    for filepath in required_files:
        if not filepath.exists():
            print(f"❌ Error: No se encontró {filepath}")
            print("\n   Ejecuta primero: python scripts/build_audio_database.py")
            return
    
    print("✅ Todos los archivos necesarios encontrados")
    
    # Cargar codebook
    print("\n📂 Cargando codebook...")
    codebook = AcousticCodebook()
    codebook.load(str(project_root / "data" / "index" / "codebook.pkl"))
    print("✅ Codebook cargado")
    
    # Cargar storage
    print("\n📂 Cargando storage e índice...")
    storage = AudioStorage()
    
    # Limpiar caché por si hay datos de construcciones anteriores
    storage.clear()
    
    # Cargar índice invertido
    storage.load_inverted_index()
    
    stats = storage.get_stats()
    print(f"✅ Sistema cargado:")
    print(f"   - Audios: {stats['n_audios']}")
    print(f"   - Dimensión: {stats['vector_dimension']}")
    
    # Obtener query audio - EXPERIMENTO CON AUDIO ESPECÍFICO  
    fma_dir = project_root / "data" / "fma_small"
    
    # Buscar un audio específico que sabemos que está en la BD
    target_audio = "000821.mp3"  # Audio que siempre aparece en Top-1
    audio_files = list(fma_dir.rglob(f"*{target_audio}"))
    
    if len(audio_files) == 0:
        print(f"❌ No se encontró {target_audio}, usando audio aleatorio")
        audio_files = list(fma_dir.rglob("*.mp3"))
        query_path = str(audio_files[0])
    else:
        query_path = str(audio_files[0])
    
    print(f"🎵 Procesando query: {Path(query_path).name}")
    
    # Procesar query
    query_vector = process_query_audio(query_path, codebook)
    
    # Configuración
    K = 10
    
    print(f"✅ Vector generado: {query_vector.shape}")
    
    # QUERY 1: Secuencial
    results_seq, time_seq = test_knn_sequential(storage, query_vector, k=K)
    print_top_results(results_seq, "Top-10 Secuencial")
    
    # QUERY 2: Indexado
    results_idx, time_idx = test_knn_indexed(storage, query_vector, k=K)
    print_top_results(results_idx, "Top-10 Indexado")
    
    # Comparar
    compare_results(results_seq, results_idx, time_seq, time_idx)
    
    print("\n" + "="*60)
    print("✅ PRUEBA COMPLETADA")
    print("="*60)


if __name__ == "__main__":
    main()