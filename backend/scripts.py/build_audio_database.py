import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Agregar raíz del proyecto al path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Imports del Proyecto 2 (multimedia)
from backend.multimedia.Extraccion import AudioFeatureExtractor
from backend.multimedia.codebook import AcousticCodebook

# Imports del Proyecto 1 (backend)
from backend.storage.audio import AudioStorage, AudioRecord
from backend.storage.indexes.inverted import InvertedIndex


def get_audio_files(directory, extensions=['.mp3', '.wav']):
    audio_files = []
    directory_path = Path(directory)
    
    for ext in extensions:
        audio_files.extend(directory_path.rglob(f"*{ext}"))
    
    return sorted(audio_files)


def extract_all_features(audio_files, max_audios=None):
    print("\n" + "="*60)
    print("PASO 1: EXTRACCIÓN DE CARACTERÍSTICAS")
    print("="*60)
    
    if max_audios:
        audio_files = audio_files[:max_audios]
    
    print(f"Procesando {len(audio_files)} audios...")
    
    extractor = AudioFeatureExtractor(sr=22050, n_mfcc=13, duration=30)
    
    all_descriptors = []
    metadata = {}
    audio_id_counter = 0  # Contador separado para audios válidos
    
    for idx, audio_path in enumerate(tqdm(audio_files, desc="Extrayendo MFCC")):
        descriptors = extractor.extract_mfcc(str(audio_path))
        
        if len(descriptors) > 0:
            all_descriptors.append(descriptors)
            metadata[audio_id_counter] = {  # Usar el contador de audios válidos
                'audio_id': audio_id_counter,
                'file_path': str(audio_path),
                'file_name': audio_path.name
            }
            audio_id_counter += 1  # Incrementar solo cuando el audio es válido
    
    print(f"Procesados: {len(all_descriptors)} audios")
    print(f"Descriptores totales: {sum(len(d) for d in all_descriptors)}")
    
    return all_descriptors, metadata


def train_codebook(all_descriptors, k=500, demo_mode=True):
    print("\n" + "="*60)
    print("PASO 2: ENTRENAMIENTO DEL CODEBOOK")
    print("="*60)
    
    if demo_mode:
        print(f" MODO DEMO: K-Means custom optimizado")
        print(f"   Entrenando K-Means con K={k} (modo demo)...")
    else:
        print(f"Entrenando K-Means con K={k}...")
    
    # Concatenar todos los descriptores
    descriptors_flat = np.vstack(all_descriptors)
    print(f"Shape total: {descriptors_flat.shape}")
    
    # Entrenar K-Means (con modo demo)
    codebook = AcousticCodebook(n_clusters=k, demo_mode=demo_mode)
    codebook.build_codebook(descriptors_flat)
    
    # Guardar codebook
    codebook_path = project_root / "data" / "index" / "codebook.pkl"
    codebook_path.parent.mkdir(parents=True, exist_ok=True)
    codebook.save(codebook_path)
    
    print(f" Codebook guardado: {codebook_path}")
    
    return codebook


def convert_to_tfidf(all_descriptors, codebook):
    print("\n" + "="*60)
    print("PASO 3: CONVERSIÓN A TF-IDF")
    print("="*60)
    
    # Convertir a histogramas
    histograms = codebook.convert_dataset_to_histograms(all_descriptors)
    
    # Aplicar TF-IDF
    tfidf_matrix = codebook.apply_tfidf(histograms)
    
    print(f"Matriz TF-IDF: {tfidf_matrix.shape}")
    
    # Guardar matriz TF-IDF para uso posterior
    tfidf_path = Path(__file__).parent.parent / "backend" / "storage" / "data" / "tfidf_matrix.npy"
    tfidf_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(tfidf_path, tfidf_matrix)
    print(f"💾 Matriz TF-IDF guardada: {tfidf_path}")
    
    return tfidf_matrix


def populate_sequential_file(metadata, tfidf_matrix):
    print("\n" + "="*60)
    print("PASO 4: POBLANDO SEQUENTIAL FILE")
    print("="*60)
    
    # Inicializar storage (usa catálogo del P1)
    storage = AudioStorage()
    
    print(f"Insertando {len(metadata)} registros...")
    print(f"Metadata keys: {list(metadata.keys())[:10]}...")  # Debug
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")  # Debug
    
    # Iterar directamente sobre las claves de metadata, no sobre un rango
    for audio_id in tqdm(metadata.keys(), desc="Insertando"):
        if audio_id >= tfidf_matrix.shape[0]:
            print(f" audio_id {audio_id} fuera de rango de tfidf_matrix")
            continue
            
        record = AudioRecord(
            audio_id=audio_id,
            file_name=metadata[audio_id]['file_name'],
            file_path=metadata[audio_id]['file_path'],
            tfidf_vector=tfidf_matrix[audio_id]
        )
        storage.insert(record)
    
    print(f"✅ {len(metadata)} registros insertados en Sequential File")
    
    return storage


def build_inverted_index(tfidf_matrix):
    print("\n" + "="*60)
    print(" PASO 5: CONSTRUYENDO ÍNDICE INVERTIDO")
    print("="*60)
    
    # Construir índice
    inverted_index = InvertedIndex()
    inverted_index.build(tfidf_matrix)
    
    # Guardar índice
    index_path = project_root / "backend" / "storage" / "data" / "inverted_index.pkl"
    inverted_index.save(str(index_path))
    
    # Stats
    stats = inverted_index.get_stats()
    print(f" Índice invertido construido:")
    print(f"   - Palabras: {stats['n_words']}")
    print(f"   - Audios: {stats['n_audios']}")
    print(f"   - Postings: {stats['total_postings']}")
    
    return inverted_index


def main():    
    print("="*60)
    print("🎵 CONSTRUCCIÓN DE BASE DE DATOS DE AUDIO")
    print("="*60)
    print(f" Directorio de trabajo: {project_root}")
    
    # Configuración MÁXIMA CALIDAD (para conseguir Top-1)
    FMA_DIR = project_root / "data" / "fma_small"
    K = 400  # Muchos más clusters para máxima granularidad
    MAX_AUDIOS = 100  # Mantener 100 audios
    DEMO_MODE = True  # Usar configuración optimizada
    
    # Obtener archivos
    print(f"\n Buscando audios en: {FMA_DIR}")
    
    if not FMA_DIR.exists():
        print(f" Error: No se encontró el directorio {FMA_DIR}")
        print(f"\n Solución:")
        print(f"   1. Crea la carpeta: mkdir -p data/fma_small")
        print(f"   2. Coloca tus archivos MP3 en: data/fma_small/")
        return
    
    audio_files = get_audio_files(str(FMA_DIR))
    print(f" Encontrados: {len(audio_files)} archivos")
    
    if len(audio_files) == 0:
        print(" No se encontraron archivos MP3")
        return
    
    # Pipeline completo
    try:
        # PASO 1: Extraer características
        all_descriptors, metadata = extract_all_features(audio_files, MAX_AUDIOS)
        
        # PASO 2: Entrenar codebook
        codebook = train_codebook(all_descriptors, k=K, demo_mode=DEMO_MODE)
        
        # PASO 3: Convertir a TF-IDF
        tfidf_matrix = convert_to_tfidf(all_descriptors, codebook)
        
        # PASO 4: Poblar Sequential File
        storage = populate_sequential_file(metadata, tfidf_matrix)
        
        # Guardar metadata para uso posterior
        metadata_path = project_root / "backend" / "storage" / "data" / "metadata.pkl"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        import pickle
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f" Metadata guardada: {metadata_path}")
        
        # PASO 5: Construir Índice Invertido
        inverted_index = build_inverted_index(tfidf_matrix)
        
        # Resumen final
        print("\n" + "="*60)
        print(" BASE DE DATOS CONSTRUIDA EXITOSAMENTE")
        print("="*60)
        print(f"\n📁 Archivos generados:")
        print(f"   - data/index/codebook.pkl")
        print(f"   - backend/storage/data/audios.dat (Sequential File P1)")
        print(f"   - backend/storage/data/inverted_index.pkl (Índice Invertido P2)")
        
        print(f"\n📊 Resumen:")
        print(f"   - Audios indexados: {len(metadata)}")
        print(f"   - Palabras acústicas: {K}")
        print(f"   - Dimensión vectores: {tfidf_matrix.shape[1]}")
        
        print(f"\n🎯 Siguiente paso:")
        print(f"   python scripts/test_knn_queries.py")
    
    except Exception as e:
        print(f"\n❌ Error durante el procesamiento: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()