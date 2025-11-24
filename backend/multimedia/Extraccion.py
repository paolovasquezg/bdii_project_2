import librosa 
import numpy as np 
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Dict
import warnings
import sys
# Agregar scripts.py al path para los imports de utils
sys.path.append(str(Path(__file__).parent.parent / "scripts.py"))
from utils import DATA_DIR, PROCESSED_DIR, save_pickle, load_pickle, get_audio_files, Timer, FMA_DIR
warnings.filterwarnings("ignore")

class AudioFeatureExtractor:
    def __init__(self, sr: int = 22050, n_mfcc: int = 13, n_fft: int = 2048, hop_length: int = 512, duration: int = 30):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.duration = duration
    
    def extract_mfcc(self, audio_path: str) -> np.ndarray:
        try:
            y, sr = librosa.load(audio_path, sr=self.sr, duration=self.duration, mono=True)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length)
            descriptors = mfcc.T  
            return descriptors
        except Exception as e:
            print(f"Error procesando el {audio_path}: {e}")
            return np.array([])
    
    def extract_dataset(self, audio_dir: str, max_files: int = None, save_individual: bool = True) -> Tuple[Dict, List[np.ndarray]]:
        print(f"Extrayendo caracteristicas de {audio_dir}")
        audio_files = get_audio_files(audio_dir)

        if max_files:
            audio_files = audio_files[:max_files]
        
        print(f"Se encontraron {len(audio_files)} archivos")

        metadata = {}
        all_descriptors = []

        for idx, audio_path in enumerate(tqdm(audio_files, desc= "Extrayendo MFCC")):
            descriptors = self.extract_mfcc(str(audio_path))
            if len(descriptors) >  0:
                # guardamos las metadata
                metadata[idx] = {
                    'id_audio': idx,
                    'file_path': str(audio_path),
                    'file_name': audio_path.name,
                    'num_descriptors': len(descriptors)
                }
                all_descriptors.append(descriptors)

                if save_individual:
                    desc_path = PROCESSED_DIR / f"descriptors_{idx}.npy"
                    np.save(desc_path, descriptors)

        print(f"Procesados: {len(all_descriptors)} audios")
        print(f"Descriptores totales: {sum(len(d) for d in all_descriptors)}")

        return metadata, all_descriptors
    


def main():
    max_files = 1000
    extractor = AudioFeatureExtractor(sr=22050, n_mfcc=13, duration=30)

    with Timer("Extraccion Total"):
        metadata, all_descriptors = extractor.extract_dataset(
            audio_dir= str(FMA_DIR),
            max_files=max_files,
            save_individual=False
        )
    metadata_path = PROCESSED_DIR / "metadata.pkl"
    save_pickle(metadata, metadata_path)

    descriptors_path = PROCESSED_DIR / "all_descriptors.pkl"
    save_pickle(all_descriptors, descriptors_path)

    all_desc_flat = np.vstack(all_descriptors)
    print(f"Descriptores MFCC: {all_desc_flat}")

    flat_path = PROCESSED_DIR / "descriptors_flat.npy"
    np.save(flat_path, all_desc_flat)

    print(f"Guardado: {flat_path} ")
    print("\n" + "=" *60)
    print(f"Archivos generados en {PROCESSED_DIR}")
    print(f" - Metadata: {metadata_path}")
    print(f" - Descriptores: {descriptors_path}")
    print(f" - Descriptores planos: {flat_path}")
    
if __name__ == "__main__":
    main()
