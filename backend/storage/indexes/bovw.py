import json
import os
import pickle
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np
from PIL import Image

from backend.catalog.settings import DATA_DIR

try:  # joblib es opcional; caemos a pickle si no está
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    joblib = None  # type: ignore


DocRecord = Union[dict, Tuple[dict, int]]


class BoVWFile:
    """
    Índice BoVW simplificado que guarda histogramas normalizados por imagen.
    Artefactos creados en base_dir:
      - codebook.joblib : metadatos mínimos (dimensión)
      - idf.npy         : vector idf
      - postings.json   : {doc_id: hist}
      - doc_map.json    : {doc_id: {"pk":..., "pos":..., "path":...}}
    """

    def __init__(self, base_dir: str, key: str, heap_file: str | None = None):
        base_dir_abs = os.path.abspath(
            os.path.join(str(DATA_DIR), base_dir) if not os.path.isabs(base_dir) else base_dir
        )
        self.base_dir = base_dir_abs
        self.key = key
        self.heap_file = heap_file

        self.read_count = 0
        self.write_count = 0

        self.doc_map: Dict[Union[int, str], dict] = {}
        self.vectors: Dict[Union[int, str], np.ndarray] = {}
        self.idf: np.ndarray | None = None
        self.tfidf_vectors: Dict[Union[int, str], np.ndarray] = {}
        self.inverted: Dict[int, List[Tuple[Union[int, str], float]]] = {}
        self.dim: int = 0
        self._loaded = False

        os.makedirs(self.base_dir, exist_ok=True)
        self._load_artifacts()

    # ------------------------------- helpers ---------------------------------------- #

    def _feature_vector(self, path: str) -> np.ndarray:
        """Histograma de intensidades (32 bins) normalizado."""
        img = Image.open(path).convert("L")
        self.read_count += 1  # contar lectura de imagen
        img = img.resize((64, 64))
        arr = np.asarray(img, dtype=np.float32)
        hist, _ = np.histogram(arr, bins=32, range=(0, 255))
        vec = hist.astype(np.float32)
        total = float(vec.sum())
        if total > 0:
            vec /= total
        return vec

    def _load_artifacts(self):
        if self._loaded:
            return

        doc_map_path = os.path.join(self.base_dir, "doc_map.json")
        if os.path.exists(doc_map_path):
            with open(doc_map_path, "r", encoding="utf-8") as f:
                raw = json.load(f) or {}
                for k, v in raw.items():
                    try:
                        dk: Union[int, str] = int(k)
                    except Exception:
                        dk = k
                    self.doc_map[dk] = v
            self.read_count += 1

        postings_path = os.path.join(self.base_dir, "postings.json")
        if os.path.exists(postings_path):
            with open(postings_path, "r", encoding="utf-8") as f:
                raw = json.load(f) or {}
                for k, v in raw.items():
                    try:
                        dk: Union[int, str] = int(k)
                    except Exception:
                        dk = k
                    self.vectors[dk] = np.asarray(v, dtype=np.float32)
            self.read_count += 1
            if self.vectors and not self.dim:
                sample = next(iter(self.vectors.values()))
                self.dim = int(sample.shape[0])

        idf_path = os.path.join(self.base_dir, "idf.npy")
        if os.path.exists(idf_path):
            try:
                self.idf = np.load(idf_path)
                self.dim = int(self.idf.shape[0])
                self.read_count += 1
            except Exception:
                self.idf = None

        inv_path = os.path.join(self.base_dir, "inverted_index.json")
        if os.path.exists(inv_path):
            try:
                with open(inv_path, "r", encoding="utf-8") as f:
                    raw = json.load(f) or {}
                    self.inverted = {int(k): [(dk if not str(dk).isdigit() else int(dk), float(w)) for dk, w in v]
                                     for k, v in raw.items()}
                self.read_count += 1
            except Exception:
                self.inverted = {}

        self._loaded = bool(self.doc_map or self.vectors)

    def _persist(self):
        os.makedirs(self.base_dir, exist_ok=True)

        # doc_map
        doc_map_path = os.path.join(self.base_dir, "doc_map.json")
        with open(doc_map_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in self.doc_map.items()}, f, ensure_ascii=False)
        self.write_count += 1

        # postings
        postings_path = os.path.join(self.base_dir, "postings.json")
        with open(postings_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v.tolist() for k, v in self.vectors.items()}, f)
        self.write_count += 1

        # idf
        if self.idf is None and self.vectors:
            self._compute_idf()
        if self.idf is not None:
            np.save(os.path.join(self.base_dir, "idf.npy"), self.idf)
            self.write_count += 1

        # codebook stub
        cb_obj = {"dim": int(self.dim or (next(iter(self.vectors.values())).shape[0] if self.vectors else 0))}
        codebook_path = os.path.join(self.base_dir, "codebook.joblib")
        try:
            if joblib:
                joblib.dump(cb_obj, codebook_path)
            else:
                raise RuntimeError()
        except Exception:
            with open(codebook_path, "wb") as f:
                pickle.dump(cb_obj, f)
        self.write_count += 1

        # inverted (opcional para kNN sparse)
        self._ensure_tfidf_and_inverted()
        if self.inverted:
            inv_path = os.path.join(self.base_dir, "inverted_index.json")
            with open(inv_path, "w", encoding="utf-8") as f:
                json.dump({int(k): [(str(d), w) for d, w in v] for k, v in self.inverted.items()}, f)
            self.write_count += 1

        # postings + doc_map se contaron en writes anteriores

    def _compute_idf(self):
        if not self.vectors:
            self.idf = None
            return
        arr = np.vstack(list(self.vectors.values()))
        df = (arr > 0).astype(np.int32).sum(axis=0)
        N = arr.shape[0]
        self.idf = np.log((N + 1) / (df + 1))
        self.dim = int(self.idf.shape[0])

    def _ensure_tfidf_and_inverted(self):
        """Construye tfidf y un índice invertido ligero sobre los clusters."""
        if not self.vectors:
            return
        if self.idf is None:
            self._compute_idf()
        # Evita tfidf nulo cuando df == N en todos los bins (idf=0); usa tf en ese caso.
        idf_eff = self.idf
        if idf_eff is not None and np.allclose(idf_eff, 0):
            idf_eff = np.ones_like(idf_eff)
        self.tfidf_vectors = {}
        self.inverted = {}
        for doc_id, vec in self.vectors.items():
            tfidf = np.array(vec, dtype=float, copy=True)
            if idf_eff is not None and idf_eff.shape[0] == vec.shape[0]:
                tfidf = tfidf * idf_eff
            self.tfidf_vectors[doc_id] = tfidf
            any_w = False
            for idx, w in enumerate(tfidf):
                if w <= 0:
                    continue
                any_w = True
                bucket = self.inverted.setdefault(idx, [])
                bucket.append((doc_id, float(w)))
            # Si tfidf quedó todo en cero (idf=0 en bins usados), usar TF puro.
            if not any_w:
                for idx, w in enumerate(vec):
                    if w <= 0:
                        continue
                    bucket = self.inverted.setdefault(idx, [])
                    bucket.append((doc_id, float(w)))

    def _ensure_loaded(self):
        self._load_artifacts()

    # ------------------------------- API público ------------------------------------ #

    def build_bulk(self, records: Iterable[DocRecord], *, image_field: str, pk_name: str, main_index: str):
        self.doc_map = {}
        self.vectors = {}

        for rec in records:
            if isinstance(rec, tuple) and len(rec) >= 2:
                row, pos = rec[0], rec[1]
            elif isinstance(rec, dict):
                row, pos = rec, rec.get("pos")
            else:
                continue

            if not isinstance(row, dict):
                continue
            if image_field not in row or pk_name not in row:
                continue

            path = str(row.get(image_field) or "").strip()
            if not path or not os.path.exists(path):
                continue

            try:
                vec = self._feature_vector(path)
            except Exception:
                continue

            doc_id: Union[int, str] = row.get(pk_name)
            if doc_id is None:
                continue
            self.doc_map[doc_id] = {"pk": row.get(pk_name), "pos": pos, "path": path}
            self.vectors[doc_id] = vec

        self._compute_idf()
        self._persist()
        self._loaded = True

    def _knn_dense(self, q_vec: np.ndarray, norm_q: float) -> List[Tuple[Union[int, str], float]]:
        scores_list: List[Tuple[Union[int, str], float]] = []
        idf_eff = self.idf
        if idf_eff is not None and np.allclose(idf_eff, 0):
            idf_eff = np.ones_like(idf_eff)
        for doc_id, vec in self.vectors.items():
            dv = vec
            if idf_eff is not None and idf_eff.shape[0] == dv.shape[0]:
                dv = dv * idf_eff
            norm_d = float(np.linalg.norm(dv))
            if norm_d == 0.0:
                dv = vec
                norm_d = float(np.linalg.norm(dv))
                if norm_d == 0.0:
                    continue
            sim = float(np.dot(q_vec, dv) / (norm_q * norm_d))
            scores_list.append((doc_id, sim))
        return scores_list

    def _knn_inverted(self, q_vec: np.ndarray, norm_q: float) -> List[Tuple[Union[int, str], float]]:
        if not self.inverted:
            return []
        scores: Dict[Union[int, str], float] = {}
        nz = np.nonzero(q_vec)[0]
        for idx in nz:
            wq = float(q_vec[idx])
            for doc_id, wd in self.inverted.get(int(idx), []):
                scores[doc_id] = scores.get(doc_id, 0.0) + wq * float(wd)
        scores_list: List[Tuple[Union[int, str], float]] = []
        for doc_id, dot in scores.items():
            tfidf_vec = self.tfidf_vectors.get(doc_id)
            if tfidf_vec is None or (np.linalg.norm(tfidf_vec) == 0.0):
                tfidf_vec = self.vectors.get(doc_id)
            norm_d = float(np.linalg.norm(tfidf_vec)) if tfidf_vec is not None else 0.0
            if norm_d == 0.0 or dot == 0.0:
                continue
            scores_list.append((doc_id, dot / (norm_q * norm_d)))
        return scores_list

    def knn(self, img_path: str, k: int = 5, use_inverted: bool = True) -> List[Union[int, str]]:
        self._ensure_loaded()
        if not self.vectors:
            return []

        try:
            q_vec = self._feature_vector(img_path)
        except Exception:
            return []

        # Asegurar tfidf + invertido
        self._ensure_tfidf_and_inverted()

        if self.idf is not None and self.idf.shape[0] == q_vec.shape[0]:
            q_vec_idf = q_vec * self.idf
            norm_q = float(np.linalg.norm(q_vec_idf))
            if norm_q > 0:
                q_vec = q_vec_idf
            else:
                norm_q = float(np.linalg.norm(q_vec))
        else:
            norm_q = float(np.linalg.norm(q_vec))
        if norm_q == 0.0:
            return []

        # kNN usando índice invertido si está disponible (o secuencial si no)
        if use_inverted and self.inverted:
            scores_list = self._knn_inverted(q_vec, norm_q)
        else:
            scores_list = self._knn_dense(q_vec, norm_q)

        scores_list.sort(key=lambda x: x[1], reverse=True)
        top = scores_list[:k]
        # guardar similitudes para consumidores (ej. File.knn)
        self.last_scores = {}
        out = []
        for did, sim in top:
            meta = self.doc_map.get(did) or {}
            pk = meta.get("pk", did)
            self.last_scores[pk] = sim
            out.append(pk)
        return out

    def close(self):
        return
