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
        self.dim: int = 0
        self._loaded = False

        os.makedirs(self.base_dir, exist_ok=True)
        self._load_artifacts()

    # ------------------------------- helpers ---------------------------------------- #

    def _feature_vector(self, path: str) -> np.ndarray:
        """Histograma de intensidades (32 bins) normalizado."""
        img = Image.open(path).convert("L")
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

        self._loaded = bool(self.doc_map or self.vectors)

    def _persist(self):
        os.makedirs(self.base_dir, exist_ok=True)

        # doc_map
        doc_map_path = os.path.join(self.base_dir, "doc_map.json")
        with open(doc_map_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in self.doc_map.items()}, f, ensure_ascii=False)

        # postings
        postings_path = os.path.join(self.base_dir, "postings.json")
        with open(postings_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v.tolist() for k, v in self.vectors.items()}, f)

        # idf
        if self.idf is None and self.vectors:
            self._compute_idf()
        if self.idf is not None:
            np.save(os.path.join(self.base_dir, "idf.npy"), self.idf)

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

        self.write_count += 4

    def _compute_idf(self):
        if not self.vectors:
            self.idf = None
            return
        arr = np.vstack(list(self.vectors.values()))
        df = (arr > 0).astype(np.int32).sum(axis=0)
        N = arr.shape[0]
        self.idf = np.log((N + 1) / (df + 1))
        self.dim = int(self.idf.shape[0])

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

    def knn(self, img_path: str, k: int = 5) -> List[Union[int, str]]:
        self._ensure_loaded()
        if not self.vectors:
            return []

        try:
            q_vec = self._feature_vector(img_path)
        except Exception:
            return []

        if self.idf is not None and self.idf.shape[0] == q_vec.shape[0]:
            q_vec = q_vec * self.idf

        norm_q = float(np.linalg.norm(q_vec))
        if norm_q == 0.0:
            return []

        scores: List[Tuple[Union[int, str], float]] = []
        for doc_id, vec in self.vectors.items():
            dv = vec
            if self.idf is not None and self.idf.shape[0] == dv.shape[0]:
                dv = dv * self.idf
            norm_d = float(np.linalg.norm(dv))
            if norm_d == 0.0:
                continue
            sim = float(np.dot(q_vec, dv) / (norm_q * norm_d))
            scores.append((doc_id, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        top = [sid for sid, _ in scores[:k]]
        # devolver siempre la PK si la tenemos en doc_map
        out = []
        for did in top:
            meta = self.doc_map.get(did) or {}
            out.append(meta.get("pk", did))
        return out

    def close(self):
        return
