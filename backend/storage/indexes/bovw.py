# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import json, os
import numpy as np

try:
    import cv2
    from PIL import Image
    from sklearn.cluster import KMeans
    import joblib
except Exception as e:
    raise RuntimeError("Instala dependencias: opencv-contrib-python-headless (o opencv-python-headless ≥4.5), pillow, scikit-learn, joblib") from e

def _imread_gray(p: Path) -> Optional[np.ndarray]:
    im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if im is None:
        try:
            im = np.array(Image.open(p).convert("L"))
        except Exception:
            return None
    return im

def _assign(desc: np.ndarray, C: np.ndarray) -> np.ndarray:
    if desc.size == 0:
        return np.empty((0,), dtype=np.int32)
    X = desc.astype(np.float32)
    x2 = (X**2).sum(axis=1, keepdims=True)
    c2 = (C**2).sum(axis=1, keepdims=True).T
    dots = X @ C.T
    d2 = x2 + c2 - 2 * dots
    return np.argmin(d2, axis=1)

def _hist(assigns: np.ndarray, V: int) -> np.ndarray:
    return np.bincount(assigns, minlength=V).astype(np.float32)

def _tfidf(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    N, _ = H.shape
    df = (H > 0).sum(axis=0)
    idf = np.log((1 + N) / (1 + df)) + 1.0
    W = H * idf
    norms = np.linalg.norm(W, axis=1) + 1e-9
    W = W / norms[:, None]
    return W, idf

class BoVWFile:
    """
    Índice BoVW secundario para un campo (columna) que guarda la ruta de imagen.
    Usa SIFT (128-D float32)
    Persistimos en un *directorio base* guardado en 'filename' del catálogo.
    Estructura:
      base_dir/
        codebook.joblib
        idf.npy
        postings.json      # {word: [[pk, w], ...]}
        doc_map.json       # {pk: {"image_path":str}}
        docvecs.npy        # (N x V) float32
        pks.npy            # (N,) int64
    """
    def __init__(self, base_dir: str, *, key: str, heap_file: Optional[str] = None):
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)
        self.key = key
        self.heap_file = heap_file
        # IO counters (para integrarse con io_merge)
        self.read_count = 0
        self.write_count = 0

        self._codebook_f = self.base / "codebook.joblib"
        self._idf_f      = self.base / "idf.npy"
        self._post_f     = self.base / "postings.json"
        self._map_f      = self.base / "doc_map.json"

        self._docvecs_f = self.base / "docvecs.npy"  # matriz densa (N x V), float32
        self._pks_f     = self.base / "pks.npy"      # vector de PKs alineado a filas de docvecs

    def _dense_grid_kp(self, shape, step=16, size=16):
        h, w = shape[:2]
        pts = []
        xs = range(size // 2, max(w - size // 2, size // 2) + 1, step)
        ys = range(size // 2, max(h - size // 2, size // 2) + 1, step)
        for y in ys:
            for x in xs:
                pts.append(cv2.KeyPoint(float(x), float(y), size))
        return pts

    # ===== Extracción SIFT (robusta) =====
    def _sift(self, max_kp: int = 4000):
        # Parámetros relajados para superficies “planas”
        try:
            return cv2.SIFT_create(
                nfeatures=max_kp,
                contrastThreshold=0.02,
                edgeThreshold=10,
                sigma=1.6,
            )
        except Exception:
            # Compat con builds viejos
            return cv2.SIFT_create(nfeatures=max_kp)

    def _extract_sift_from_gray(self, img: np.ndarray, max_kp: int) -> np.ndarray:
        sift = self._sift(max_kp)
        kp, des = sift.detectAndCompute(img, None)
        if des is not None and len(des) > 0:
            return des.astype(np.float32)

        # Fallback 1: keypoints densos (grilla)
        kp = self._dense_grid_kp(img.shape, step=max(8, min(img.shape[:2]) // 24))
        _, des = sift.compute(img, kp)
        if des is not None and len(des) > 0:
            return des.astype(np.float32)

        # Fallback 2: grilla más fina
        kp = self._dense_grid_kp(img.shape, step=max(8, min(img.shape[:2]) // 16))
        _, des = sift.compute(img, kp)
        if des is not None and len(des) > 0:
            return des.astype(np.float32)

        # Nada encontrado
        return np.empty((0, 128), dtype=np.float32)

    # Acepta path (lo usa query); aplica mismos pasos robustos
    def _extract_descriptors(self, img_path: str) -> Optional[np.ndarray]:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        return self._extract_sift_from_gray(img, max_kp=4000)

    # ===== Utilidad para resolver rutas de query =====
    def _resolve_query_path(self, qpath: str) -> Optional[Path]:
        p = Path(qpath)
        if p.exists():
            return p

        # Intenta relativo al directorio donde están las imágenes indexadas (doc_map)
        try:
            _, _, _, docmap = self._load()
            some = next(iter(docmap.values()))
            base = Path(some["image_path"]).parent
            cand1 = base / qpath
            cand2 = base / Path(qpath).name
            for c in (cand1, cand2):
                if c.exists():
                    return c
        except Exception:
            pass
        return None

    # ------------------ BUILD (bulk) ------------------ #
    def build_bulk(self, records: List[Any], *, image_field: str, pk_name: str,
                   main_index: str, sample_cap: int = 200_000, kmeans_k: int = 512, max_kp: int = 600):
        """
        records:
          - si main_index == 'heap': lista de (row_dict, pos)
          - else: lista de row_dict
        """
        # 1) reunir rutas e IDs
        pairs: List[Tuple[int, Path]] = []
        for r in records:
            row = r[0] if main_index == "heap" else r
            if row is None or image_field not in row:
                continue
            try:
                pk = int(row[pk_name])
            except Exception:
                continue
            p = Path(row[image_field])
            pairs.append((pk, p))

        if not pairs:
            raise RuntimeError("No hay imágenes válidas para construir el índice BoVW.")

        # 2) extraer SIFT y muestrear para KMeans
        all_descs = []
        imgs_descs = []
        pk_list: List[int] = []
        for pk, img_p in pairs:
            im = _imread_gray(img_p)
            if im is None:
                imgs_descs.append(np.empty((0, 128), dtype=np.float32))
                pk_list.append(pk)
                continue
            D = self._extract_sift_from_gray(im, max_kp=max_kp)
            imgs_descs.append(D)
            pk_list.append(pk)
            if D is not None and len(D) > 0:
                take = min(len(D), 200)
                sel = D[np.random.choice(len(D), size=take, replace=False)]
                all_descs.append(sel)

        if not all_descs:
            raise RuntimeError("No se extrajeron descriptores SIFT.")

        X = np.vstack(all_descs).astype(np.float32)
        if len(X) > sample_cap:
            X = X[np.random.choice(len(X), sample_cap, replace=False)]

        n_samples = int(X.shape[0])

        # 3) KMeans (diccionario visual) con K adaptativo
        target = max(8, n_samples // 4)
        K = min(512, target)
        K = max(2, min(K, n_samples))
        if K < 2:
            raise RuntimeError(f"No hay suficientes descriptores para BoVW: n={n_samples}")

        try:
            km = KMeans(n_clusters=K, n_init="auto", random_state=42)
        except TypeError:
            km = KMeans(n_clusters=K, n_init=10, random_state=42)

        km.fit(X)
        joblib.dump(km, self._codebook_f); self.write_count += 1

        # 4) BoVW por imagen
        V = km.n_clusters
        H = np.zeros((len(imgs_descs), V), dtype=np.float32)
        for i, D in enumerate(imgs_descs):
            if D is None or D.size == 0:
                continue
            ass = _assign(D.astype(np.float32), km.cluster_centers_)
            H[i] = _hist(ass, V)

        # 5) TF-IDF y postings
        W, idf = _tfidf(H)
        postings: Dict[str, List[List[float]]] = {}
        for j, pk in enumerate(pk_list):
            row = W[j]
            nz = np.nonzero(row)[0]
            for v in nz:
                postings.setdefault(str(int(v)), []).append([int(pk), float(row[int(v)])])

        # Persistimos también matriz densa y pks para knn_seq
        np.save(self._docvecs_f, W.astype(np.float32)); self.write_count += 1
        np.save(self._pks_f, np.array(pk_list, dtype=np.int64)); self.write_count += 1

        # 6) persist resto
        np.save(self._idf_f, idf); self.write_count += 1
        with open(self._post_f, "w", encoding="utf-8") as f:
            json.dump(postings, f); self.write_count += 1
        with open(self._map_f, "w", encoding="utf-8") as f:
            json.dump({int(pk): {"image_path": str(p)} for pk, p in pairs}, f); self.write_count += 1

    # ------------------ QUERY ------------------ #
    def _load(self):
        km = joblib.load(self._codebook_f); self.read_count += 1
        idf = np.load(self._idf_f);        self.read_count += 1
        with open(self._post_f, "r", encoding="utf-8") as f:
            postings = json.load(f);       self.read_count += 1
        with open(self._map_f, "r", encoding="utf-8") as f:
            docmap = json.load(f);         self.read_count += 1
        return km, idf, postings, docmap

    def _vec_query(self, img_path: Path) -> Optional[np.ndarray]:
        # Cargar codebook e idf primero (necesitamos V para vector cero)
        km, idf, _, _ = self._load()

        # Resolver ruta (acepta absoluta, relativa o basename)
        rp = self._resolve_query_path(str(img_path)) or Path(img_path)
        if not rp.exists():
            return None

        # Usa el MISMO extractor robusto que en build (SIFT)
        D = self._extract_descriptors(str(rp))
        if D is None or D.size == 0:
            return np.zeros((km.n_clusters,), dtype=np.float32)

        ass = _assign(D.astype(np.float32), km.cluster_centers_)
        h = _hist(ass, km.n_clusters)
        w = h * idf
        n = np.linalg.norm(w) + 1e-9
        return w / n

    def knn(self, query_image_path: str, k: int) -> list[int]:
        km, idf, postings, docmap = self._load()
        q = self._vec_query(Path(query_image_path))
        if q is None:
            return []

        active = np.nonzero(q)[0]
        scores: dict[int, float] = {}
        for v in active:
            pv = postings.get(str(int(v)))
            if not pv:
                continue
            qv = float(q[int(v)])
            for pk, w in pv:
                scores[int(pk)] = scores.get(int(pk), 0.0) + qv * float(w)

        # top-k normal
        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:int(k)]
        res = [pk for pk, _ in top]

        # Self-match si la query es una imagen del índice
        qstr = str(query_image_path)
        self_pk = None
        for pk_str, meta in docmap.items():
            try:
                if meta.get("image_path") == qstr:
                    self_pk = int(pk_str)
                    break
            except Exception:
                pass

        if self_pk is not None and self_pk not in res:
            res = [self_pk] + res
            if len(res) > k:
                res = res[:k]

        return res

    # --- métodos nuevos en BoVWFile (sin cambios de firma) ---
    def _load_dense(self):
        """
        Carga (o reconstruye) la matriz densa de documentos y el vector de PKs.
        Si no existen, se reconstruye desde postings y se persiste para usos futuros.
        """
        import numpy as _np
        if self._docvecs_f.exists() and self._pks_f.exists():
            D = _np.load(self._docvecs_f); self.read_count += 1
            P = _np.load(self._pks_f);     self.read_count += 1
            return D, P

        km, idf, postings, _ = self._load()
        V = int(km.n_clusters)
        doc_ids = sorted({int(pk) for lst in postings.values() for pk, _w in lst})
        if not doc_ids:
            return _np.zeros((0, V), dtype=_np.float32), _np.array([], dtype=_np.int64)
        idx = {int(pk): i for i, pk in enumerate(doc_ids)}
        D = _np.zeros((len(doc_ids), V), dtype=_np.float32)

        for vs, lst in postings.items():
            v = int(vs)
            for pk, w in lst:
                i = idx[int(pk)]
                D[i, v] = float(w)

        _np.save(self._docvecs_f, D); self.write_count += 1
        _np.save(self._pks_f, _np.array(doc_ids)); self.write_count += 1
        return D, _np.array(doc_ids, dtype=_np.int64)

    def knn_seq(self, query_image_path: str, k: int) -> list[int]:
        """
        Baseline kNN secuencial: producto punto entre el vector TF-IDF de la query
        y la matriz densa (N x V). Retorna las PKs top-k.
        """
        import numpy as _np
        q = self._vec_query(Path(query_image_path))
        if q is None:
            return []
        D, P = self._load_dense()
        if D.size == 0:
            return []
        scores = D @ q.astype(_np.float32)
        k = int(k)
        if k <= 0:
            return []
        top = _np.argsort(-scores)[:k]
        return [int(P[i]) for i in top if scores[i] > 0.0]