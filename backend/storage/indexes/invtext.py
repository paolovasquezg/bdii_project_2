import json
import math
import os
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple, Union

from backend.catalog.settings import DATA_DIR


DocRecord = Union[dict, Tuple[dict, int]]


class InvertedTextFile:
    """
    Índice invertido simple con TF‑IDF en disco.
    Artefactos persistidos:
      - vocab.json         : lista de términos
      - idf.json           : {termino: idf}
      - postings.jsonl     : líneas {"term": t, "postings": {doc_id: tf}}
      - doc_map.json       : {doc_id: {"pk":..., "pos":..., "text":...}}
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

        self.vocab: Dict[str, bool] = {}
        self.idf: Dict[str, float] = {}
        self.postings: Dict[str, Dict[str, int]] = {}
        self.doc_map: Dict[Union[int, str], dict] = {}
        self.doc_vectors: Dict[Union[int, str], Dict[str, float]] = {}
        self.doc_norms: Dict[Union[int, str], float] = {}
        self._loaded = False

        os.makedirs(self.base_dir, exist_ok=True)

    # ------------------------------- helpers internos ------------------------------- #

    def _tokenize(self, text: str) -> List[str]:
        """Tokenizador liviano: minúsculas + caracteres alfanuméricos (incluye tildes)."""
        if not isinstance(text, str):
            return []
        lowered = text.lower()
        return re.findall(r"[a-z0-9áéíóúüñ]+", lowered, flags=re.IGNORECASE)

    def _persist(self):
        os.makedirs(self.base_dir, exist_ok=True)

        vocab_path = os.path.join(self.base_dir, "vocab.json")
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(sorted(self.vocab.keys()), f, ensure_ascii=False)
        self.write_count += 1

        idf_path = os.path.join(self.base_dir, "idf.json")
        with open(idf_path, "w", encoding="utf-8") as f:
            json.dump(self.idf, f, ensure_ascii=False)
        self.write_count += 1

        postings_path = os.path.join(self.base_dir, "postings.jsonl")
        with open(postings_path, "w", encoding="utf-8") as f:
            for term in sorted(self.postings.keys()):
                json.dump({"term": term, "postings": self.postings[term]}, f, ensure_ascii=False)
                f.write("\n")
                self.write_count += 1

        doc_map_path = os.path.join(self.base_dir, "doc_map.json")
        with open(doc_map_path, "w", encoding="utf-8") as f:
            # serializar claves como str
            jmap = {str(k): v for k, v in self.doc_map.items()}
            json.dump(jmap, f, ensure_ascii=False)
        self.write_count += 1

    def _load_if_exists(self):
        if self._loaded:
            return

        vocab_path = os.path.join(self.base_dir, "vocab.json")
        if os.path.exists(vocab_path):
            with open(vocab_path, "r") as f:
                for t in json.load(f) or []:
                    self.vocab[str(t)] = True
            self.read_count += 1  # vocab

        idf_path = os.path.join(self.base_dir, "idf.json")
        if os.path.exists(idf_path):
            with open(idf_path, "r") as f:
                self.idf = {str(k): float(v) for k, v in (json.load(f) or {}).items()}
            self.read_count += 1  # idf

        postings_path = os.path.join(self.base_dir, "postings.jsonl")
        if os.path.exists(postings_path):
            with open(postings_path, "r") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        term = obj.get("term")
                        p = obj.get("postings") or obj.get("lista_posteo") or {}
                        if term:
                            self.postings[str(term)] = {str(k): int(v) for k, v in p.items()}
                    except Exception:
                        continue
            self.read_count += 1  # postings

        doc_map_path = os.path.join(self.base_dir, "doc_map.json")
        if os.path.exists(doc_map_path):
            with open(doc_map_path, "r") as f:
                try:
                    raw = json.load(f) or {}
                    for k, v in raw.items():
                        dk: Union[int, str]
                        try:
                            dk = int(k)
                        except Exception:
                            dk = k
                        self.doc_map[dk] = v
                except Exception:
                    self.doc_map = {}
            self.read_count += 1  # doc_map

        # reconstruir vectores (no persiste)
        if self.doc_map:
            self._recompute_from_doc_map(persist=False)

        self._loaded = True

    def _recompute_from_doc_map(self, *, persist: bool):
        """Recalcula postings, idf y normas desde doc_map.text."""
        tf_docs: Dict[Union[int, str], Dict[str, int]] = {}
        df = defaultdict(int)
        N = len(self.doc_map)

        for doc_id, meta in self.doc_map.items():
            text = (meta or {}).get("text", "")
            tokens = self._tokenize(text)
            tf = defaultdict(int)
            for t in tokens:
                tf[t] += 1
            tf_docs[doc_id] = tf
            for t in tf:
                df[t] += 1

        self.vocab = {t: True for t in df}
        self.idf = {}
        for t, freq in df.items():
            if freq > 0 and N > 0:
                self.idf[t] = math.log10(N / float(freq))
            else:
                self.idf[t] = 0.0

        self.postings = {}
        for t in df:
            self.postings[t] = {}
        for doc_id, tf in tf_docs.items():
            for t, freq in tf.items():
                self.postings[t][str(doc_id)] = int(freq)

        self.doc_vectors = {}
        self.doc_norms = {}
        for doc_id, tf in tf_docs.items():
            vec = {}
            for t, freq in tf.items():
                idf = self.idf.get(t, 0.0)
                if idf == 0.0:
                    continue
                vec[t] = float(freq) * idf
            norm = math.sqrt(sum(w * w for w in vec.values())) if vec else 0.0
            self.doc_vectors[doc_id] = vec
            self.doc_norms[doc_id] = norm

        if persist:
            self._persist()

    # ------------------------------- API público ------------------------------------ #

    def _ensure_loaded(self):
        self._load_if_exists()

    def build_bulk(self, records: Iterable[DocRecord], *, text_field: str, pk_name: str, main_index: str):
        """
        Construye el índice desde cero a partir de registros.
        records:
            - heap: [(row_dict, pos), ...]
            - otros: [row_dict, ...]
        """
        self.doc_map = {}
        for rec in records:
            if isinstance(rec, tuple) and len(rec) >= 2:
                row, pos = rec[0], rec[1]
                if not isinstance(row, dict) or text_field not in row or pk_name not in row:
                    continue
                text = row.get(text_field)
                if text is None or str(text).strip() == "":
                    continue
                doc_id: Union[int, str] = pos if main_index == "heap" else row.get(pk_name)
                if doc_id is None:
                    continue
                self.doc_map[doc_id] = {
                    "pk": row.get(pk_name),
                    "pos": pos,
                    "text": str(text),
                }
            elif isinstance(rec, dict):
                if text_field not in rec or pk_name not in rec:
                    continue
                text = rec.get(text_field)
                if text is None or str(text).strip() == "":
                    continue
                doc_id = rec.get(pk_name)
                if doc_id is None:
                    continue
                self.doc_map[doc_id] = {
                    "pk": rec.get(pk_name),
                    "pos": rec.get("pos"),
                    "text": str(text),
                }

        self._recompute_from_doc_map(persist=True)
        self._loaded = True

    def index_doc(self, doc_id: Union[int, str], text: str):
        self._ensure_loaded()
        doc_key: Union[int, str]
        try:
            doc_key = int(doc_id)
        except Exception:
            doc_key = doc_id
        meta = dict(self.doc_map.get(doc_key, {}))
        meta.setdefault("pk", doc_key)
        meta.setdefault("pos", doc_key)
        meta["text"] = str(text)
        self.doc_map[doc_key] = meta
        self._recompute_from_doc_map(persist=True)

    def remove_doc(self, doc_id: Union[int, str], text: str | None = None):
        self._ensure_loaded()
        try:
            doc_key: Union[int, str] = int(doc_id)
        except Exception:
            doc_key = doc_id
        if doc_key in self.doc_map:
            self.doc_map.pop(doc_key, None)
            self._recompute_from_doc_map(persist=True)

    def knn(self, query_text: str, k: int = 5) -> List[Union[int, str]]:
        self._ensure_loaded()
        tokens = self._tokenize(query_text)
        if not tokens or not self.doc_vectors:
            return []

        tf_q = defaultdict(int)
        for t in tokens:
            tf_q[t] += 1
        vec_q: Dict[str, float] = {}
        for t, freq in tf_q.items():
            idf = self.idf.get(t, 0.0)
            if idf == 0.0:
                continue
            vec_q[t] = float(freq) * idf
        norm_q = math.sqrt(sum(w * w for w in vec_q.values())) if vec_q else 0.0
        if norm_q == 0.0:
            return []

        scores: List[Tuple[Union[int, str], float]] = []
        for doc_id, vec_d in self.doc_vectors.items():
            norm_d = self.doc_norms.get(doc_id) or 0.0
            if norm_d == 0.0:
                continue
            dot = 0.0
            for t, wq in vec_q.items():
                wd = vec_d.get(t)
                if wd:
                    dot += wq * wd
            if dot == 0.0:
                continue
            scores.append((doc_id, dot / (norm_q * norm_d)))
            # Lectura lógica de postings/doc_vectors
            self.read_count += 1

        scores.sort(key=lambda x: x[1], reverse=True)
        # guarda last_scores para consumidores que quieran similitud
        self.last_scores = {sid: sim for sid, sim in scores[:k]}
        return [sid for sid, _ in scores[:k]]

    def close(self):
        """Compatibilidad básica; nada que cerrar."""
        return
