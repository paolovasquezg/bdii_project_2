# backend/storage/indexes/invtext.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional
import json, math, re, os

_TOK = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+", re.UNICODE)

def _tokenize(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return [t.lower() for t in _TOK.findall(text)]

class InvertedTextFile:
    """
    Índice invertido simple con TF-IDF + coseno.
    Estructura en disco (directorio base):
      - vocab.json       : {term -> term_id}
      - idf.json         : {term_id -> idf}
      - postings.jsonl   : JSONL de {"tid": int, "docs": {doc_id: tf}}
      - doc_map.json     : {doc_id: {"pk": <val>} ó {"pos": <int>}}
    """
    def __init__(self, base_dir: str, key: str, heap_file: Optional[str] = None):
        self.base = Path(base_dir)
        self.key = key
        self.heap_file = heap_file
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[int, float] = {}
        self.doc_map: Dict[int, Dict] = {}
        # postings en memoria perezoso (cargamos on-demand)
        self._postings_path = self.base / "postings.jsonl"
        self._postings_cache: Optional[Dict[int, Dict[int, float]]] = None

    # ------------- helpers de E/S -------------
    def _save_json(self, p: Path, obj):
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)

    def _load_json(self, p: Path, default):
        if not p.exists(): return default
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _load_all(self):
        self.vocab = self._load_json(self.base/"vocab.json", {})
        self.idf   = {int(k): v for k, v in self._load_json(self.base/"idf.json", {}).items()}
        self.doc_map = {int(k): v for k, v in self._load_json(self.base/"doc_map.json", {}).items()}
        self._postings_cache = None  # lazily

    def _save_all(self):
        self._save_json(self.base/"vocab.json", self.vocab)
        self._save_json(self.base/"idf.json", {str(k): v for k, v in self.idf.items()})
        self._save_json(self.base/"doc_map.json", {str(k): v for k, v in self.doc_map.items()})

    def _ensure_loaded(self):
        if not self.vocab or not self.idf or not self.doc_map:
            self._load_all()

    def _load_postings(self) -> Dict[int, Dict[int, float]]:
        if self._postings_cache is not None:
            return self._postings_cache
        postings: Dict[int, Dict[int, float]] = {}
        if self._postings_path.exists():
            with self._postings_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    row = json.loads(line)
                    tid = int(row["tid"])
                    postings[tid] = {int(k): float(v) for k, v in row["docs"].items()}
        self._postings_cache = postings
        return postings

    def _save_postings(self, postings: Dict[int, Dict[int, float]]):
        self._postings_path.parent.mkdir(parents=True, exist_ok=True)
        with self._postings_path.open("w", encoding="utf-8") as f:
            for tid, docs in postings.items():
                f.write(json.dumps({"tid": tid, "docs": {str(k): v for k, v in docs.items()}}, ensure_ascii=False) + "\n")

    # ------------- build -------------
    def build_bulk(self, records: Iterable, text_field: str, pk_name: str, main_index: str):
        """
        records:
          - si main_index == 'heap': iterable de (row_dict, pos)
          - si no: iterable de row_dict
        """
        vocab: Dict[str, int] = {}
        df: Dict[int, int] = {}
        tf: Dict[int, Dict[int, float]] = {}  # doc_id -> {tid: tf}
        doc_map: Dict[int, Dict] = {}

        def add_term(term: str) -> int:
            if term not in vocab:
                vocab[term] = len(vocab)
            return vocab[term]

        N = 0
        for rec in records:
            if main_index == "heap":
                row, pos = rec
            else:
                row, pos = rec, None
            if not isinstance(row, dict) or text_field not in row:
                continue
            text = row[text_field]
            tokens = _tokenize(text)
            if not tokens:
                continue
            N += 1
            doc_id = N - 1
            # doc map
            if main_index == "heap":
                doc_map[doc_id] = {"pos": int(pos)}
            else:
                doc_map[doc_id] = {"pk": row[pk_name]}
            # tf por doc
            tcount: Dict[int, float] = {}
            for t in tokens:
                tid = add_term(t)
                tcount[tid] = tcount.get(tid, 0.0) + 1.0
            # normaliza TF
            norm = math.sqrt(sum(v*v for v in tcount.values())) or 1.0
            for tid in list(tcount.keys()):
                tcount[tid] /= norm
            tf[doc_id] = tcount

        # IDF
        for doc_id, vec in tf.items():
            for tid in vec.keys():
                df[tid] = df.get(tid, 0) + 1
        idf: Dict[int, float] = {}
        for tid, dfi in df.items():
            idf[tid] = math.log((N + 1) / (dfi + 1)) + 1.0  # suavizado

        # aplicar IDF a TF para obtener TF-IDF
        postings: Dict[int, Dict[int, float]] = {}
        for doc_id, vec in tf.items():
            for tid, tfv in vec.items():
                w = tfv * idf[tid]
                postings.setdefault(tid, {})[doc_id] = w

        # guardar
        self.vocab = vocab
        self.idf = idf
        self.doc_map = doc_map
        self._save_all()
        self._save_postings(postings)

    # ------------- consulta -------------
    def knn(self, query_text: str, k: int) -> List[int]:
        """
        Retorna lista de doc_ids (ordenados por similitud desc).
        """
        self._ensure_loaded()
        postings = self._load_postings()
        toks = _tokenize(query_text)
        if not toks: return []
        # TF normalizado de query
        qtf: Dict[int, float] = {}
        for t in toks:
            tid = self.vocab.get(t)
            if tid is None:
                continue
            qtf[tid] = qtf.get(tid, 0.0) + 1.0
        if not qtf: return []
        qnorm = math.sqrt(sum(v*v for v in qtf.values())) or 1.0
        for tid in list(qtf.keys()):
            qtf[tid] /= qnorm
        # TF-IDF query
        for tid in list(qtf.keys()):
            qtf[tid] *= self.idf.get(tid, 0.0)

        # coseno: acumular producto punto
        score: Dict[int, float] = {}
        for tid, qw in qtf.items():
            docs = postings.get(tid, {})
            for doc_id, dw in docs.items():
                score[doc_id] = score.get(doc_id, 0.0) + (qw * dw)

        top = sorted(score.items(), key=lambda x: x[1], reverse=True)[:k]
        return [doc_id for doc_id, _ in top]