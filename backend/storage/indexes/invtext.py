# backend/storage/indexes/invtext.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional, Set
import json, math, re, os, unicodedata, shutil

# ------------------------------------------------------------
# Tokenización + normalización + stopwords + stemming (ES)
# ------------------------------------------------------------

import unicodedata, os, re
from typing import List, Set

_TOK = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+", re.UNICODE)

def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

# --- Stemmer: NLTK Snowball -> snowballstemmer -> identity ---
try:
    from nltk.stem.snowball import SnowballStemmer  # type: ignore
    _SNOW_STEMMER = SnowballStemmer("spanish")
    _STEMMER_KIND = "nltk"
except Exception:
    try:
        import snowballstemmer  # type: ignore
        _SNOW_STEMMER = snowballstemmer.stemmer("spanish")
        _STEMMER_KIND = "snowballstemmer"
    except Exception:
        _SNOW_STEMMER = None
        _STEMMER_KIND = "none"

_USE_STEM = os.getenv("BD2_INVTEXT_STEM", "1").lower() in ("1","true","yes","on")

# --- Stopwords provider (sin hardcode) ---
# Prioridad por defecto (auto): stopwordsiso -> spaCy -> NLTK -> vacío
def _load_stopwords() -> Set[str]:
    choice = os.getenv("BD2_INVTEXT_STOPWORDS", "auto").lower()

    def _normset(words) -> Set[str]:
        return {_strip_accents(w.lower()) for w in words if isinstance(w, str) and w}

    # 1) stopwordsiso (paquete ligero, no requiere descargas)
    if choice in ("auto", "stopwordsiso"):
        try:
            import stopwordsiso as stopwordsiso  # pip install stopwordsiso
            return _normset(stopwordsiso.stopwords("es"))
        except Exception:
            if choice == "stopwordsiso": return set()

    # 2) spaCy (no requiere modelo para STOP_WORDS)
    if choice in ("auto", "spacy"):
        try:
            from spacy.lang.es.stop_words import STOP_WORDS as SPACY_STOP_ES  # pip install spacy
            return _normset(SPACY_STOP_ES)
        except Exception:
            if choice == "spacy": return set()

    # 3) NLTK corpus (requiere descargar 'stopwords' si no está)
    if choice in ("auto", "nltk"):
        try:
            from nltk.corpus import stopwords as nltk_stop  # pip install nltk
            return _normset(nltk_stop.words("spanish"))
        except Exception:
            if choice == "nltk": return set()

    # 4) fallback vacío
    return set()

_STOP_NORM: Set[str] = _load_stopwords()

_ADDS = {w.strip() for w in os.getenv("BD2_INVTEXT_STOP_ADD","").split(",") if w.strip()}
_REMS = {w.strip() for w in os.getenv("BD2_INVTEXT_STOP_REMOVE","").split(",") if w.strip()}

if _ADDS or _REMS:
    base = set(_STOP_NORM)
    base |= {_strip_accents(w.lower()) for w in _ADDS}
    base -= {_strip_accents(w.lower()) for w in _REMS}
    _STOP_NORM = base

def _normalize(token: str) -> str:
    t = _strip_accents(token.lower())
    if not t:
        return ""
    # filtra stopwords externas ya normalizadas
    if t in _STOP_NORM:
        return ""
    if _USE_STEM and _SNOW_STEMMER is not None:
        try:
            if _STEMMER_KIND == "nltk":
                return _SNOW_STEMMER.stem(t)
            else:
                return _SNOW_STEMMER.stemWord(t)  # type: ignore[attr-defined]
        except Exception:
            return t
    return t

def _tokenize(text: str) -> List[str]:
    if not isinstance(text, str) or not text:
        return []
    raw = _TOK.findall(text)
    out: List[str] = []
    for tok in raw:
        norm = _normalize(tok)
        if norm:
            out.append(norm)
    return out

# ------------------------------------------------------------
# Índice Invertido con TF-IDF + Coseno + Streaming + SPIMI
# ------------------------------------------------------------

class InvertedTextFile:
    """
    Índice invertido TF-IDF con coseno exacto.
    Archivos en disco (base_dir):
      - vocab.json          : {term -> tid}
      - idf.json            : {tid -> idf}
      - doc_map.json        : {doc_id -> {"pk":...,} ó {"pos":...}}
      - doc_norms.json      : {doc_id -> ||d||_tfidf}
      - postings.jsonl      : JSONL {"tid": int, "docs": {doc_id: weight, ...}}
      - postings.delta.jsonl: JSONL {"tid": int, "doc": int, "w": float} (append-only)
      - deleted_docs.json   : [doc_id, ...] (borrado lógico)
      - meta.json           : info del tokenizador/stemmer
    SPIMI:
      - spimi_blocks/block_*.jsonl         : {"term": str, "docs": {doc_id: count}}
      - spimi_blocks/block_*.docstats.json : {doc_id: sumsq_counts}
    """
    def __init__(self, base_dir: str, key: str, heap_file: Optional[str] = None):
        self.base = Path(base_dir)
        self.key = key
        self.heap_file = heap_file

        self.vocab: Dict[str, int] = {}
        self.idf: Dict[int, float] = {}
        self.doc_map: Dict[int, Dict] = {}
        self.doc_norms: Dict[int, float] = {}
        self.meta_path = self.base / "meta.json"
        self.meta = {"tokenizer": f"snowball_es({_STEMMER_KIND})", "accents": "stripped"}

        # contadores IO (para io_merge)
        self.read_count = 0
        self.write_count = 0

        # paths
        self._postings_path = self.base / "postings.jsonl"
        self._delta_path    = self.base / "postings.delta.jsonl"
        self._deleted_path  = self.base / "deleted_docs.json"
        self._blocks_dir    = self.base / "spimi_blocks"

        self._postings_cache: Optional[Dict[int, Dict[int, float]]] = None  # compat opcional

    # ---------------- util E/S ----------------
    def _ensure_base(self):
        self.base.mkdir(parents=True, exist_ok=True)

    def _save_json(self, p: Path, obj):
        self._ensure_base()
        with p.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)
        self.write_count += 1

    def _load_json(self, p: Path, default):
        if not p.exists():
            return default
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        self.read_count += 1
        return data

    def _load_all(self):
        self.vocab     = self._load_json(self.base/"vocab.json", {})
        self.idf       = {int(k): float(v) for k, v in self._load_json(self.base/"idf.json", {}).items()}
        self.doc_map   = {int(k): v for k, v in self._load_json(self.base/"doc_map.json", {}).items()}
        self.doc_norms = {int(k): float(v) for k, v in self._load_json(self.base/"doc_norms.json", {}).items()}
        try:
            self.meta = self._load_json(self.meta_path, self.meta)
        except Exception:
            pass
        self._postings_cache = None

    def _save_all(self):
        self._save_json(self.base/"vocab.json", self.vocab)
        self._save_json(self.base/"idf.json", {str(k): v for k, v in self.idf.items()})
        self._save_json(self.base/"doc_map.json", {str(k): v for k, v in self.doc_map.items()})
        self._save_json(self.base/"doc_norms.json", {str(k): v for k, v in self.doc_norms.items()})
        self._save_json(self.meta_path, self.meta)

    # compat con llamadores que usan este nombre
    def _ensure_loaded(self):
        try:
            self._load_all()
        except Exception:
            pass

    # ---------------- lectura streaming ----------------
    def _load_postings_all(self) -> Dict[int, Dict[int, float]]:
        if self._postings_cache is not None:
            return self._postings_cache
        postings: Dict[int, Dict[int, float]] = {}
        if self._postings_path.exists():
            with self._postings_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    tid = int(row["tid"])
                    postings[tid] = {int(k): float(v) for k, v in row["docs"].items()}
            self.read_count += 1
        self._postings_cache = postings
        return postings

    def _iter_postings_for_terms(self, tids: Set[int]):
        if not tids:
            return
        if self._postings_path.exists():
            with self._postings_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    tid = int(row.get("tid", -1))
                    if tid in tids:
                        yield tid, {int(k): float(v) for k, v in row.get("docs", {}).items()}
            self.read_count += 1

    def _iter_delta_for_terms(self, tids: Set[int]):
        if not tids or not self._delta_path.exists():
            return
        with self._delta_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                try:
                    tid = int(row.get("tid"))
                except Exception:
                    continue
                if tid in tids:
                    doc = int(row.get("doc"))
                    w = float(row.get("w", 0.0))
                    yield tid, doc, w
        self.read_count += 1

    # ---------------- build (in-memory o SPIMI) ----------------
    def build_bulk(self, records: Iterable, text_field: str, pk_name: str, main_index: str):
        """
        records:
          - si main_index == 'heap': iterable de (row_dict, pos)
          - si no: iterable de row_dict
        """
        self._build_bulk_spimi(records, text_field, pk_name, main_index)


    # ---- SPIMI (Single-Pass In-Memory Indexing por bloques) ----
    def _build_bulk_spimi(self, records: Iterable, text_field: str, pk_name: str, main_index: str):
        self._ensure_base()
        # limpiar bloques previos
        if self._blocks_dir.exists():
            shutil.rmtree(self._blocks_dir, ignore_errors=True)
        self._blocks_dir.mkdir(parents=True, exist_ok=True)

        block_docs_max = int(os.getenv("BD2_INVTEXT_BLOCK_DOCS", "5000"))
        block_idx = 0
        docs_in_block = 0

        doc_map: Dict[int, Dict] = {}
        doc_sumsq_global: Dict[int, float] = {}
        next_doc_id = 0

        # estructuras del bloque (términos como strings)
        block_terms: Dict[str, Dict[int, int]] = {}   # term -> {doc_id: count}
        block_sumsq: Dict[int, int] = {}             # doc_id -> sum of squares (counts)

        def flush_block(idx: int):
            if not block_terms:
                return
            bpath = self._blocks_dir / f"block_{idx}.jsonl"
            with bpath.open("w", encoding="utf-8") as f:
                for term, docs in block_terms.items():
                    f.write(json.dumps({"term": term, "docs": {str(k): int(v) for k, v in docs.items()}}, ensure_ascii=False) + "\n")
            sspath = self._blocks_dir / f"block_{idx}.docstats.json"
            with sspath.open("w", encoding="utf-8") as f:
                json.dump({str(k): int(v) for k, v in block_sumsq.items()}, f, ensure_ascii=False)
            self.write_count += 2
            block_terms.clear()
            block_sumsq.clear()

        # 1) Pasada única: crear bloques con cuentas crudas por doc (SPIMI)
        for rec in records:
            if main_index == "heap":
                row, pos = rec
            else:
                row, pos = rec, None
            if not isinstance(row, dict) or text_field not in row:
                continue
            toks = _tokenize(row[text_field])
            if not toks:
                continue

            doc_id = next_doc_id
            next_doc_id += 1
            if main_index == "heap":
                doc_map[doc_id] = {"pos": int(pos)}
            else:
                doc_map[doc_id] = {"pk": row[pk_name]}

            # cuenta por doc (en memoria solo para este doc)
            local: Dict[str, int] = {}
            for t in toks:
                local[t] = local.get(t, 0) + 1

            # acumula en estructuras del bloque
            sumsq = 0
            for t, c in local.items():
                dmap = block_terms.setdefault(t, {})
                dmap[doc_id] = c
                sumsq += c * c
            block_sumsq[doc_id] = sumsq
            docs_in_block += 1

            if docs_in_block >= block_docs_max:
                flush_block(block_idx)
                block_idx += 1
                docs_in_block = 0

        # flush final
        flush_block(block_idx)

        # persist doc_map (global)
        self.doc_map = doc_map
        self._save_json(self.base/"doc_map.json", {str(k): v for k, v in doc_map.items()})

        # 2) Merge de bloques:
        # 2.1) DF por término y N
        df_by_term: Dict[str, int] = {}
        N = len(doc_map)
        # también juntamos todas las docstats para la normalización
        doc_sumsq_all: Dict[int, float] = {}
        for sspath in sorted(self._blocks_dir.glob("block_*.docstats.json")):
            stats = self._load_json(sspath, {})
            for k, v in stats.items():
                doc_sumsq_all[int(k)] = float(v)

        for bpath in sorted(self._blocks_dir.glob("block_*.jsonl")):
            with bpath.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    term = row["term"]
                    docs = row.get("docs", {})
                    df_by_term[term] = df_by_term.get(term, 0) + len(docs)
            self.read_count += 1

        # 2.2) vocab + idf
        vocab: Dict[str, int] = {}
        idf_by_tid: Dict[int, float] = {}
        for term in sorted(df_by_term.keys()):  # asigna tid determinista
            tid = len(vocab)
            vocab[term] = tid
            dfi = df_by_term[term]
            idf_by_tid[tid] = math.log((N + 1) / (dfi + 1)) + 1.0

        # 2.3) postings finales (TF-IDF) + doc_norms
        postings_by_tid: Dict[int, Dict[int, float]] = {}
        doc_norm2: Dict[int, float] = {}

        for bpath in sorted(self._blocks_dir.glob("block_*.jsonl")):
            with bpath.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    term = row["term"]
                    docs = row.get("docs", {})
                    tid = vocab[term]
                    idf = idf_by_tid[tid]
                    pdst = postings_by_tid.setdefault(tid, {})
                    for sd, cnt in docs.items():
                        did = int(sd)
                        sumsq = float(doc_sumsq_all.get(did, 1.0)) or 1.0
                        tf = float(cnt) / math.sqrt(sumsq)
                        w = tf * idf
                        pdst[did] = pdst.get(did, 0.0) + w
                        doc_norm2[did] = doc_norm2.get(did, 0.0) + (w*w)
            self.read_count += 1

        doc_norms = {doc_id: (math.sqrt(v) or 1.0) for doc_id, v in doc_norm2.items()}

        # 2.4) persist base
        self.vocab = vocab
        self.idf = idf_by_tid
        self.doc_norms = doc_norms
        self._save_all()

        with (self._postings_path).open("w", encoding="utf-8") as f:
            for tid in range(len(vocab)):
                docs = postings_by_tid.get(tid, {})
                if not docs:
                    continue
                f.write(json.dumps({"tid": tid, "docs": {str(k): float(v) for k, v in docs.items()}}, ensure_ascii=False) + "\n")
        self.write_count += 1

        # limpia delta/borrados y borra bloques SPIMI
        if self._delta_path.exists():
            try: self._delta_path.unlink()
            except Exception: pass
        if self._deleted_path.exists():
            try: self._deleted_path.unlink()
            except Exception: pass
        shutil.rmtree(self._blocks_dir, ignore_errors=True)

    # ---------------- inserción/borrado incremental ----------------
    def index_doc(self, doc_id: int, text: str):
        """
        Inserta/actualiza un documento usando append-only en postings.delta.jsonl.
        IDF se mantiene estático (para términos nuevos, idf=1.0).
        """
        self._ensure_base()
        self._load_all()

        if self.heap_file:
            self.doc_map[int(doc_id)] = {"pos": int(doc_id)}
        else:
            self.doc_map[int(doc_id)] = {"pk": doc_id}

        toks = _tokenize(text or "")
        if not toks:
            self._save_all()
            return

        tids: List[int] = []
        for t in toks:
            tid = self.vocab.get(t)
            if tid is None:
                tid = len(self.vocab)
                self.vocab[t] = tid
                self.idf[tid] = 1.0  # default para términos nuevos
            tids.append(tid)

        # TF L2
        tcount: Dict[int, float] = {}
        for tid in tids:
            tcount[tid] = tcount.get(tid, 0.0) + 1.0
        norm = math.sqrt(sum(v*v for v in tcount.values())) or 1.0
        for tid in list(tcount.keys()):
            tcount[tid] /= norm

        # pesos + norma doc
        norm2 = 0.0
        lines = []
        for tid, tfv in tcount.items():
            w = tfv * float(self.idf.get(tid, 1.0))
            norm2 += (w*w)
            lines.append({"tid": int(tid), "doc": int(doc_id), "w": float(w)})

        self.doc_norms[int(doc_id)] = math.sqrt(norm2) or 1.0
        self._save_all()

        with (self._delta_path).open("a", encoding="utf-8") as f:
            for row in lines:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.write_count += 1

    def remove_doc(self, doc_id: int, _text: Optional[str] = None):
        """Marca un documento como eliminado (tumba)."""
        self._ensure_base()
        dels = set(self._load_json(self._deleted_path, []))
        dels.add(int(doc_id))
        self._save_json(self._deleted_path, sorted(int(x) for x in dels))
        if int(doc_id) in self.doc_map:
            del self.doc_map[int(doc_id)]
            self._save_json(self.base/"doc_map.json", {str(k): v for k, v in self.doc_map.items()})

    # ---------------- consulta (coseno + streaming) ----------------
    def knn(self, query_text: str, k: int) -> List[int]:
        """
        Retorna lista de doc_ids (ordenados por similitud desc) usando coseno exacto.
        Lee solo postings de los términos de la query.
        """
        self._load_all()
        toks = _tokenize(query_text or "")
        if not toks:
            return []

        # Query TF L2
        qtf: Dict[int, float] = {}
        tids: Set[int] = set()
        for t in toks:
            tid = self.vocab.get(t)
            if tid is None:
                continue
            qtf[tid] = qtf.get(tid, 0.0) + 1.0
            tids.add(tid)
        if not qtf:
            return []

        qnorm = math.sqrt(sum(v*v for v in qtf.values())) or 1.0
        for tid in list(qtf.keys()):
            qtf[tid] /= qnorm
            qtf[tid] *= float(self.idf.get(tid, 0.0))  # TF-IDF de la query

        # dot products
        scores: Dict[int, float] = {}

        # postings base
        for tid, docs in self._iter_postings_for_terms(tids):
            qw = float(qtf.get(tid, 0.0))
            if qw == 0.0:
                continue
            for doc_id, dw in docs.items():
                scores[doc_id] = scores.get(doc_id, 0.0) + (qw * float(dw))

        # postings delta
        for tid, doc, w in self._iter_delta_for_terms(tids):
            qw = float(qtf.get(tid, 0.0))
            if qw == 0.0:
                continue
            scores[int(doc)] = scores.get(int(doc), 0.0) + (qw * float(w))

        if not scores:
            return []

        # filtra eliminados y normaliza por ||d||
        deleted = set(int(x) for x in self._load_json(self._deleted_path, []))
        for doc_id in list(scores.keys()):
            if doc_id in deleted:
                scores.pop(doc_id, None)
                continue
            dnorm = float(self.doc_norms.get(int(doc_id), 1.0)) or 1.0
            scores[doc_id] = scores[doc_id] / dnorm

        # top-k
        try:
            kk = int(k)
        except Exception:
            kk = 8
        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:kk]
        return [int(doc_id) for doc_id, _ in top]

    # ---------------- util opcional: close() ----------------
    def close(self):
        return