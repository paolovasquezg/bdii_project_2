# backend/storage/indexes/invtext.py
# -*- coding: utf-8 -*-
import json, math, os, re, unicodedata, heapq
from collections import defaultdict, Counter
from pathlib import Path
from typing import Iterable, Tuple, List, Dict

_SP_STOP = {
    "a","al","algo","algunas","algunos","ante","antes","como","con","contra","cual",
    "cuando","de","del","desde","donde","dos","el","ella","ellas","ellos","en","entre",
    "era","erais","eran","eras","eres","es","esa","esas","ese","eso","esos","esta",
    "estaba","estaban","estado","estados","estamos","estar","estas","este","esto",
    "estos","fue","fueron","ha","habeis","habia","habian","han","has","hasta","hay",
    "la","las","le","les","lo","los","mas","me","mi","mis","mucho","muy","no","nos",
    "nosotros","o","os","otra","otras","otro","otros","para","pero","poco","por","porque",
    "que","se","sea","sean","segun","ser","si","sin","sobre","sois","somos","son","soy",
    "su","sus","tambien","tanto","te","tenemos","tiene","tienen","toda","todas","todo",
    "todos","tu","tus","un","una","uno","vosotros","y","ya"
}

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

def _norm_txt(s: str) -> str:
    s = s.lower()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return s

def _tokenize(text: str) -> List[str]:
    text = _norm_txt(text or "")
    toks = _TOKEN_RE.findall(text)
    return [t for t in toks if t not in _SP_STOP and len(t) > 1]

class InvTextFile:
    """
    Índice invertido en disco para texto (TF-IDF + coseno).
    Artefactos:
      - vocab.json      : {"term": {"df": int, "i": idx_en_arrays}, ...}
      - postings.jsonl  : una línea por término => {"t": "term", "p": [[doc, w], ...]}
      - doc_map.json    : {"N": num_docs, "ids": [pk_doc0, pk_doc1, ...]}
      - norms.json      : [||d0||, ||d1||, ...]
      - meta.json       : {"field": "...", "version": 1}
    """
    def __init__(self, base_dir: str, field: str):
        self.base = Path(base_dir); self.base.mkdir(parents=True, exist_ok=True)
        self.field = field
        self.vocab_path   = self.base/"vocab.json"
        self.postings_jl  = self.base/"postings.jsonl"
        self.docmap_path  = self.base/"doc_map.json"
        self.norms_path   = self.base/"norms.json"
        self.meta_path    = self.base/"meta.json"

    # ---------- BUILD (SPIMI simplificado con merge único) ----------
    def build_from_docs(self, docs: Iterable[Tuple[int, str]]):
        """
        docs: iterable de (pk, text_concatenado)
        Hace un pase por memoria (TF) y escribe postings ordenados por término.
        """
        # 1) recolecta TF por doc y DF por palabra
        per_doc_terms: List[Dict[str, float]] = []
        doc_ids: List[int] = []
        df: Counter = Counter()
        for pk, text in docs:
            toks = _tokenize(text)
            if not toks:
                per_doc_terms.append({})
                doc_ids.append(pk)
                continue
            tf = Counter(toks)
            # normalizamos TF log(1+tf)
            tfw = {t: 1.0 + math.log(v) for t, v in tf.items()}
            per_doc_terms.append(tfw)
            doc_ids.append(pk)
            df.update(tfw.keys())
        N = len(per_doc_terms)
        # 2) IDF
        idf: Dict[str, float] = {t: math.log((N + 1) / (df[t] + 1)) + 1.0 for t in df.keys()}
        # 3) TF-IDF por doc + normas
        doc_norms: List[float] = [0.0]*N
        inv: Dict[str, List[Tuple[int, float]]] = defaultdict(list)  # t -> [(doc_idx, w)]
        for i, tfw in enumerate(per_doc_terms):
            acc2 = 0.0
            for t, tfw_i in tfw.items():
                w = tfw_i * idf[t]
                inv[t].append((i, w))
                acc2 += w*w
            doc_norms[i] = math.sqrt(acc2) if acc2 > 0 else 1.0

        # 4) persistencia: vocab, postings.jsonl, doc_map, norms, meta
        vocab = {}
        for idx, t in enumerate(sorted(inv.keys())):
            vocab[t] = {"df": len(inv[t]), "i": idx}
        with self.vocab_path.open("w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False)

        with self.postings_jl.open("w", encoding="utf-8") as f:
            for t in sorted(inv.keys()):
                f.write(json.dumps({"t": t, "p": inv[t]}, ensure_ascii=False) + "\n")

        with self.docmap_path.open("w", encoding="utf-8") as f:
            json.dump({"N": N, "ids": doc_ids}, f, ensure_ascii=False)

        with self.norms_path.open("w", encoding="utf-8") as f:
            json.dump(doc_norms, f)

        with self.meta_path.open("w", encoding="utf-8") as f:
            json.dump({"field": self.field, "version": 1}, f)

    # ---------- QUERY ----------
    def _load_vocab(self):
        if not self.vocab_path.exists():
            return {}
        return json.loads(self.vocab_path.read_text(encoding="utf-8"))

    def _iter_postings(self, need_terms: List[str]):
        need = set(need_terms)
        with self.postings_jl.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                if obj["t"] in need:
                    yield obj["t"], obj["p"]

    def _load_docmap_norms(self):
        dm = json.loads(self.docmap_path.read_text(encoding="utf-8"))
        norms = json.loads(self.norms_path.read_text(encoding="utf-8"))
        return dm["ids"], norms

    def knn(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """
        Retorna lista [(pk, score)] ordenada desc por score, top-k por coseno.
        Sin cargar el índice completo: lee sólo postings de términos de la consulta.
        """
        vocab = self._load_vocab()
        if not vocab:
            return []
        q_toks = _tokenize(query)
        if not q_toks:
            return []
        # TF-IDF de la consulta
        tfq = Counter(q_toks)
        tfq_w = {t: 1.0 + math.log(v) for t, v in tfq.items() if t in vocab}
        if not tfq_w:
            return []
        # IDF desde vocab (aprox: df -> idf)
        N = json.loads(self.docmap_path.read_text(encoding="utf-8"))["N"]
        idf = {t: math.log((N + 1) / (vocab[t]["df"] + 1)) + 1.0 for t in tfq_w.keys()}
        qvec = {t: tfq_w[t] * idf[t] for t in tfq_w.keys()}
        qnorm = math.sqrt(sum(w*w for w in qvec.values())) or 1.0

        # Acumula producto punto usando sólo postings de términos de la consulta
        acc: Dict[int, float] = defaultdict(float)
        for t, postings in self._iter_postings(list(qvec.keys())):
            qw = qvec[t]
            for doc_i, w in postings:
                acc[doc_i] += qw * w

        doc_ids, norms = self._load_docmap_norms()
        # Top-k con min-heap
        heap: List[Tuple[float, int]] = []  # (score, pk)
        for doc_i, dot in acc.items():
            score = dot / (qnorm * (norms[doc_i] or 1.0))
            pk = doc_ids[doc_i]
            if len(heap) < k:
                heapq.heappush(heap, (score, pk))
            else:
                if score > heap[0][0]:
                    heapq.heapreplace(heap, (score, pk))
        heap.sort(reverse=True)
        return [(pk, sc) for sc, pk in heap]