"""
Benchmark comparando KNN indexado (<->) vs no indexado (<-->) con BoVW.
- Usa imágenes en backend/data/images (nombres *.jpg).
- Escalas probadas: 1k, 2k, 4k, 8k, 16k, 32k, 64k (se salta si no hay suficientes).
- Genera CSV, crea tabla + índices, ejecuta KNN indexado y no indexado, y guarda:
    * backend/testing/benchmark/bovw/results.csv
    * backend/testing/benchmark/bovw/results.png (si matplotlib está disponible)

Ejecutar desde raíz del repo:
    python -m backend.testing.benchmark.bovw
    # o directo: python backend/testing/benchmark/bovw.py
"""

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

# Permitir ejecución directa: python backend/testing/benchmark/bovw.py
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from backend.engine.engine import Engine


SIZES = [1000, 2000, 4000, 8000, 16000, 32000, 64000]
BACKEND_DIR = Path(__file__).resolve().parents[2]
IMAGES_DIR = BACKEND_DIR / "data" / "images"
OUT_DIR = BACKEND_DIR / "testing" / "benchmark" / "bovw"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _available_images() -> List[str]:
    return sorted(p.name for p in IMAGES_DIR.glob("*.jpg"))


def _csv_path_for(n: int) -> Path:
    return OUT_DIR / f"images_{n}.csv"


def _timed_run(eng: Engine, sql: str) -> Dict[str, float]:
    t0 = time.perf_counter()
    res = eng.run(sql)
    dt = (time.perf_counter() - t0) * 1000.0
    return {"ok": bool(res.get("ok", False)), "ms": dt, "raw": res}


def benchmark_size(n: int, eng: Engine, names: List[str]) -> Dict[str, object]:
    if len(names) < n:
        return {"size": n, "indexed_ms": None, "seq_ms": None, "note": f"SKIP: solo {len(names)} imgs"}

    csv_path = _csv_path_for(n)
    if not csv_path.exists():
        return {"size": n, "indexed_ms": None, "seq_ms": None, "note": f"CSV no encontrado: {csv_path.name}"}
    table = f"fashion_knn_bench_{n}"

    eng.run(f"DROP TABLE IF EXISTS {table};")
    eng.run(
        f"""
        CREATE TABLE {table}(
            id INT PRIMARY KEY USING heap,
            title VARCHAR(128),
            image_path VARCHAR(512),
            image_text VARCHAR(256)
        );
        """
    )
    imp = eng.run(f"CREATE TABLE {table} FROM FILE '{csv_path.as_posix()}';")
    if not imp.get("ok", False):
        return {"size": n, "indexed_ms": None, "seq_ms": None, "note": "IMPORT FAIL"}

    res_idx1 = eng.run(f"CREATE INDEX ON {table}(image_path) USING bovw;")
    if (not res_idx1.get("ok", False)):
        return {"size": n, "indexed_ms": None, "seq_ms": None, "note": "INDEX FAIL"}

    probe = f"images/{names[0]}"
    q_idx = (
        f"SELECT id,title,image_path,similarity FROM {table} "
        f"WHERE image_path KNN <-> IMG('{probe}') LIMIT 5;"
    )
    q_seq = (
        f"SELECT id,title,image_path,similarity FROM {table} "
        f"WHERE image_path KNN <--> IMG('{probe}') LIMIT 5;"
    )

    r_idx = _timed_run(eng, q_idx)
    r_seq = _timed_run(eng, q_seq)

    return {
        "size": n,
        "indexed_ms": r_idx["ms"],
        "seq_ms": r_seq["ms"],
        "indexed_ok": r_idx["ok"],
        "seq_ok": r_seq["ok"],
        "note": "",
    }


def save_results(rows: List[Dict[str, object]]):
    csv_path = OUT_DIR / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["size", "indexed_ms", "seq_ms", "indexed_ok", "seq_ok", "note"])
        w.writeheader()
        w.writerows(rows)
    return csv_path


def plot_results(rows: List[Dict[str, object]]):
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        print("matplotlib no disponible; se omite gráfico.")
        return None

    sizes = [r["size"] for r in rows if r.get("indexed_ms") is not None]
    idx_ms = [r["indexed_ms"] for r in rows if r.get("indexed_ms") is not None]
    seq_ms = [r["seq_ms"] for r in rows if r.get("indexed_ms") is not None]

    plt.figure(figsize=(8, 4.5))
    plt.plot(sizes, idx_ms, marker="o", label="KNN indexado (<->)")
    plt.plot(sizes, seq_ms, marker="o", label="KNN no indexado (<-->)")
    plt.xlabel("N registros")
    plt.ylabel("Tiempo (ms)")
    plt.title("BoVW: <-> vs <--> por tamaño de tabla")
    plt.xscale("log", basex=2)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    out_png = OUT_DIR / "results.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    return out_png


def main():
    names = _available_images()
    if not names:
        print("No hay imágenes en backend/data/images")
        return 1

    eng = Engine()
    all_rows = []
    # for n in SIZES:
    #     print(f"=== N={n} ===")
    #     row = benchmark_size(n, eng, names)
    #     print(row)
    #     all_rows.append(row)
    #     # guardar progreso parcial para poder graficar si se interrumpe
    #     save_results(all_rows)

    csv_path = save_results(all_rows)
    png_path = plot_results(all_rows)
    print(f"CSV: {csv_path}")
    if png_path:
        print(f"PNG: {png_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
