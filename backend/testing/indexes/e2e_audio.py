# -*- coding: utf-8 -*-
"""
E2E • Audio (índice invertido de vectores TF-IDF) usando archivos reales.
- Toma audios de DATA_DIR/fma_small (hasta 6).
- Inserta vía SQL, crea índice USING audio.
- Ejecuta kNN con AudioStorage (sin usar File.knn).
"""
import sys
from pathlib import Path
import json

import numpy as np

try:
    from backend.engine.engine import Engine
    from backend.storage.audio import AudioStorage, AudioRecord
except Exception:  # pragma: no cover
    from engine import Engine  # type: ignore
    from storage.audio import AudioStorage, AudioRecord  # type: ignore


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIO_ROOT = PROJECT_ROOT / "data" / "fma_small"


def vec_from_path(p: Path, dim: int = 16) -> np.ndarray:
    seed = int.from_bytes(p.as_posix().encode("utf-8")[:8], "little", signed=False)
    rng = np.random.default_rng(seed)
    return np.abs(rng.standard_normal(dim))


def main() -> int:
    if not AUDIO_ROOT.exists():
        print(f"✗ no existe {AUDIO_ROOT}")
        return 1

    files = [p for p in AUDIO_ROOT.rglob("*") if p.is_file() and p.suffix.lower() in {".mp3", ".wav", ".flac", ".ogg"}]
    files = files[:6]
    if not files:
        print("✗ no se encontraron audios en el path dado")
        return 1

    eng = Engine()
    ok_all = True

    for pk in ("heap", "sequential"):
        tbl = f"audios_audio_{pk}"
        eng.run(f"DROP TABLE IF EXISTS {tbl};")
        eng.run(
            f"""
            CREATE TABLE {tbl}(
                audio_id INT PRIMARY KEY USING {pk},
                file_name VARCHAR(128),
                file_path VARCHAR(300),
                tfidf_vector VARCHAR(4000)
            );
            """
        )

        vals = []
        for i, p in enumerate(files, start=1):
            v = vec_from_path(p, dim=32).tolist()
            vals.append((i, p.name, str(p), json.dumps(v)))

        for v in vals:
            eng.run(
                f"INSERT INTO {tbl}(audio_id,file_name,file_path,tfidf_vector) "
                f"VALUES ({v[0]},'{v[1]}','{v[2]}','{v[3]}');"
            )

        res_idx = eng.run(f"CREATE INDEX ON {tbl}(file_path) USING audio;")
        if not res_idx.get("ok", False):
            print(f"✗ create index audio ({pk})")
            ok_all = False
            continue

        store = AudioStorage(tbl)
        try:
            store.load_inverted_index()
        except Exception:
            store.build_inverted_index()
        qv = vec_from_path(files[0], dim=32)
        res, _ = store.knn_indexed(qv, k=3)
        ids = [r.get("audio_id") for r in res if isinstance(r, dict)]
        if not ids or files[0].name not in [r.get("file_name") for r in res]:
            print(f"✗ knn audio ({pk}) ->", res)
            ok_all = False
        else:
            print(f"✓ audio {pk} -> ids {ids}")

    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
