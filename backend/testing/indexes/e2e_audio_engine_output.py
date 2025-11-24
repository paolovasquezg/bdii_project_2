# -*- coding: utf-8 -*-
"""
Muestra el JSON devuelto por Engine.run para un kNN de audio.
- Usa audios reales de /home/bianca/Downloads/fma_small (hasta 500).
- Crea tabla, inserta via SQL, crea índice USING audio-inverted.
- Ejecuta SELECT ... WHERE file_path KNN <-> AUDIO('ruta') LIMIT 5.
"""
import json
from pathlib import Path
import numpy as np

from backend.storage.audio import AudioStorage

try:
    from backend.engine.engine import Engine
except Exception:  # pragma: no cover
    from engine import Engine  # type: ignore

AUDIO_ROOT = Path("/home/bianca/Downloads/fma_small")


def vec_from_path(p: Path) -> np.ndarray:
    # Usa el mismo generador que AudioStorage para que la query KNN encuentre vecinos
    return AudioStorage.vector_from_path(p)


def main():
    if not AUDIO_ROOT.exists():
        print(json.dumps({"ok": False, "error": f"no existe {AUDIO_ROOT}"}))
        return

    files = [p for p in AUDIO_ROOT.rglob("*") if p.is_file() and p.suffix.lower() in {".mp3", ".wav", ".flac", ".ogg"}]
    files = files[:500]  # limitar para evitar timeouts
    if not files:
        print(json.dumps({"ok": False, "error": "no hay audios"}))
        return

    eng = Engine()
    tbl = "audios_json"
    eng.run(f"DROP TABLE IF EXISTS {tbl};")
    eng.run(
        f"""
        CREATE TABLE {tbl}(
            audio_id INT PRIMARY KEY USING sequential,
            file_name VARCHAR(128),
            file_path VARCHAR(300),
            tfidf_vector VARCHAR(4000)
        );
        """
    )

    for i, p in enumerate(files, start=1):
        v = vec_from_path(p).tolist()
        eng.run(
            f"INSERT INTO {tbl}(audio_id,file_name,file_path,tfidf_vector) "
            f"VALUES ({i},'{p.name}','{p}','{json.dumps(v)}');"
        )

    eng.run(f"CREATE INDEX ON {tbl}(file_name) USING audio-inverted;")

    sql = f"""
    SELECT audio_id, file_name, file_path
    FROM {tbl}
    WHERE file_name KNN <-> AUDIO('{files[0]}')
    LIMIT 5;
    """
    env = eng.run(sql)
    print(json.dumps(env, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
