# test_invtext_knn.py
import json
import sys
from typing import Any, Dict, Iterable, List, Union

from backend.engine.engine import Engine

ENGINE = Engine()

# ---------------- helpers ----------------

def run_sql(sql: str) -> Dict[str, Any]:
    env = ENGINE.run(sql)
    print("\nSQL >>>", sql.strip().splitlines()[0])
    print(json.dumps(env, indent=2, ensure_ascii=False))
    return env

Json = Union[Dict[str, Any], List[Any], tuple, Any]

def _iter_nodes(x: Json) -> Iterable[Any]:
    if isinstance(x, dict):
        for v in x.values():
            yield v
            yield from _iter_nodes(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield v
            yield from _iter_nodes(v)

def rows_from_env(env) -> list:
    """
    Extrae filas desde la respuesta del Engine soportando SELECT y KNN.
    Formatos aceptados:
      - {"results":[{"action":"knn","data":[{...}, ...]}]}
      - {"results":[{"action":"select","data":[...]}]}
      - {"rows":[...]}  / {"data":[...]}
    """
    if not isinstance(env, dict):
        return []

    # 1) formatos directos
    for key in ("rows", "data"):
        if isinstance(env.get(key), list):
            return env[key]

    # 2) resultados del engine (soporta 'knn' y 'select')
    out = []
    results = env.get("results")
    if isinstance(results, list):
        for r in results:
            if not isinstance(r, dict):
                continue
            if r.get("action") in ("knn", "select"):
                data = r.get("data") or r.get("rows")
                if isinstance(data, list):
                    out.extend(data)
    if out:
        return out

    # 3) fallback muy defensivo: busca la 1ra lista asociada a 'data' en cualquier nodo
    def walk(x):
        if isinstance(x, dict):
            # prioriza claves 'data'
            if "data" in x and isinstance(x["data"], list):
                return x["data"]
            for v in x.values():
                got = walk(v)
                if got is not None:
                    return got
        elif isinstance(x, (list, tuple)):
            for v in x:
                got = walk(v)
                if got is not None:
                    return got
        return None

    any_data = walk(env)
    return any_data if isinstance(any_data, list) else []

def extract_ids(rows: List[Any]) -> List[int]:
    ids = []
    for r in rows:
        if isinstance(r, dict) and "id" in r:
            try:
                ids.append(int(r["id"]))
            except Exception:
                pass
        elif isinstance(r, (list, tuple)) and r:
            try:
                ids.append(int(r[0]))
            except Exception:
                pass
    return ids

def expect_any(actual_ids: List[int], expected_any: List[int], label: str) -> bool:
    ok = bool(set(actual_ids) & set(expected_any))
    print(("✓" if ok else "✗"), f"{label}: got {actual_ids}, expected any of {expected_any}")
    return ok

# ---------------- test ----------------

def main():
    # 0) Fixture
    run_sql("DROP TABLE IF EXISTS multimedia;")
    run_sql("""
    CREATE TABLE multimedia (
      id          INT PRIMARY KEY USING heap,
      title       VARCHAR(100),
      image_text  VARCHAR(300)
    );
    """)
    run_sql("""
    INSERT INTO multimedia (id, title, image_text) VALUES
    (1, 'Foto playa',       'playa mar arena verano'),
    (2, 'Foto nieve',       'nieve invierno montaña'),
    (3, 'Surf en playa',    'playa sol arena surf'),
    (4, 'Techno night',     'techno electronica rave'),
    (5, 'Atardecer playa',  'playa atardecer costa');
    """)
    run_sql("""
    CREATE INDEX IF NOT EXISTS multimedia_invtext_image_text
      ON multimedia(image_text) USING invtext;
    """)

    # 1) Consultas con tu gramática
    q1 = "SELECT id, title FROM multimedia WHERE image_text KNN <-> 'playa'  LIMIT 3;"
    q2 = "SELECT id, title FROM multimedia WHERE image_text KNN <-> 'nieve'  LIMIT 1;"
    q3 = "SELECT id, title FROM multimedia WHERE image_text KNN <-> 'techno' LIMIT 1;"
    q4 = "SELECT id, title FROM multimedia WHERE image_text @@ 'playa' LIMIT 3;"

    env1 = run_sql(q1); rows1 = rows_from_env(env1); ids1 = extract_ids(rows1)
    env2 = run_sql(q2); rows2 = rows_from_env(env2); ids2 = extract_ids(rows2)
    env3 = run_sql(q3); rows3 = rows_from_env(env3); ids3 = extract_ids(rows3)
    env4 = run_sql(q4); rows4 = rows_from_env(env4); ids4 = extract_ids(rows4)

    # 2) Checks
    ok = True
    ok &= expect_any(ids1, [1, 3, 5], "KNN 'playa'")
    ok &= expect_any(ids2, [2],       "KNN 'nieve'")
    ok &= expect_any(ids3, [4],       "KNN 'techno'")
    print(("✓" if rows4 else "✗"), f"@@ 'playa' -> {ids4 or rows4}")

    # 3) Exit code
    print("\nResumen:", "OK" if ok and rows4 else "FALLÓ")
    sys.exit(0 if (ok and rows4) else 1)

if __name__ == "__main__":
    main()
