# backend/testing/indexes/test_invtext_pk_variants.py
import json
import sys
from typing import Any, Dict, Iterable, List, Union

from backend.engine.engine import Engine

ENGINE = Engine()

# ---------------- helpers ----------------

def run_sql(sql: str) -> Dict[str, Any]:
    env = ENGINE.run(sql)
    # imprime solo la primera línea para ubicar el bloque
    first = sql.strip().splitlines()[0] if sql.strip() else sql
    print("\nSQL >>>", first)
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

def rows_from_env(env: Dict[str, Any]) -> List[Any]:
    """
    Extrae filas soportando SELECT y KNN.
    Formatos aceptados:
      - {"results":[{"action":"knn","data":[{...}, ...]}]}
      - {"results":[{"action":"select","data":[...]}]}
      - {"rows":[...]}  / {"data":[...]}
    """
    if not isinstance(env, dict):
        return []

    # directos
    for key in ("rows", "data"):
        if isinstance(env.get(key), list):
            return env[key]  # type: ignore

    # results del engine
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

    # fallback defensivo
    def walk(x):
        if isinstance(x, dict):
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

def used_invtext(env: Dict[str, Any], field: str = "image_text") -> bool:
    """Confirma que el plan reportó uso del índice invtext en esa consulta."""
    for r in env.get("results", []):
        meta = r.get("meta", {})
        for u in meta.get("index_usage", []):
            if u.get("index") == "invtext" and u.get("field") == field and u.get("op") in ("knn", "search"):
                return True
    return False

# ---------------- fixture & checks ----------------

def setup_table(pk_method: str, tname: str):
    run_sql(f"DROP TABLE IF EXISTS {tname};")
    run_sql(f"""
    CREATE TABLE {tname} (
      id          INT PRIMARY KEY USING {pk_method},
      title       VARCHAR(100),
      image_text  VARCHAR(300)
    );
    """)
    run_sql(f"""
    INSERT INTO {tname} (id, title, image_text) VALUES
    (1, 'Foto playa',       'playa mar arena verano'),
    (2, 'Foto nieve',       'nieve invierno montaña'),
    (3, 'Surf en playa',    'playa sol arena surf'),
    (4, 'Techno night',     'techno electronica rave'),
    (5, 'Atardecer playa',  'playa atardecer costa');
    """)
    run_sql(f"""
    CREATE INDEX IF NOT EXISTS ix_invtext_{tname}_image_text
      ON {tname}(image_text) USING invtext;
    """)

def run_checks(pk_method: str) -> bool:
    tname = f"multimedia_{pk_method}"
    print(f"\n==============================")
    print(f"E2E • InvText KNN • {tname}")
    print(f"==============================")
    setup_table(pk_method, tname)

    ok = True

    # Consultas con tu gramática
    q_playa = f"SELECT id, title FROM {tname} WHERE image_text KNN <-> 'playa'  LIMIT 3;"
    q_nieve = f"SELECT id, title FROM {tname} WHERE image_text KNN <-> 'nieve'  LIMIT 1;"
    q_tech  = f"SELECT id, title FROM {tname} WHERE image_text KNN <-> 'techno' LIMIT 1;"
    q_at    = f"SELECT id, title FROM {tname} WHERE image_text @@ 'playa' LIMIT 3;"

    env1 = run_sql(q_playa); rows1 = rows_from_env(env1); ids1 = extract_ids(rows1)
    env2 = run_sql(q_nieve); rows2 = rows_from_env(env2); ids2 = extract_ids(rows2)
    env3 = run_sql(q_tech);  rows3 = rows_from_env(env3); ids3 = extract_ids(rows3)
    env4 = run_sql(q_at);    rows4 = rows_from_env(env4); ids4 = extract_ids(rows4)

    ok &= expect_any(ids1, [1, 3, 5], f"{pk_method} :: KNN 'playa'")
    ok &= expect_any(ids2, [2],       f"{pk_method} :: KNN 'nieve'")
    ok &= expect_any(ids3, [4],       f"{pk_method} :: KNN 'techno'")
    print(("✓" if rows4 else "✗"), f"{pk_method} :: @@ 'playa' -> {ids4 or rows4}")

    # Uso de índice invtext reportado por el engine
    inv_used = used_invtext(env1) and used_invtext(env2) and used_invtext(env3)
    print(("✓" if inv_used else "✗"), f"{pk_method} :: uso de índice invtext reportado")
    ok &= inv_used

    return ok

def main():
    backends = ["heap", "sequential", "isam", "bplus"]
    all_ok = True
    for pk in backends:
        try:
            ok = run_checks(pk)
        except Exception as ex:
            ok = False
            print(f"✗ excepción en backend {pk}: {ex}")
        all_ok = all_ok and ok

    print("\n==============================")
    print("RESUMEN GLOBAL:", "✓ OK" if all_ok else "✗ FALLÓ ALGUNO")
    print("==============================")
    sys.exit(0 if all_ok else 1)

if __name__ == "__main__":
    main()