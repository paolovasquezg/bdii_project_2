# -*- coding: utf-8 -*-
"""
E2E • InvText (texto) sólo con SQL
- Crea tabla (id, title, tags) probando 4 backends de PK: heap, sequential, isam, bplus.
- Inserta data por SQL (fallback a File.insert si tu motor aún no soporta INSERT SQL).
- Crea índice secundario USING invtext en (tags).
- Corre kNN por SQL: WHERE tags KNN <-> 'consulta'.
- Valida que el índice exista en runtime/ y que los top-k contengan IDs esperados.
- NO usa carpetas temporales; todo queda en runtime/.
"""
import os, sys, time, json
from pathlib import Path

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Engine (layout dual)
try:
    from backend.engine.engine import Engine
except Exception:
    from engine import Engine  # type: ignore

ENGINE = Engine()
PK_METHODS = ["heap", "sequential", "isam", "bplus"]

CHECK = "✓"
CROSS = "✗"

def _first_result(env: dict):
    if isinstance(env, dict) and env.get("results"):
        return env["results"][0]
    return env

def _short_plan(res0: dict) -> str:
    p = res0.get("plan") or {}
    act = p.get("action")
    tbl = p.get("table")
    fld = p.get("field")
    if act and tbl and fld:
        return f"{act}:{tbl}.{fld}"
    if act and tbl:
        return f"{act}:{tbl}"
    return act or "-"

def _short_usage(meta: dict) -> str:
    if not isinstance(meta, dict):
        return "-"
    t = meta.get("time_ms")
    idx = meta.get("index_usage") or []
    used = "index:none"
    if idx:
        parts = []
        for u in idx:
            where = u.get("where") or u.get("kind")
            ind = u.get("index") or u.get("method")
            field = u.get("field")
            op = u.get("op")
            if where and ind and field:
                parts.append(f"{where}:{ind}({field})" + (f"/{op}" if op else ""))
        if parts:
            used = ", ".join(parts)
    return f"{used}; {t:.2f}ms" if t is not None else used

def run_sql(sql: str) -> dict:
    env = ENGINE.run(sql)
    return env

def assert_ok(env: dict, msg=""):
    if not isinstance(env, dict) or not env.get("ok", False):
        raise AssertionError(f"Envelope NOT ok: {msg or '-'}")
    r0 = _first_result(env)
    if not r0.get("ok", False):
        err = r0.get("error") or {}
        raise AssertionError(f"Result NOT ok: {msg or '-'} • {err.get('code','?')}: {err.get('message','-')}")
    return r0

def _print_step_ok(label, res0):
    meta = res0.get("meta") or {}
    print(f"  {CHECK} {label}  • plan={_short_plan(res0)}  • {_short_usage(meta)}")

def _print_step_bad(label, env_or_res0, reason):
    if isinstance(env_or_res0, dict) and env_or_res0.get("results"):
        res0 = env_or_res0["results"][0]
        meta = res0.get("meta") or {}
        print(f"  {CROSS} {label}  • plan={_short_plan(res0)}  • {_short_usage(meta)}")
    else:
        print(f"  {CROSS} {label}")
    print(f"      └─ {reason}")

def create_table(tbl: str, pk_method: str):
    res0 = assert_ok(run_sql(f"""
        CREATE TABLE {tbl}(
            id INT PRIMARY KEY USING {pk_method},
            title VARCHAR(128),
            tags  VARCHAR(512)
        );
    """), msg="create table")
    _print_step_ok("CREATE TABLE", res0)

def insert_rows_sql_or_fallback(tbl: str, rows):
    inserted_sql = True
    try:
        last = None
        for rid, title, tags in rows:
            tq = title.replace("'", "''")
            gq = tags.replace("'", "''")
            last = assert_ok(run_sql(
                f"INSERT INTO {tbl}(id,title,tags) VALUES ({rid}, '{tq}', '{gq}');"
            ), msg=f"insert {rid}")
        _print_step_ok("INSERT (SQL)", last or {})
    except AssertionError as e:
        inserted_sql = False
        _print_step_bad("INSERT (SQL)", {}, f"falló, usando fallback File.insert • {e}")

    if not inserted_sql:
        try:
            try:
                from backend.storage.file import File
            except Exception:
                from storage.file import File  # type: ignore
            F = File(tbl)
            for rid, title, tags in rows:
                F.execute({"op":"insert", "record":{"id": rid, "title": title, "tags": tags}})
            print(f"  {CHECK} INSERT fallback (File.insert) • ok")
        except Exception as e:
            raise AssertionError(f"Fallback File.insert falló: {e}")

def create_invtext_index(tbl: str):
    env = run_sql(f"CREATE INDEX ON {tbl}(tags) USING invtext;")
    if not env.get("ok", False) or not _first_result(env).get("ok", False):
        raise AssertionError(f"CREATE INDEX invtext NOT ok: {json.dumps(env, indent=2, ensure_ascii=False)}")
    _print_step_ok("CREATE INDEX invtext(tags)", _first_result(env))

    # sanity: la carpeta base del índice debe existir en runtime
    try:
        try:
            from backend.storage.file import File
        except Exception:
            from storage.file import File  # type: ignore
        F = File(tbl)
        idx = F.indexes
        base = Path(idx["tags"]["filename"])
        if not base.exists() or not base.is_dir():
            raise AssertionError(f"No existe el directorio del índice: {base}")
        print(f"  {CHECK} InvText artifacts • ok ({base})")
    except Exception as e:
        raise AssertionError(f"No pude validar artefactos InvText: {e}")

def query_knn_text_sql(tbl: str, query: str, expected_ids_any: set, k=3):
    q = query.replace("'", "''")
    env = run_sql(f"""
        SELECT id, title
        FROM {tbl}
        WHERE tags KNN <-> '{q}'
        LIMIT {k};
    """)
    res0 = assert_ok(env, msg="knn text sql")
    rows = res0.get("data") or res0.get("rows") or []
    ids = {int(r["id"]) for r in rows if "id" in r}
    if not (ids & expected_ids_any):
        raise AssertionError(f"kNN(text) esperaba alguno de {sorted(expected_ids_any)}, got {sorted(ids)}")
    _print_step_ok("kNN TEXT (SQL)", res0)
    print(f"     └─ rows:{len(rows)} ids:{sorted(ids)}")

def run_step(tbl: str, label: str, func, failures: list):
    t0 = time.perf_counter()
    try:
        func()
        dt = (time.perf_counter() - t0) * 1000
        print(f"→ {CHECK} {label}  ({dt:.1f} ms)")
    except AssertionError as e:
        dt = (time.perf_counter() - t0) * 1000
        print(f"→ {CROSS} {label}  ({dt:.1f} ms)")
        print(f"    └─ {e}")
        failures.append((tbl, label, str(e)))
    except Exception as e:
        dt = (time.perf_counter() - t0) * 1000
        print(f"→ {CROSS} {label}  ({dt:.1f} ms)")
        print(f"    └─ EXCEPTION {type(e).__name__}: {e}")
        failures.append((tbl, label, f"EXCEPTION {type(e).__name__}: {e}"))

def main():
    global_failures = []

    for pk in PK_METHODS:
        tbl = f"invtext_items_{pk}"
        failures = []

        print("\n" + "="*70)
        print(f"E2E • InvText con PK USING {pk} • tabla {tbl}")
        print("="*70)

        run_sql(f"DROP TABLE IF EXISTS {tbl};")
        run_step(tbl, "CREATE TABLE", lambda: create_table(tbl, pk), failures)

        rows = [
            (1, "Playa",            "mar sol arena surfing"),
            (2, "Montaña",          "trekking nieve frio paisaje"),
            (3, "Atardecer playa",  "sunset mar naranja playa"),
            (4, "Techno",           "musica festival techno edm"),
            (5, "Ceviche",          "comida mar pescado limon"),
            (6, "Bosque",           "arboles sendero hojas naturaleza"),
        ]
        run_step(tbl, "INSERT DATA", lambda: insert_rows_sql_or_fallback(tbl, rows), failures)
        run_step(tbl, "CREATE INDEX INVTEXT", lambda: create_invtext_index(tbl), failures)

        # consultas: deben devolver alguno de los esperados en top-k
        run_step(tbl, "KNN(SQL) playa",   lambda: query_knn_text_sql(tbl, "playa mar atardecer", {1,3,5}, k=3), failures)
        run_step(tbl, "KNN(SQL) nieve",   lambda: query_knn_text_sql(tbl, "trekking nieve", {2}, k=3), failures)
        run_step(tbl, "KNN(SQL) techno",  lambda: query_knn_text_sql(tbl, "techno edm", {4}, k=3), failures)

        if failures:
            print(f"\n⟡ Resumen {tbl}: {len(failures)} fallos")
            for _, label, msg in failures:
                print(f"   - {label}: {msg}")
        else:
            print(f"✔ E2E {tbl} OK")

        global_failures.extend(failures)

    if global_failures:
        print("\n" + "#" * 70)
        print(f"RESUMEN GLOBAL: {len(global_failures)} fallos")
        for tbl, label, msg in global_failures:
            print(f" - {tbl} :: {label} -> {msg}")
        print("#" * 70)
        sys.exit(1)
    else:
        print("\n✅ E2E InvText (texto) en todos los backends de PK: OK.")
        sys.exit(0)

if __name__ == "__main__":
    main()
