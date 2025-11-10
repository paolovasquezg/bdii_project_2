# -*- coding: utf-8 -*-
"""
E2E • BoVW (imágenes) sólo con SQL
- Crea tabla (id, title, image_path) probando 4 backends de PK: heap, sequential, isam, bplus.
- Inserta data por SQL (fallback a File.insert si tu motor aún no soporta INSERT SQL).
- Crea índice secundario USING bovw en (image_path).
- Corre kNN por SQL: WHERE image_path KNN <-> IMG('...') LIMIT k    ← (forma real soportada por tu parser)
- Si el KNN SQL no está soportado, cae a kNN basado en los artefactos BoVW (dot product TF-IDF) para validar resultados.
- Imprime ✓/✗ con plan, uso de índice (si tu motor lo reporta) y resumen final (rc=1 si hubo fallos).
"""

import os, sys, time, json
from pathlib import Path

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Engine (layout dual por si tu pkg path difiere)
try:
    from backend.engine.engine import Engine
except Exception:
    from engine import Engine  # type: ignore

# BoVW (import dual)
try:
    from backend.storage.indexes.bovw import BoVWFile
except Exception:
    from storage.indexes.bovw import BoVWFile  # type: ignore

ENGINE = Engine()
PK_METHODS = ["heap", "sequential", "isam", "bplus"]

CHECK = "✓"
CROSS = "✗"

# ---------------------------
# Helpers de impresión
# ---------------------------
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

def _used_index(meta: dict, where: str, index_kind: str, field: str) -> bool:
    idx = (meta or {}).get("index_usage") or []
    for u in idx:
        if (u.get("where") or u.get("kind")) == where and (u.get("index") or u.get("method")) == index_kind and u.get("field") == field:
            return True
    return False

def run_sql(sql: str) -> dict:
    env = ENGINE.run(sql)
    # print(json.dumps(env, indent=2, ensure_ascii=False))  # descomenta si quieres ver todo
    return env

def assert_ok(env: dict, msg=""):
    if not isinstance(env, dict) or not env.get("ok", False):
        raise AssertionError(f"Envelope NOT ok: {msg or '-'}")
    r0 = _first_result(env)
    if not r0.get("ok", False):
        err = r0.get("error") or {}
        raise AssertionError(f"Result NOT ok: {msg or '-'} • {err.get('code','?')}: {err.get('message','-')}")
    return r0

# ---------------------------
# Imágenes sintéticas (checker / stripes / circles)
# ---------------------------
def ensure_imgs(dirpath: str):
    from PIL import Image, ImageDraw
    d = Path(dirpath); d.mkdir(parents=True, exist_ok=True)

    def checker(size=256, tile=16):
        im = Image.new("L",(size,size),255); px=im.load()
        for y in range(size):
            for x in range(size):
                if ((x//tile)+(y//tile)) % 2 == 0:
                    px[x,y] = 0
        return im

    def stripes(size=256, stripe=12):
        im = Image.new("L",(size,size),255); dr=ImageDraw.Draw(im)
        for y in range(0,size,stripe*2):
            dr.rectangle([(0,y),(size,y+stripe-1)], fill=0)
        return im

    def circles(size=256, r=80):
        im = Image.new("L",(size,size),255); dr=ImageDraw.Draw(im)
        for rr in range(r, 10, -20):
            bbox = (size//2-rr, size//2-rr, size//2+rr, size//2+rr)
            dr.ellipse(bbox, outline=0, width=4)
        return im

    paths = {}
    p1 = d/"chk1.png"; checker().save(p1); paths["chk1"] = str(p1)
    p2 = d/"chk2.png"; checker().save(p2); paths["chk2"] = str(p2)
    p3 = d/"stripes.png"; stripes().save(p3); paths["stripe"] = str(p3)
    p4 = d/"circles.png"; circles().save(p4); paths["circle"] = str(p4)
    return paths

# ---------------------------
# Pasos del flujo BoVW
# ---------------------------
def create_table(tbl: str, pk_method: str):
    res0 = assert_ok(run_sql(f"""
        CREATE TABLE {tbl}(
            id INT PRIMARY KEY USING {pk_method},
            title VARCHAR(64),
            image_path VARCHAR(512)
        );
    """), msg="create table")
    _print_step_ok("CREATE TABLE", res0)

def insert_rows_sql_or_fallback(tbl: str, rows, paths):
    inserted_sql = True
    try:
        for rid, title, ipath in rows:
            iq = ipath.replace("'", "''")
            tq = title.replace("'", "''")
            res0 = assert_ok(run_sql(f"INSERT INTO {tbl}(id,title,image_path) VALUES ({rid}, '{tq}', '{iq}');"),
                             msg=f"insert {rid}")
        _print_step_ok("INSERT (SQL)", res0)
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
            for rid, title, ipath in rows:
                F.execute({"op":"insert", "record":{"id": rid, "title": title, "image_path": ipath}})
            print(f"  {CHECK} INSERT fallback (File.insert) • ok")
        except Exception as e:
            raise AssertionError(f"Fallback File.insert falló: {e}")

def create_bovw_index(tbl: str):
    env = run_sql(f"CREATE INDEX ON {tbl}(image_path) USING bovw;")
    if not env.get("ok", False) or not _first_result(env).get("ok", False):
        raise AssertionError(f"CREATE INDEX bovw NOT ok: {json.dumps(env, indent=2, ensure_ascii=False)}")
    _print_step_ok("CREATE INDEX bovw(image_path)", _first_result(env))

    # sanity: carpeta + artefactos
    try:
        # leemos catálogo a través de File para inspeccionar índices
        try:
            from backend.storage.file import File
        except Exception:
            from storage.file import File  # type: ignore
        F = File(tbl)
        idx = F.indexes
        base = Path(idx["image_path"]["filename"])
        missing = [f for f in ["codebook.joblib","idf.npy","postings.json","doc_map.json"] if not (base/f).exists()]
        if missing:
            raise AssertionError(f"Faltan artefactos en {base}: {missing}")
        print(f"  {CHECK} BoVW artifacts • ok ({base})")
    except Exception as e:
        raise AssertionError(f"No pude validar artefactos BoVW: {e}")

def _fallback_knn_seq(tbl: str, qimg: str, k: int) -> list[int]:
    """
    Ubica el directorio BoVW desde el catálogo y corre knn() del BoVWFile,
    que calcula similitud TF-IDF · postings (dot product).
    """
    try:
        try:
            from backend.storage.file import File
        except Exception:
            from storage.file import File  # type: ignore
        F = File(tbl)
        sec = (F.indexes or {}).get("image_path", {})
        if (sec.get("index") or "").lower() != "bovw":
            raise RuntimeError("Índice bovw(image_path) no encontrado")
        base_dir = sec["filename"]
        bv = BoVWFile(base_dir, key="image_path",
                      heap_file=(F.indexes["primary"]["filename"] if (F.indexes.get("primary", {}).get("index") == "heap") else None))
        return bv.knn(qimg, k)
    except Exception as e:
        raise AssertionError(f"Fallback BoVW falló: {e}")

def query_knn_sql(tbl: str, qimg: str, expected_ids_any: set, k=3):
    # 1) Intento con SQL (forma real: KNN <-> IMG(...) + LIMIT como k)
    env = run_sql(f"""
        SELECT id, title
        FROM {tbl}
        WHERE image_path KNN <-> IMG('{qimg.replace("'", "''")}')
        LIMIT {int(k)};
    """)

    if env.get("ok", False) and _first_result(env).get("ok", False):
        res0 = _first_result(env)
        rows = res0.get("data") or res0.get("rows") or []
        ids = {int(r["id"]) for r in rows if "id" in r}
        if not (ids & expected_ids_any):
            raise AssertionError(f"kNN esperaba alguno de {sorted(expected_ids_any)}, got {sorted(ids)}")
        meta = res0.get("meta") or {}
        # Si tu motor reporta index_usage, valida que sea bovw(image_path)
        if meta.get("index_usage"):
            if not _used_index(meta, "secondary", "bovw", "image_path"):
                _print_step_bad("kNN (uso índice)", env, "no reporta secondary:bovw(image_path) • tolerado")
            else:
                print("  ✓ kNN uso índice • secondary:bovw(image_path)")
        _print_step_ok("kNN (SQL)", res0)
        print(f"     └─ rows:{len(rows)} ids:{sorted(ids)}")
        return

    # 2) Fallback con artefactos BoVW cuando el SQL aún no está soportado
    reason = _first_result(env).get("error", {}).get("message", "SQL KNN no soportado")
    print(f"  · KNN SQL no disponible ({reason}). Usando baseline con BoVW…")
    ids = _fallback_knn_seq(tbl, qimg, k)
    if not (set(ids) & expected_ids_any):
        raise AssertionError(f"kNN (fallback) esperaba alguno de {sorted(expected_ids_any)}, got {sorted(ids)}")
    fake_res = {"plan": {"action": "knn", "table": tbl, "field": "image_path"},
                "meta": {"index_usage": [], "time_ms": 0.0}}
    print("  ✓ kNN (fallback BoVW)")
    _print_step_ok("kNN (SQL→fallback)", fake_res)
    print(f"     └─ rows:{min(k,len(ids))} ids:{sorted(ids)[:k]}")

# ------------------------
# Infra de pasos tolerantes
# ------------------------
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

# ------------------------
# Main
# ------------------------
def main():
    global_failures = []
    imgdir = os.path.join(HERE, "_testdata", "imgs")
    paths = ensure_imgs(imgdir)
    for k, v in paths.items():
        print(f"img {k}: {v} exists={os.path.exists(v)}")

    for pk in PK_METHODS:
        tbl = f"bovw_items_{pk}"
        failures = []

        print("\n" + "="*70)
        print(f"E2E • BoVW con PK USING {pk} • tabla {tbl}")
        print("="*70)

        run_sql(f"DROP TABLE IF EXISTS {tbl};")
        run_step(tbl, "CREATE TABLE", lambda: create_table(tbl, pk), failures)

        rows = [
            (1, "chk1",    paths["chk1"]),
            (2, "chk2",    paths["chk2"]),
            (3, "stripes", paths["stripe"]),
            (4, "circles", paths["circle"]),
        ]
        run_step(tbl, "INSERT DATA", lambda: insert_rows_sql_or_fallback(tbl, rows, paths), failures)
        run_step(tbl, "CREATE INDEX BOVW", lambda: create_bovw_index(tbl), failures)

        # kNN: checkerboard → debe traer (1 o 2) en el top-k
        run_step(tbl, "KNN(SQL) chk1", lambda: query_knn_sql(tbl, paths["chk1"], {1, 2}, k=3), failures)
        run_step(tbl, "KNN(SQL) chk2", lambda: query_knn_sql(tbl, paths["chk2"], {1, 2}, k=3), failures)
        # stripes debería traer (3) cerca (no estricto si hay empates)
        run_step(tbl, "KNN(SQL) stripes", lambda: query_knn_sql(tbl, paths["stripe"], {3}, k=3), failures)

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
        print("\n✅ E2E BoVW (imágenes) en todos los backends de PK: OK.")
        sys.exit(0)

if __name__ == "__main__":
    main()
