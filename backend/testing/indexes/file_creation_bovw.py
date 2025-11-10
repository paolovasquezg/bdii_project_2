# backend/testing/indexes/test_bovw_runtime.py
# -*- coding: utf-8 -*-
import os
from pathlib import Path
import pytest

# --- Imports duales para que funcione igual si corres desde repo raíz o desde paquete ---
try:
    from backend.engine.engine import Engine
except Exception:  # pragma: no cover
    from engine import Engine  # type: ignore

try:
    from backend.storage.file import File
except Exception:  # pragma: no cover
    from storage.file import File  # type: ignore


PK_METHODS = ["heap", "sequential", "isam", "bplus"]

HERE = Path(__file__).resolve().parent
ROOT = (HERE / ".." / "..").resolve()
RUNTIME_DIR = (ROOT / "runtime" / "files").resolve()   # <<— destino “del sistema”, no tmp
IMG_DIR = (HERE / "_testdata" / "imgs").resolve()


def ensure_imgs(dirpath: Path):
    """Genera 4 imágenes sintéticas y las guarda en testing/_testdata/imgs (no tmp)."""
    from PIL import Image, ImageDraw
    d = Path(dirpath)
    d.mkdir(parents=True, exist_ok=True)

    def checker(size=256, tile=16):
        im = Image.new("L", (size, size), 255)
        px = im.load()
        for y in range(size):
            for x in range(size):
                if ((x // tile) + (y // tile)) % 2 == 0:
                    px[x, y] = 0
        return im

    def stripes(size=256, stripe=12):
        im = Image.new("L", (size, size), 255)
        dr = ImageDraw.Draw(im)
        for y in range(0, size, stripe * 2):
            dr.rectangle([(0, y), (size, y + stripe - 1)], fill=0)
        return im

    def circles(size=256, r=80):
        im = Image.new("L", (size, size), 255)
        dr = ImageDraw.Draw(im)
        for rr in range(r, 10, -20):
            bbox = (size // 2 - rr, size // 2 - rr, size // 2 + rr, size // 2 + rr)
            dr.ellipse(bbox, outline=0, width=4)
        return im

    p1 = d / "chk1.png"; checker().save(p1)
    p2 = d / "chk2.png"; checker().save(p2)
    p3 = d / "stripes.png"; stripes().save(p3)
    p4 = d / "circles.png"; circles().save(p4)

    return {
        "chk1": str(p1),
        "chk2": str(p2),
        "stripe": str(p3),
        "circle": str(p4),
    }


@pytest.fixture(scope="module")
def engine():
    return Engine()


@pytest.fixture(scope="module")
def img_paths():
    return ensure_imgs(IMG_DIR)


def _run(engine: Engine, sql: str) -> dict:
    env = engine.run(sql)
    assert isinstance(env, dict), "engine.run debe devolver dict envelope"
    return env


def _assert_ok(env: dict, ctx: str) -> dict:
    assert env.get("ok", False), f"Envelope NOT ok: {ctx}"
    res0 = env["results"][0]
    assert res0.get("ok", False), f"Result NOT ok ({ctx}): {res0.get('error')}"
    return res0


@pytest.mark.parametrize("pk", PK_METHODS)
def test_bovw_runtime_artifacts(engine: Engine, img_paths: dict, pk: str):
    tbl = f"bovw_sys_{pk}"

    # Arranque limpio
    _run(engine, f"DROP TABLE IF EXISTS {tbl};")

    # 1) CREATE TABLE (PK USING <pk>)
    res = _run(engine, f"""
        CREATE TABLE {tbl}(
            id INT PRIMARY KEY USING {pk},
            title VARCHAR(64),
            image_path VARCHAR(512)
        );
    """)
    _assert_ok(res, "create table")

    # 2) INSERT (SQL)
    rows_sql = (
        f"INSERT INTO {tbl}(id,title,image_path) VALUES "
        f"(1,'chk1','{img_paths['chk1']}'),"
        f"(2,'chk2','{img_paths['chk2']}'),"
        f"(3,'stripes','{img_paths['stripe']}'),"
        f"(4,'circles','{img_paths['circle']}');"
    )
    _assert_ok(_run(engine, rows_sql), "insert rows")

    # 3) CREATE INDEX ... USING bovw
    _assert_ok(_run(engine, f"CREATE INDEX ON {tbl}(image_path) USING bovw;"),
               "create bovw index")

    # 4) Validar que el índice quedó en runtime (NO tmp)
    F = File(tbl)
    sec = F.indexes.get("image_path", {})
    assert sec.get("index") == "bovw", "Índice bovw(image_path) no fue registrado"
    base = Path(sec["filename"]).resolve()

    # Debe colgar de backend/runtime/files/<tabla>/<tabla>-bovw-image_path
    assert str(base).startswith(str(RUNTIME_DIR)), \
        f"El directorio del índice no está en runtime: {base}  (esperado prefijo {RUNTIME_DIR})"

    for fname in ("codebook.joblib", "idf.npy", "postings.json", "doc_map.json"):
        assert (base / fname).exists(), f"Falta artefacto {fname} en {base}"


@pytest.mark.parametrize("pk", PK_METHODS)
def test_bovw_runtime_knn_sql_or_fallback(engine: Engine, img_paths: dict, pk: str):
    tbl = f"bovw_sys_{pk}"

    # Intento 1: KNN nativo (forma soportada por tu parser)
    env = _run(engine, f"""
        SELECT id, title
        FROM {tbl}
        WHERE image_path KNN <-> IMG('{img_paths['chk1']}')
        LIMIT 3;
    """)
    if env.get("ok", False) and env["results"][0].get("ok", False):
        rows = env["results"][0].get("data") or []
        ids = {int(r["id"]) for r in rows}
        # Debe contener alguno de {1,2} (checker-like)
        assert ids & {1, 2}, f"kNN(SQL) esperaba alguno de {{1,2}}, got {sorted(ids)}"
        return

    # Intento 2 (fallback): usa los artefactos BoVW en runtime para hacer knn por código
    #   Nota: No usamos tmp en ningún momento.
    try:
        try:
            from backend.storage.indexes.bovw import BoVWFile
        except Exception:  # pragma: no cover
            from storage.indexes.bovw import BoVWFile  # type: ignore
        F = File(tbl)
        sec = F.indexes.get("image_path", {})
        base = sec["filename"]
        bovw = BoVWFile(base, key="image_path")
        ids = bovw.knn(img_paths["chk1"], 3)
        assert set(ids) & {1, 2}, f"kNN(fallback) esperaba alguno de {{1,2}}, got {sorted(ids)}"
    except Exception as e:
        pytest.fail(f"No hay KNN SQL y falló fallback BoVW: {e}")
