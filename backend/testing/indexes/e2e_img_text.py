# -*- coding: utf-8 -*-
"""
E2E mixto: texto (InvText) + imágenes (BoVW).
- Crea tabla con PK configurable y columnas image_path / image_text.
- Construye ambos índices.
- Corre kNN de texto y de imagen verificando IDs y uso de índice.
"""
import os
from pathlib import Path
from typing import Dict, Any

try:
    from backend.engine.engine import Engine
except Exception:  # pragma: no cover
    from engine import Engine  # type: ignore

PK_METHODS = ["heap", "sequential", "isam", "bplus"]
HERE = Path(__file__).resolve().parent
IMG_DIR = HERE / "_testdata" / "imgs"

ENGINE = Engine()

CHECK = "✓"
CROSS = "✗"


def ensure_imgs(dirpath: Path) -> Dict[str, str]:
    """Genera 4 imágenes sintéticas reutilizando lógica de otros tests."""
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


def run_sql(sql: str) -> Dict[str, Any]:
    return ENGINE.run(sql)


def assert_ok(env: dict, msg="") -> dict:
    if not env.get("ok", False):
        raise AssertionError(f"Envelope NOT ok: {msg}")
    res0 = (env.get("results") or [{}])[0]
    if not res0.get("ok", False):
        raise AssertionError(f"Result NOT ok: {msg} • {res0.get('error')}")
    return res0


def used_index(env: dict, index_kind: str, field: str) -> bool:
    try:
        res0 = (env.get("results") or [{}])[0]
        meta = res0.get("meta") or {}
        idx = meta.get("index_usage") or []
        for u in idx:
            if (u.get("index") or u.get("method")) == index_kind and u.get("field") == field:
                return True
    except Exception:
        pass
    return False


def run_for_pk(pk_method: str, imgs: Dict[str, str]) -> bool:
    tbl = f"media_mix_{pk_method}"
    print("\n" + "=" * 70)
    print(f"E2E • Texto+Imagen • PK {pk_method}")
    print("=" * 70)

    run_sql(f"DROP TABLE IF EXISTS {tbl};")
    assert_ok(run_sql(f"""
        CREATE TABLE {tbl}(
            id INT PRIMARY KEY USING {pk_method},
            title VARCHAR(64),
            image_path VARCHAR(512),
            image_text VARCHAR(256)
        );
    """), "create table")

    rows_sql = (
        f"INSERT INTO {tbl}(id,title,image_path,image_text) VALUES "
        f"(1,'chk1','{imgs['chk1']}','playa mar arena verano'),"
        f"(2,'chk2','{imgs['chk2']}','nieve invierno montaña'),"
        f"(3,'stripes','{imgs['stripe']}','playa sol arena surf'),"
        f"(4,'circles','{imgs['circle']}','techno electronica rave');"
    )
    assert_ok(run_sql(rows_sql), "insert rows")

    assert_ok(run_sql(f"CREATE INDEX ON {tbl}(image_path)  USING bovw;"), "create bovw")
    assert_ok(run_sql(f"CREATE INDEX ON {tbl}(image_text) USING invtext;"), "create invtext")

    # Texto -> texto
    env_txt = run_sql(f"""
        SELECT id, title FROM {tbl}
        WHERE image_text KNN <-> 'playa'
        LIMIT 3;
    """)
    assert_ok(env_txt, "knn text playa")
    ids_txt = {int(r["id"]) for r in (env_txt["results"][0].get("data") or [])}
    ok_txt = bool(ids_txt & {1, 3})
    if not ok_txt:
        print(f"{CROSS} Texto 'playa' -> {sorted(ids_txt)}")
    else:
        print(f"{CHECK} Texto 'playa' -> {sorted(ids_txt)}")
    ok_txt &= used_index(env_txt, "invtext", "image_text")
    print(("✓" if used_index(env_txt, "invtext", "image_text") else "✗"), "uso índice invtext")

    # Imagen -> imagen
    env_img = run_sql(f"""
        SELECT id, title FROM {tbl}
        WHERE image_path KNN <-> IMG('{imgs['chk1']}')
        LIMIT 3;
    """)
    assert_ok(env_img, "knn image chk1")
    ids_img = {int(r["id"]) for r in (env_img["results"][0].get("data") or [])}
    ok_img = bool(ids_img & {1, 2, 3})
    if not ok_img:
        print(f"{CROSS} Imagen chk1 -> {sorted(ids_img)}")
    else:
        print(f"{CHECK} Imagen chk1 -> {sorted(ids_img)}")
    ok_img &= used_index(env_img, "bovw", "image_path")
    print(("✓" if used_index(env_img, "bovw", "image_path") else "✗"), "uso índice bovw")

    return ok_txt and ok_img


def main():
    imgs = ensure_imgs(IMG_DIR)
    all_ok = True
    for pk in PK_METHODS:
        try:
            if not run_for_pk(pk, imgs):
                all_ok = False
        except AssertionError as e:
            print(f"{CROSS} {pk} fallo: {e}")
            all_ok = False
        except Exception as e:  # pragma: no cover
            print(f"{CROSS} {pk} excepción: {e}")
            all_ok = False

    print("\n==============================")
    print("RESUMEN:", "✓ OK" if all_ok else "✗ FALLÓ ALGUNO")
    print("==============================")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
