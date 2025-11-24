# -*- coding: utf-8 -*-
"""
E2E mixto (texto + imagen) importando un CSV “grande”:
- Genera 12 imágenes sintéticas y un CSV con rutas + textos.
- Crea tabla, importa con INSERT FROM FILE, crea índices BoVW/InvText.
- Verifica kNN de texto, kNN de imagen y SELECT de columnas específicas.
"""
import csv
import os
from pathlib import Path
from typing import Dict, Any, List

try:
    from backend.engine.engine import Engine
except Exception:  # pragma: no cover
    from engine import Engine  # type: ignore

ENGINE = Engine()
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "_testdata"
IMG_DIR = DATA_DIR / "imgs_csv"
CSV_PATH = DATA_DIR / "csv" / "media_big.csv"

CHECK = "✓"
CROSS = "✗"


def ensure_imgs(dirpath: Path) -> Dict[str, str]:
    from PIL import Image, ImageDraw
    d = Path(dirpath)
    d.mkdir(parents=True, exist_ok=True)

    def checker(size=256, tile=16, invert=False):
        im = Image.new("L", (size, size), 255)
        px = im.load()
        for y in range(size):
            for x in range(size):
                val = 0 if ((x // tile) + (y // tile)) % 2 == 0 else 255
                px[x, y] = 255 - val if invert else val
        return im

    def stripes(size=256, stripe=12, vertical=False):
        im = Image.new("L", (size, size), 255)
        dr = ImageDraw.Draw(im)
        if vertical:
            for x in range(0, size, stripe * 2):
                dr.rectangle([(x, 0), (x + stripe - 1, size)], fill=0)
        else:
            for y in range(0, size, stripe * 2):
                dr.rectangle([(0, y), (size, y + stripe - 1)], fill=0)
        return im

    def circles(size=256, r=80):
        im = Image.new("L", (size, size), 255)
        dr = ImageDraw.Draw(im)
        for rr in range(r, 20, -20):
            bbox = (size // 2 - rr, size // 2 - rr, size // 2 + rr, size // 2 + rr)
            dr.ellipse(bbox, outline=0, width=3)
        return im

    paths: Dict[str, str] = {}
    imgs = {
        "chk1": checker(),
        "chk2": checker(invert=True),
        "stripe_h": stripes(),
        "stripe_v": stripes(vertical=True),
        "circles": circles(),
        "circles_small": circles(r=60),
        "checker_fine": checker(tile=8),
        "checker_coarse": checker(tile=32),
        "stripe_thin": stripes(stripe=6),
        "stripe_wide": stripes(stripe=20),
        "checker_alt": checker(tile=10, invert=True),
        "circles_dense": circles(r=100),
    }
    for name, img in imgs.items():
        p = d / f"{name}.png"
        img.save(p)
        paths[name] = str(p)
    return paths


def ensure_csv(paths: Dict[str, str]) -> Path:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        (1, "Playa uno", paths["chk1"], "playa mar arena verano"),
        (2, "Playa dos", paths["chk2"], "playa sol arena surf"),
        (3, "Nieve", paths["stripe_h"], "nieve invierno montaña"),
        (4, "Techno", paths["circles"], "techno electronica rave"),
        (5, "Atardecer", paths["circles_small"], "atardecer costa playa"),
        (6, "Montaña", paths["checker_fine"], "montaña bosque frio"),
        (7, "Ciudad", paths["stripe_v"], "ciudad edificios noche"),
        (8, "Campo", paths["checker_coarse"], "campo verde pradera"),
        (9, "Surf", paths["stripe_thin"], "surf olas mar playa"),
        (10, "Bosque", paths["stripe_wide"], "bosque arboles naturaleza"),
        (11, "Festival", paths["checker_alt"], "festival musica luces techno"),
        (12, "Mar profundo", paths["circles_dense"], "mar profundo azul"),
    ]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "title", "image_path", "image_text"])
        w.writerows(rows)
    return CSV_PATH


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
        idx = (res0.get("meta") or {}).get("index_usage") or []
        for u in idx:
            if (u.get("index") or u.get("method")) == index_kind and u.get("field") == field:
                return True
    except Exception:
        pass
    return False


def run_for_pk(pk_method: str, csv_path: Path, imgs: Dict[str, str]) -> bool:
    tbl = f"media_csv_{pk_method}"
    print("\n" + "=" * 70)
    print(f"E2E • CSV Texto+Imagen • PK {pk_method}")
    print("=" * 70)

    run_sql(f"DROP TABLE IF EXISTS {tbl};")
    clean = str(csv_path).replace("\\", "/")
    assert_ok(run_sql(
        f"CREATE TABLE {tbl} FROM FILE '{clean}' USING INDEX {pk_method}(id);"
    ), "create table from file")

    assert_ok(run_sql(f"CREATE INDEX ON {tbl}(image_path) USING bovw;"), "create bovw")
    assert_ok(run_sql(f"CREATE INDEX ON {tbl}(image_text) USING invtext;"), "create invtext")

    ok_all = True

    # Texto
    env_txt = run_sql(f"SELECT id, title FROM {tbl} WHERE image_text KNN <-> 'playa' LIMIT 4;")
    assert_ok(env_txt, "knn text playa")
    ids_txt = {int(r["id"]) for r in (env_txt["results"][0].get("data") or [])}
    ok_txt = bool(ids_txt & {1, 2, 5, 9})
    print(("✓" if ok_txt else "✗"), f"Texto 'playa' -> {sorted(ids_txt)}")
    ok_txt &= used_index(env_txt, "invtext", "image_text")
    ok_all &= ok_txt

    # Imagen
    env_img = run_sql(f"SELECT * FROM {tbl} WHERE image_path KNN <-> IMG('{imgs['chk1']}') LIMIT 5;")
    assert_ok(env_img, "knn img chk1")
    ids_img = {int(r["id"]) for r in (env_img["results"][0].get("data") or [])}
    ok_img = bool(ids_img & {1, 2, 9})
    print(("✓" if ok_img else "✗"), f"Imagen chk1 -> {sorted(ids_img)}")
    ok_img &= used_index(env_img, "bovw", "image_path")
    ok_all &= ok_img

    # SELECT columnas específicas
    env_proj = run_sql(f"SELECT title, image_text FROM {tbl} WHERE id = 4 LIMIT 1;")
    res_proj = assert_ok(env_proj, "select projection")
    row = (res_proj.get("data") or [{}])[0]
    proj_ok = isinstance(row, dict) and set(row.keys()) == {"title", "image_text"}
    print(("✓" if proj_ok else "✗"), f"SELECT columnas específicas -> {row}")
    ok_all &= proj_ok

    return ok_all


def main():
    imgs = ensure_imgs(IMG_DIR)
    csv_path = ensure_csv(imgs)
    all_ok = True
    for pk in ["heap", "sequential", "isam", "bplus"]:
        try:
            if not run_for_pk(pk, csv_path, imgs):
                all_ok = False
        except AssertionError as e:
            print(f"{CROSS} {pk} fallo: {e}")
            all_ok = False
    print("\n==============================")
    print("RESUMEN:", "✓ OK" if all_ok else "✗ FALLÓ ALGUNO")
    print("==============================")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
