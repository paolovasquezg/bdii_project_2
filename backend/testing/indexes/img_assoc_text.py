# backend/testing/indexes/test_invtext_bovw_e2e.py
import json, sys, os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

# --- Engine de tu proyecto ---
from backend.engine.engine import Engine

# ============ CONFIG ============

ENGINE = Engine()
IMG_DIR = (Path.cwd() / "img_demo").resolve()  # carpeta donde generamos las imágenes

# ============ HELPERS ============

def run_sql(sql: str) -> Dict[str, Any]:
    env = ENGINE.run(sql)
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
    if not isinstance(env, dict):
        return []
    for key in ("rows", "data"):
        if isinstance(env.get(key), list):
            return env[key]  # type: ignore
    out = []
    results = env.get("results")
    if isinstance(results, list):
        for r in results:
            if isinstance(r, dict) and r.get("action") in ("knn","select"):
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
                got = walk(v);
                if got is not None: return got
        elif isinstance(x, (list, tuple)):
            for v in x:
                got = walk(v);
                if got is not None: return got
        return None
    any_data = walk(env)
    return any_data if isinstance(any_data, list) else []

def extract_ids(rows: List[Any]) -> List[int]:
    ids = []
    for r in rows:
        if isinstance(r, dict) and "id" in r:
            try: ids.append(int(r["id"]))
            except: pass
        elif isinstance(r, (list, tuple)) and r:
            try: ids.append(int(r[0]))
            except: pass
    return ids

def expect_any(actual_ids: List[int], expected_any: List[int], label: str) -> bool:
    ok = bool(set(actual_ids) & set(expected_any))
    print(("✓" if ok else "✗"), f"{label}: got {actual_ids}, expected any of {expected_any}")
    return ok

def used_index(env: Dict[str, Any], index: str, field: str) -> bool:
    for r in env.get("results", []):
        for u in r.get("meta", {}).get("index_usage", []):
            if u.get("index") == index and u.get("field") == field:
                return True
    return False

# ============ GENERAR IMÁGENES ============

def generate_demo_images() -> Dict[str, str]:
    from PIL import Image, ImageDraw, ImageFont
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    W, H = 768, 512
    paths = {}

    def label(draw, text):
        bar_h = 46
        draw.rectangle([(0, H - bar_h), (W, H)], fill=(0, 0, 0, 160))
        try:
            # Pillow ≥ 8/10: mide con textbbox
            bbox = draw.textbbox((0, 0), text)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            x = (W - tw) // 2
            # Ajuste por posibles offsets del bbox (top puede ser negativo)
            y = H - bar_h + (bar_h - th) // 2 - bbox[1]
        except AttributeError:
            # Pillow < 10: fallback
            tw, th = draw.textsize(text)
            x = (W - tw) // 2
            y = H - bar_h + (bar_h - th) // 2

        draw.text((x, y), text, fill=(255, 255, 255, 230))

    def save(img, name):
        p = str((IMG_DIR / name).resolve())
        img.convert("RGB").save(p, quality=92)
        paths[name] = p

    # playa1.jpg
    img = Image.new("RGBA", (W,H), (135,206,235,255))
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle([(0, H*0.45), (W, H*0.75)], fill=(30,144,255,255))
    d.rectangle([(0, H*0.75), (W, H)], fill=(237,201,175,255))
    d.ellipse([ (W*0.78, H*0.10), (W*0.92, H*0.24) ], fill=(255,223,0,255))
    label(d, "playa1"); save(img, "playa1.jpg")

    # nieve1.jpg
    img = Image.new("RGBA", (W,H), (176,196,222,255))
    d = ImageDraw.Draw(img, "RGBA")
    d.polygon([(W*0.15,H*0.85),(W*0.35,H*0.35),(W*0.55,H*0.85)], fill=(105,105,105,255))
    d.polygon([(W*0.45,H*0.85),(W*0.70,H*0.30),(W*0.90,H*0.85)], fill=(119,136,153,255))
    d.polygon([(W*0.33,H*0.45),(W*0.35,H*0.35),(W*0.40,H*0.48)], fill=(255,255,255,255))
    d.polygon([(W*0.68,H*0.40),(W*0.70,H*0.30),(W*0.75,H*0.45)], fill=(255,255,255,255))
    d.rectangle([(0,H*0.82),(W,H)], fill=(250,250,250,255))
    label(d, "nieve1"); save(img, "nieve1.jpg")

    # playa2.jpg
    img = Image.new("RGBA", (W,H), (70,130,180,255))
    d = ImageDraw.Draw(img, "RGBA")
    for y in range(int(H*0.3), H):
        t = (y - H*0.3) / (H*0.7)
        c = (int(20+60*t), int(110+80*t), int(160+80*t), 255)
        d.line([(0,y),(W,y)], fill=c, width=1)
    for r in range(0,180,6):
        d.arc([W*0.15 - r, H*0.55 - r, W*0.85 + r, H*1.15 + r], start=200, end=260, fill=(224,255,255,200), width=6)
    label(d, "playa2 (surf)"); save(img, "playa2.jpg")

    # techno1.jpg
    img = Image.new("RGBA", (W,H), (10,10,20,255))
    d = ImageDraw.Draw(img, "RGBA")
    for y in range(int(H*0.45), int(H*0.75)):
        a = int(180 - 3*(y - H*0.45))
        d.line([(0,y),(W,y)], fill=(255,0,128,a))
    for x in range(0, W, 32):
        d.line([(x,H*0.55),(x,H)], fill=(0,255,255,100))
    for y in range(int(H*0.55), H, 28):
        d.line([(0,y),(W,y)], fill=(0,255,255,100))
    d.ellipse([(W*0.35,H*0.10),(W*0.65,H*0.40)], outline=(255,0,128,220), width=4)
    label(d, "techno1"); save(img, "techno1.jpg")

    # playa3.jpg
    img = Image.new("RGBA", (W,H), (0,0,0,0))
    d = ImageDraw.Draw(img, "RGBA")
    for y in range(H):
        t = y / H
        r = int(255*(1-t) + 20*t)
        g = int(120*(1-t) + 30*t)
        b = int(50*(1-t) + 80*t)
        d.line([(0,y),(W,y)], fill=(r,g,b,255))
    for y in range(int(H*0.62), H):
        d.line([(0,y),(W,y)], fill=(20,70,120,255))
    d.pieslice([(W*0.35,H*0.42),(W*0.65,H*0.72)], start=0, end=180, fill=(255,170,0,230))
    label(d, "playa3 (sunset)"); save(img, "playa3.jpg")

    return paths  # dict name->abs_path

# ============ FIXTURE + TESTS ============

def setup_table(pk_method: str, tname: str, p: Dict[str,str]):
    run_sql(f"DROP TABLE IF EXISTS {tname};")
    run_sql(f"""
    CREATE TABLE {tname} (
      id          INT PRIMARY KEY USING {pk_method},
      title       VARCHAR(120),
      description VARCHAR(300),
      image_path  VARCHAR(300),
      image_text  VARCHAR(400)
    );
    """)
    run_sql(f"""
    INSERT INTO {tname} (id, title, description, image_path, image_text) VALUES
      (1, 'Foto playa',      'Arena y mar cálido',        '{p["playa1.jpg"]}',  'playa mar arena verano'),
      (2, 'Foto nieve',      'Nieve en la montaña',       '{p["nieve1.jpg"]}',  'nieve invierno montaña'),
      (3, 'Surf en playa',   'Surf por la mañana',        '{p["playa2.jpg"]}',  'playa sol arena surf'),
      (4, 'Techno night',    'Rave electrónica nocturna', '{p["techno1.jpg"]}', 'techno electronica rave'),
      (5, 'Atardecer playa', 'Atardecer en la costa',     '{p["playa3.jpg"]}',  'playa atardecer costa');
    """)
    run_sql(f"CREATE INDEX IF NOT EXISTS ix_bovw_{tname}_image_path  ON {tname}(image_path)  USING bovw;")
    run_sql(f"CREATE INDEX IF NOT EXISTS ix_invtext_{tname}_text     ON {tname}(image_text) USING invtext;")

def run_checks_for(pk_method: str, p: Dict[str,str]) -> bool:
    t = f"multimedia_{pk_method}"
    print("\n==============================")
    print(f"E2E • {t}")
    print("==============================")
    setup_table(pk_method, t, p)

    ok = True

    # --- KNN texto ---
    env1 = run_sql(f"SELECT id, title FROM {t} WHERE image_text KNN <-> 'playa'  LIMIT 3;")
    env2 = run_sql(f"SELECT id, title FROM {t} WHERE image_text KNN <-> 'nieve'  LIMIT 1;")
    env3 = run_sql(f"SELECT id, title FROM {t} WHERE image_text KNN <-> 'techno' LIMIT 1;")

    ids1 = extract_ids(rows_from_env(env1))
    ids2 = extract_ids(rows_from_env(env2))
    ids3 = extract_ids(rows_from_env(env3))

    ok &= expect_any(ids1, [1,3,5], f"{pk_method} :: text KNN 'playa'")
    ok &= expect_any(ids2, [2],     f"{pk_method} :: text KNN 'nieve'")
    ok &= expect_any(ids3, [4],     f"{pk_method} :: text KNN 'techno'")
    ok &= (used_index(env1, "invtext", "image_text") and used_index(env2, "invtext", "image_text") and used_index(env3, "invtext", "image_text"))
    print(("✓" if used_index(env1, "invtext", "image_text") else "✗"), f"{pk_method} :: usa índice invtext")

    # --- Imagen → Imagen (BoVW) ---
    anchor = p["playa1.jpg"]
    env_img = run_sql(f"""
    SELECT id, title, description, image_path
    FROM {t}
    WHERE image_path KNN <-> IMG('{anchor}')
    LIMIT 4;
    """)
    rows_img = rows_from_env(env_img)
    print(("✓" if rows_img else "✗"), f"{pk_method} :: image→image KNN -> {[r.get('id') if isinstance(r,dict) else r for r in rows_img]}")
    ok &= bool(rows_img) and used_index(env_img, "bovw", "image_path")
    print(("✓" if used_index(env_img, "bovw", "image_path") else "✗"), f"{pk_method} :: usa índice bovw")

    # --- Imagen → Texto (asociación textual) ---
    # usa el image_text de la ancla id=1 para buscar vecinos de texto
    env_txt = run_sql(f"SELECT image_text FROM {t} WHERE id = 1 LIMIT 1;")
    rows_txt = rows_from_env(env_txt)
    q_text = (rows_txt[0]["image_text"] if rows_txt and isinstance(rows_txt[0], dict) else rows_txt[0]) or "playa mar arena verano"
    env_assoc = run_sql(f"""
    SELECT id, title, description, image_path
    FROM {t}
    WHERE image_text KNN <-> '{q_text}'
    LIMIT 6;
    """)
    assoc_rows = rows_from_env(env_assoc)
    print(("✓" if assoc_rows else "✗"), f"{pk_method} :: image→text relacionados -> {[r.get('id') if isinstance(r,dict) else r for r in assoc_rows]}")
    ok &= bool(assoc_rows)

    return ok

def main():
    # 1) genera imágenes en disco y obtiene rutas absolutas
    paths = generate_demo_images()
    print(f"\nImágenes generadas en: {IMG_DIR}")
    for k, v in paths.items():
        print(" -", k, "->", v)

    # 2) corre batería para 4 backends
    backends = ["heap", "sequential", "isam", "bplus"]
    all_ok = True
    for pk in backends:
        try:
            ok = run_checks_for(pk, paths)
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
