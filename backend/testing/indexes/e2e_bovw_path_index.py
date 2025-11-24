# -*- coding: utf-8 -*-
"""
E2E • Verifica que el índice BoVW se cree sobre image_path y responda KNN.
"""
from pathlib import Path
from backend.engine.engine import Engine
from backend.testing.indexes.e2e_bovw_index import ensure_imgs


def test_bovw_on_image_path(tmp_path: Path):
    img_dir = tmp_path / "imgs"
    paths = ensure_imgs(img_dir)
    imgs = list(paths.values())
    eng = Engine()
    eng.run("DROP TABLE IF EXISTS img_knn_path;")
    eng.run(
        """
        CREATE TABLE img_knn_path(
            id INT PRIMARY KEY USING heap,
            title VARCHAR(64),
            image_path VARCHAR(512)
        );
        """
    )
    for i, p in enumerate(imgs, start=1):
        eng.run(f"INSERT INTO img_knn_path(id,title,image_path) VALUES ({i}, 'img{i}', '{p}');")
    res = eng.run("CREATE INDEX ON img_knn_path(image_path) USING bovw;")
    assert res.get("ok", False)
    q = eng.run(f"SELECT id FROM img_knn_path WHERE image_path KNN <-> IMG('{imgs[0]}') LIMIT 3;")
    assert q.get("ok", False)
    r0 = q["results"][0]
    assert r0["ok"] and len(r0.get("data") or []) == 3
    assert (r0.get("plan") or {}).get("field") == "image_path"
