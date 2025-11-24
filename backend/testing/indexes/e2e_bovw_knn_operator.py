# -*- coding: utf-8 -*-
"""
E2E • BoVW con operadores <-> (indexado) y <--> (secuencial)

- Genera 4 imágenes sintéticas.
- Crea tabla, inserta filas y crea índice USING bovw.
- Ejecuta dos consultas KNN:
    * image_path KNN <-> IMG('...')    -> debe usar índice
    * image_path KNN <--> IMG('...')   -> debe ir por ruta no indexada
- Valida que ambas devuelvan resultados y que el plan refleje use_indexed.
"""
import json
from pathlib import Path

import pytest

from backend.engine.engine import Engine
from backend.testing.indexes.e2e_bovw_index import ensure_imgs


def _run(sql: str):
    eng = Engine()
    return eng.run(sql)


def _first_result(env: dict) -> dict:
    return (env or {}).get("results", [{}])[0]


def _assert_ok(env: dict, label: str):
    assert env.get("ok", False), f"{label}: envelope not ok"
    res0 = _first_result(env)
    assert res0.get("ok", False), f"{label}: result not ok -> {res0.get('error')}"
    return res0


def test_bovw_knn_indexed_vs_sequential(tmp_path: Path):
    # 1) imágenes sintéticas
    img_dir = tmp_path / "imgs"
    paths = ensure_imgs(img_dir)
    imgs = list(paths.values())

    eng = Engine()
    eng.run("DROP TABLE IF EXISTS img_knn_ops;")
    eng.run(
        """
        CREATE TABLE img_knn_ops(
            id INT PRIMARY KEY USING heap,
            title VARCHAR(64),
            image_path VARCHAR(512)
        );
        """
    )
    for i, p in enumerate(imgs, start=1):
        eng.run(f"INSERT INTO img_knn_ops(id,title,image_path) VALUES ({i}, 'img{i}', '{p}');")

    # Índice sobre file_path (unificamos criterio con audio)
    _assert_ok(eng.run("CREATE INDEX ON img_knn_ops(image_path) USING bovw;"), "create index")

    # 2) KNN indexado
    env_idx = eng.run(
        f"SELECT id,title FROM img_knn_ops WHERE image_path KNN <-> IMG('{imgs[0]}') LIMIT 3;"
    )
    res_idx = _assert_ok(env_idx, "knn indexed")
    data_idx = res_idx.get("data") or []
    plan_idx = res_idx.get("plan") or {}
    assert len(data_idx) == 3, f"knn indexed debe devolver 3, got {len(data_idx)}"
    assert plan_idx.get("use_indexed") is True

    # 3) KNN no indexado (<-->)
    env_seq = eng.run(
        f"SELECT id,title FROM img_knn_ops WHERE image_path KNN <--> IMG('{imgs[0]}') LIMIT 3;"
    )
    res_seq = _assert_ok(env_seq, "knn no indexado")
    data_seq = res_seq.get("data") or []
    plan_seq = res_seq.get("plan") or {}
    assert len(data_seq) == 3, f"knn no indexado debe devolver 3, got {len(data_seq)}"
    assert plan_seq.get("use_indexed") is False

    # Opcional: primera coincidencia debe ser la misma
    assert data_idx[0]["id"] == data_seq[0]["id"] == 1

    # Log rápido por si se ejecuta standalone
    print(json.dumps({"indexed": data_idx, "non_indexed": data_seq}, indent=2))
