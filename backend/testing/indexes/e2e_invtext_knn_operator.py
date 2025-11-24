# -*- coding: utf-8 -*-
"""
E2E • InvText con operadores <-> (indexado) y <--> (no indexado)

- Crea tabla, inserta textos y crea índice USING invtext.
- Ejecuta dos consultas KNN:
    * content KNN <-> 'palabra'   -> usa índice
    * content KNN <--> 'palabra'  -> ruta sin índice (flag use_indexed=False)
- Verifica que ambas devuelvan resultados y que el plan refleje use_indexed.
"""
from backend.engine.engine import Engine


def _first(env: dict) -> dict:
    return (env or {}).get("results", [{}])[0]


def _assert_ok(env: dict, label: str):
    assert env.get("ok", False), f"{label}: envelope not ok"
    r0 = _first(env)
    assert r0.get("ok", False), f"{label}: result not ok -> {r0.get('error')}"
    return r0


def test_invtext_knn_indexed_vs_seq():
    eng = Engine()
    eng.run("DROP TABLE IF EXISTS text_knn_ops;")
    eng.run(
        """
        CREATE TABLE text_knn_ops(
            id INT PRIMARY KEY USING heap,
            content VARCHAR(256)
        );
        """
    )
    docs = [
        (1, "lorem ipsum dolor sit amet"),
        (2, "ipsum dolor amet amet ipsum"),
        (3, "quick brown fox jumps"),
        (4, "lorem lorem quick"),
    ]
    for i, txt in docs:
        eng.run(f"INSERT INTO text_knn_ops(id,content) VALUES ({i}, '{txt}');")

    _assert_ok(eng.run("CREATE INDEX ON text_knn_ops(content) USING invtext;"), "create index")

    env_idx = eng.run("SELECT id,content FROM text_knn_ops WHERE content KNN <-> 'lorem' LIMIT 2;")
    r_idx = _assert_ok(env_idx, "knn indexed")
    data_idx = r_idx.get("data") or []
    plan_idx = r_idx.get("plan") or {}
    assert len(data_idx) == 2, f"indexed debería devolver 2, got {len(data_idx)}"
    assert plan_idx.get("use_indexed") is True

    env_seq = eng.run("SELECT id,content FROM text_knn_ops WHERE content KNN <--> 'lorem' LIMIT 2;")
    r_seq = _assert_ok(env_seq, "knn no indexado")
    data_seq = r_seq.get("data") or []
    plan_seq = r_seq.get("plan") or {}
    assert len(data_seq) == 2, f"no indexado debería devolver 2, got {len(data_seq)}"
    assert plan_seq.get("use_indexed") is False

    # primera coincidencia debería ser un doc con 'lorem'
    assert data_idx[0]["id"] in (1, 4)
    assert data_seq[0]["id"] in (1, 4)

