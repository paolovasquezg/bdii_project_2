import os
import uuid
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
from backend.catalog.ddl import load_tables
from backend.engine.engine import Engine
from fastapi.middleware.cors import CORSMiddleware

from backend.catalog.catalog import load_tables, get_json, table_meta_path

app = FastAPI(title="DB2 Project")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)
engine = Engine()

class Query(BaseModel):
    content: str

UPLOAD_DIR = Path(__file__).resolve().parent / "runtime" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CSV_UPLOAD_DIR = UPLOAD_DIR / "csv"
CSV_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@app.get("/")
def root():
    return "Testing: Databases 2 Project"

@app.get("/tables")
def get_tables():
    tables_names = load_tables()
    tables = {}

    for table in tables_names:

        meta = table_meta_path(table)
        relation, indexes = get_json(str(meta),2)
        tables[table] = {"relation": relation, "indexes": indexes}

    return tables

#aca debe usarse el parser
@app.post("/query")
def do_query(query: Query):
    return engine.run(query.content)

@app.post("/create-table-from-csv")
async def create_table_from_csv(table: str = Form(...), file: UploadFile = File(...)):
    """
    Sube un CSV y crea la tabla usando CREATE TABLE <table> FROM FILE '<path>'.
    Guarda el CSV en runtime/uploads/csv.
    """
    if not table or not table.strip():
        raise HTTPException(status_code=400, detail="Nombre de tabla requerido")

    dest = CSV_UPLOAD_DIR / f"{uuid.uuid4().hex}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Guarda tal cual el archivo sin tocar encoding
    try:
        raw = await file.read()
        dest.write_bytes(raw)
    finally:
        await file.close()

    sql = f"CREATE TABLE {table} FROM FILE '{dest}';"
    try:
        result = engine.run(sql)
        count = None
        try:
            first = (result.get("results") or [None])[0] or {}
            meta = first.get("meta") or {}
            count = meta.get("affected") or meta.get("count")
        except Exception:
            count = None
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error creando tabla: {e}")

    return {"ok": True, "table": table, "path": str(dest), "result": result, "count": count}

@app.get("/file")
def get_file(path: str):
    """
    Sirve un archivo local (imagen/audio) en modo lectura.
    Nota: asegúrate de pasar rutas válidas dentro del contenedor/host.
    """
    p = Path(path).resolve()
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="Archivo no encontrado")
    return FileResponse(p)


@app.post("/upload-media")
async def upload_media(file: UploadFile = File(...)):
    """
    Recibe imagen/audio y lo guarda en backend/runtime/uploads/<uuid>.<ext>.
    Devuelve la ruta absoluta para usarla en queries (ej. INSERT o KNN IMG()).
    """
    # Validación básica de tipo (permitimos imagen, audio y csv para reutilizar el mismo endpoint)
    ct = (file.content_type or "").lower()
    if not (
        ct.startswith("image/")
        or ct.startswith("audio/")
        or ct in ("application/octet-stream", "text/csv", "application/csv")
    ):
        raise HTTPException(status_code=400, detail="Solo se permiten imágenes, audios o CSV.")

    # Derivar extensión
    filename = file.filename or ""
    ext = ""
    if "." in filename:
        ext = filename.rsplit(".", 1)[1].lower()
    if not ext:
        # fallback por mime
        if ct.startswith("image/"):
            ext = ct.split("/", 1)[1]
        elif ct.startswith("audio/"):
            ext = ct.split("/", 1)[1]
        elif ct in ("text/csv", "application/csv"):
            ext = "csv"
        else:
            ext = "bin"

    dest_name = f"{uuid.uuid4().hex}.{ext}"
    dest_path = UPLOAD_DIR / dest_name
    # Por si el volumen fue limpiado en runtime
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with dest_path.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
    finally:
        await file.close()

    # guardar path en env para fallback
    os.environ["BD2_LAST_UPLOAD_PATH"] = str(dest_path)

    return {"ok": True, "path": str(dest_path)}
