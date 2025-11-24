import os
import uuid
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File
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
    # Validación básica de tipo
    ct = (file.content_type or "").lower()
    if not (ct.startswith("image/") or ct.startswith("audio/") or ct in ("application/octet-stream",)):
        raise HTTPException(status_code=400, detail="Solo se permiten imágenes o audios.")

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
