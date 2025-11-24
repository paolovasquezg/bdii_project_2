import sys
from pathlib import Path
import json

# Agregar raíz del proyecto
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def create_audios_table():
    """Crea la tabla Audios con su schema"""
    
    print("="*60)
    print("📋 CREANDO TABLA AUDIOS")
    print("="*60)
    
    # Definir schema de la tabla
    schema = {
        "table_name": "Audios",
        "columns": [
            {
                "name": "audio_id",
                "type": "INT",
                "primary_key": True
            },
            {
                "name": "file_name",
                "type": "VARCHAR",
                "size": 255
            },
            {
                "name": "file_path",
                "type": "VARCHAR",
                "size": 512
            },
            {
                "name": "tfidf_vector",
                "type": "ARRAY",
                "size": 500  # 500 dimensiones
            }
        ],
        "index_type": "sequential",
        "data_file": "backend/storage/data/audios.dat"
    }
    
    # Crear directorio si no existe
    data_dir = project_root / "backend" / "storage" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Guardar schema
    schema_file = data_dir / "audios_schema.json"
    with open(schema_file, 'w') as f:
        json.dump(schema, f, indent=4)
    
    print(f"\n✅ Schema creado: {schema_file}")
    print(f"\n📊 Estructura de la tabla:")
    print(f"   Tabla: {schema['table_name']}")
    print(f"   Columnas:")
    for col in schema['columns']:
        pk = " [PRIMARY KEY]" if col.get('primary_key') else ""
        size_info = f"({col.get('size', '')})" if col.get('size') else ""
        print(f"      - {col['name']}: {col['type']}{size_info}{pk}")
    
    print(f"\n💾 Archivo de datos: {schema['data_file']}")
    
    # Crear archivo .dat vacío
    data_file = project_root / schema['data_file']
    if not data_file.exists():
        # Crear archivo binario vacío para Sequential File
        with open(data_file, 'wb') as f:
            # Escribir header (cantidad de registros = 0)
            f.write((0).to_bytes(4, byteorder='little'))
        print(f"✅ Archivo de datos creado: {data_file}")
    else:
        print(f"⚠️  Archivo de datos ya existe: {data_file}")
    
    print("\n" + "="*60)
    print("✅ TABLA AUDIOS CREADA")
    print("="*60)
    print("\n🎯 Siguiente paso:")
    print("   python scripts/build_audio_database.py")


def main():
    try:
        create_audios_table()
    except Exception as e:
        print(f"\n❌ Error al crear tabla: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
