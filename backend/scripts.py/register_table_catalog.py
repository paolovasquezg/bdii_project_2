import sys
from pathlib import Path

# Agregar raíz del proyecto
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.catalog.catalog import load_tables, save_tables, table_dir, put_json


def register_audios_table():
    # Nombre de la tabla
    table_name = "audios"
    
    # Schema de la tabla (formato del P1 - ahora con soporte para arrays)
    schema = {
        "audio_id": {"type": "int"},
        "file_name": {"type": "varchar", "length": 50}, 
        "file_path": {"type": "varchar", "length": 200},
        "tfidf_vector": {"type": "array", "length": 400},  # Ajustado a K=400
        "deleted": {"type": "bool"}  # Campo requerido por Sequential File
    }
    
    # Registrar en el catálogo del P1
    print(f"\n📝 Registrando tabla '{table_name}' en catálogo...")
    print(f"   Columnas: {list(schema.keys())}")
    
    try:
        # Cargar catálogo existente
        catalog = load_tables()
        
        # Agregar nuestra tabla al catálogo
        catalog[table_name] = {
            "schema": schema,
            "type": "sequential"  # Indica que usa Sequential File
        }
        
        # Guardar catálogo actualizado
        save_tables(catalog)
        
        print(f"\n Tabla '{table_name}' registrada exitosamente en el catálogo")
        
        # Crear directorio de la tabla
        table_directory = table_dir(table_name)
        table_directory.mkdir(parents=True, exist_ok=True)
        
        # Crear archivo .dat vacío para la tabla
        data_file = table_directory / f"{table_name}.dat"
        if not data_file.exists():
            # Escribir esquema inicial usando put_json
            put_json(str(data_file), schema)
            print(f"Archivo creado con esquema: {data_file}")
        else:
            print(f"Archivo ya existe: {data_file}")
        
        print("TABLA AUDIOS REGISTRADA")
        
    except Exception as e:
        print(f"\n Error al registrar tabla: {e}")
        print("\n Verifica que:")
        print("   1. El catálogo del P1 funcione correctamente")
        print("   2. Tengas permisos de escritura")
        import traceback
        traceback.print_exc()


def main():
    try:
        register_audios_table()
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
