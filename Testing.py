from TableCreate import create_table
from File import File
import shutil


def HeapTest():

    shutil.rmtree("files", ignore_errors=True)

    table = "heap_products"
    fields = [
        {"name": "product_id", "type": "i", "key": "primary"},
        {"name": "name", "type": "s", "length": 32},
        {"name": "price", "type": "f"},
        {"name": "stock", "type": "i"},
        {"name": "description", "type": "s", "length": 10000000}
    ]

    create_table(table, fields)
    file = File(table)

    products = [
        {"product_id": 1, "name": "Widget", "price": 9.99, "stock": 100, "description": "El Widget es un componente modular diseñado para integrarse fácilmente con sistemas existentes. Ofrece un rendimiento estable incluso bajo cargas intensivas y está fabricado con materiales de alta resistencia. Ideal para aplicaciones industriales, educativas y prototipado rápido. Su tamaño compacto permite instalarlo en espacios reducidos sin perder funcionalidad."},
        {"product_id": 2, "name": "Gadget", "price": 12.50, "stock": 50, "description": "Este Gadget multifuncional combina portabilidad con eficiencia. Incluye sensores de última generación que permiten un monitoreo preciso del entorno. Su interfaz intuitiva hace que cualquier usuario pueda configurarlo sin experiencia previa. Es perfecto para tareas de automatización, domótica y proyectos de ingeniería ligera."},
        {"product_id": 3, "name": "Tool", "price": 15.00, "stock": 30, "description": "La Tool es una herramienta versátil pensada para profesionales que requieren precisión. Está optimizada para soportar uso prolongado y ofrece compatibilidad con accesorios adicionales. Su diseño ergonómico reduce la fatiga durante largas sesiones de trabajo. Ideal para mantenimiento técnico, ensamblaje y reparación en campo."},
        {"product_id": 4, "name": "Device", "price": 8.75, "stock": 75, "description": "El Device es un equipo compacto con sistema inteligente integrado. Permite la sincronización con múltiples plataformas mediante protocolos modernos de comunicación. Gracias a su bajo consumo energético y su estructura robusta, es una opción confiable para operaciones continuas. Recomendado para laboratorios, talleres y entornos de prueba."}
    ]
    for prod in products:
        params = {
            "op": "insert",
            "record": prod
        }
        file.execute(params)

    file.execute({"op": "create_inverted_text", "column": "description"})
    print(file.execute({"op": "text_search", "consulta":"busco un dispositivo compacto con sensores integrados que permita automatizar tareas y funcione bien en entornos industriales", "column": "description", "k": 4}))
    file.execute({"op": "drop_inverted_text", "column": "description"})

if __name__ == "__main__":
    HeapTest()

[{"product_id": 1, "name": "Paolo", "price": 1, "id": 0}, {"product_id": 2}]

