# CONFIGURACIÓN DEMO RÁPIDA
DEMO_CONFIG = {
    "n_clusters": 400,        # Muchos más clusters para máxima granularidad
    "max_samples": 100000,    # Subset de datos (vs 1.3M)
    "max_iter": 150,          # Más iteraciones para convergencia perfecta
    "tol": 1e-6,              # Tolerancia muy estricta
    "demo_mode": True
}

# CONFIGURACIÓN PRODUCCIÓN (ORIGINAL)
PROD_CONFIG = {
    "n_clusters": 500,        # Clusters originales
    "max_samples": None,      # Todos los datos
    "max_iter": 100,          # Iteraciones para mejor convergencia
    "tol": 1e-4,              # Tolerancia original
    "demo_mode": False
}
