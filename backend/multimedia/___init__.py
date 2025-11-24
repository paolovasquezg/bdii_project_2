"""
Módulo Multimedia - Proyecto 2
Procesamiento de audio para búsqueda por similitud

Componentes:
- extraction: Extracción de características MFCC
- codebook: Construcción de vocabulario acústico (K-Means)
- indexing: Construcción de índice invertido
- utils: Utilidades comunes
"""

__version__ = '1.0.0'
__author__ = 'Proyecto 2 - BDII'

# Exports principales
from .extraction import AudioFeatureExtractor
from .codebook import AcousticCodebook

__all__ = [
    'AudioFeatureExtractor',
    'AcousticCodebook',
]