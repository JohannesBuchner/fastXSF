"""Fast X-ray spectral fitting."""

__author__ = """Johannes Buchner"""
__email__ = 'johannes.buchner.acad@gmx.com'
__version__ = '1.2.0'

from .data import load_pha
from .model import Table, logPoissonPDF, logPoissonPDF_vectorized, x, xvec
