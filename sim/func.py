"""Small math helpers for the SimPy engine.

Kept inside ``sim`` so the package is self-contained and installable on its own
(the legacy ``classes.py`` engine has its own root-level ``func.py``).
"""
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
