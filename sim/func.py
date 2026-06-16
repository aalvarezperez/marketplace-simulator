"""Small math helpers for the SimPy engine.

Kept inside ``sim`` so the package is self-contained and installable on its own
(the legacy ``classes.py`` engine has its own root-level ``func.py``).
"""
import numpy as np


def sigmoid(x):
    """Logistic function 1 / (1 + e^-x), mapping any real to (0, 1).

    Used to turn a log-odds expression into a probability: every implicit-fidelity
    funnel rate is ``sigmoid(log(base) + slope * log(driver))``, so ``base`` is the
    rate at ``driver == 1`` and ``slope`` is its elasticity to the driver.
    """
    return 1 / (1 + np.exp(-x))
