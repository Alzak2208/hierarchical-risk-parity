"""Hierarchical Risk Parity and its benchmarks (CLA minimum variance, inverse variance).

Public entry points:

- :func:`src.allocation.hrp_weights`: Hierarchical Risk Parity
- :func:`src.allocation.inverse_variance_weights`: traditional risk parity benchmark
- :func:`src.cla.min_variance_weights`: minimum variance via the Critical Line Algorithm
- :class:`src.cpcv.CombinatorialPurgedCV` and :func:`src.backtest.run_cpcv_backtest`
"""

from src.allocation import equal_weights, hrp_weights, inverse_variance_weights
from src.backtest import run_cpcv_backtest
from src.cla import critical_line, min_variance_weights
from src.cpcv import CombinatorialPurgedCV

__all__ = [
    "CombinatorialPurgedCV",
    "critical_line",
    "equal_weights",
    "hrp_weights",
    "inverse_variance_weights",
    "min_variance_weights",
    "run_cpcv_backtest",
]
