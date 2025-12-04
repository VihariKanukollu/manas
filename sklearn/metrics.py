"""
Minimal stub of `sklearn.metrics` for use by transformers' candidate generator.

We intentionally do NOT implement real metric computations; the EN-to-DSL
training code in this project never calls these utilities. The stub simply
allows imports like:

    from sklearn.metrics import roc_curve

to succeed without pulling in the compiled scikit-learn package that is
binary-incompatible with the current NumPy version on this system.
"""

from typing import Any, Iterable, Tuple


def roc_curve(
    y_true: Iterable[Any],
    y_score: Iterable[float],
    pos_label: Any = None,
    sample_weight: Iterable[float] | None = None,
    drop_intermediate: bool = True,
) -> Tuple[list[float], list[float], list[float]]:
    """
    Placeholder implementation of sklearn.metrics.roc_curve.

    This stub is not intended for real metric computation. It only exists so
    that optional dependencies inside transformers can import successfully.
    If this function is ever called at runtime, we raise a clear error.
    """
    raise RuntimeError(
        "stub sklearn.metrics.roc_curve was called. "
        "Install a compatible scikit-learn if you need real ROC computation."
    )


