"""
Lightweight local stub of the `sklearn` package.

Transformers 4.51+ optionally depends on scikit-learn for some generation
utilities (e.g. candidate generator). On this project we do not require full
scikit-learn, and the globally installed version appears to be binary
incompatible with the current NumPy, causing import errors.

To avoid those issues while still allowing `from sklearn.metrics import roc_curve`
to succeed, we provide a minimal pure-Python stub of the `sklearn` package in
this repository.

Only the pieces needed by transformers are implemented in `sklearn.metrics`.
If you need real scikit-learn functionality for other purposes, install a
compatible scikit-learn in a separate environment instead of relying on this
stub.
"""


