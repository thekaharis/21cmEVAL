"""21cmEVAL: end-to-end evaluation for 21cm field emulators.

The public surface is the runner plus the two data contracts:

    from cm21eval import Settings, Sample, evaluate, report

An adapter only has to implement ``predict(sample) -> {name: array}``; see
``cm21eval.adapters``.
"""
from .runner import evaluate
from .settings import Settings
from .spec import Meta, Sample

__version__ = "0.1.0"
__all__ = ["Settings", "Sample", "Meta", "evaluate", "__version__"]
