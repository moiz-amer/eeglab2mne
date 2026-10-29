"""eeglab2mne: translate EEGLAB pipelines into auditable MNE-Python code."""

__version__ = "0.2.0"

from .core import Translation, history_from_set, translate  # noqa: E402

__all__ = ["translate", "history_from_set", "Translation", "__version__"]
