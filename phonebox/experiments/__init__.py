"""G2P lexicon and scoring experiments (it_IT, pt_BR)."""

from phonebox.experiments.equiv import equiv_for_locale, locale_uses_relaxed_scoring
from phonebox.experiments.normalize import NORMALIZE_POLICIES, apply_train_normalize

__all__ = [
    "NORMALIZE_POLICIES",
    "apply_train_normalize",
    "equiv_for_locale",
    "locale_uses_relaxed_scoring",
]
