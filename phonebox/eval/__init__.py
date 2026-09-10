"""G2P evaluation: 1:1 vs MultigramG2P comparison."""

from phonebox.eval.accuracy import (
    AccuracyResult,
    evaluate_accuracy,
    load_pronunciation_entries,
)
from phonebox.eval.experiments import ExperimentSpec, run_experiments
from phonebox.eval.g2p_compare import (
    build_gold_variants,
    build_train_exceptions,
    cook_pair,
    evaluate,
    load_baseline,
    load_lexicon,
    predict_cooked_phones,
    print_results_table,
    run_compare,
    train_baseline,
    train_multigram,
)
from phonebox.eval.g2p_compare_all import (
    CompareAllConfig,
    run_compare_all,
    write_compare_all,
)
from phonebox.eval.g2p_sweep import format_g2p_sweep, run_g2p_sweep
from phonebox.eval.locale_registry import EVALUATION_LOCALES, EvaluationLocale
from phonebox.eval.multigram_units import (
    MultigramUnit,
    MultigramUnitAnalysis,
    analyze_multigram_units,
    format_multigram_units,
)

__all__ = [
    "build_gold_variants",
    "AccuracyResult",
    "CompareAllConfig",
    "EVALUATION_LOCALES",
    "EvaluationLocale",
    "ExperimentSpec",
    "MultigramUnit",
    "MultigramUnitAnalysis",
    "analyze_multigram_units",
    "build_train_exceptions",
    "cook_pair",
    "evaluate",
    "evaluate_accuracy",
    "format_g2p_sweep",
    "format_multigram_units",
    "load_lexicon",
    "load_pronunciation_entries",
    "load_baseline",
    "predict_cooked_phones",
    "print_results_table",
    "run_compare",
    "run_compare_all",
    "run_experiments",
    "run_g2p_sweep",
    "train_baseline",
    "train_multigram",
    "write_compare_all",
]
