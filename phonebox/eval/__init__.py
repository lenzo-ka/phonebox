"""G2P evaluation: 1:1 vs MultigramG2P comparison."""

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

__all__ = [
    "build_gold_variants",
    "build_train_exceptions",
    "cook_pair",
    "evaluate",
    "load_lexicon",
    "load_baseline",
    "predict_cooked_phones",
    "print_results_table",
    "run_compare",
    "train_baseline",
    "train_multigram",
]
