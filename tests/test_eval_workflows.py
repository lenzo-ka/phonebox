import pytest

from phonebox.cli.main import main
from phonebox.eval import (
    accuracy,
    experiments,
    g2p_compare_all,
    g2p_sweep,
    multigram_units,
)


def test_sweep_accepts_explicit_lexicon_mapping(monkeypatch, tmp_path):
    class Vec:
        def cook_letters(self, word, g2p=True):
            return list(word)

        def cook_phones(self, phones):
            return phones

    class Model:
        def pronounce_letters(self, letters, word=None):
            return ["x"]

    seen = []

    def prepare(locale, path, **kwargs):
        seen.append(path)
        return Vec(), [], [("a", ["x"])], {}

    monkeypatch.setattr(g2p_sweep, "prepare_sweep_data", prepare)
    monkeypatch.setattr(g2p_sweep, "train_multigram", lambda *args, **kwargs: Model())
    monkeypatch.setattr(
        g2p_sweep,
        "evaluate",
        lambda *args, **kwargs: {"wer_pct": 0.0, "per_pct": 0.0},
    )
    path = tmp_path / "custom.lex"
    rows = g2p_sweep.run_g2p_sweep(
        {"xx": path}, locales=["xx"], letter_spans=[2], lm_orders=[1]
    )
    assert seen == [path]
    assert rows["xx"][(2, 1)]["wer_pct"] == 0


def test_compare_all_uses_explicit_mappings_without_printing(
    monkeypatch, tmp_path, capsys
):
    seen = []

    def compare(**kwargs):
        seen.append(kwargs)
        return {"locale": kwargs["locale"], "results": []}

    monkeypatch.setattr(g2p_compare_all, "run_compare", compare)
    lexicon = tmp_path / "custom.lex"
    model = tmp_path / "custom.g2p.gz"
    summaries = g2p_compare_all.run_compare_all(
        lexicons={"xx": lexicon}, baseline_models={"xx": model}
    )
    assert summaries[0]["locale"] == "xx"
    assert seen[0]["lexicon"] == lexicon
    assert seen[0]["baseline_model"] == model
    assert seen[0]["quiet"] is True
    assert capsys.readouterr() == ("", "")


def test_curated_directory_mode_rejects_unknown_locale_without_traceback(
    tmp_path, capsys
):
    status = main(
        ["compare", "sweep", "--lexicon-dir", str(tmp_path), "--locales", "xx"]
    )
    assert status == 2
    assert "unsupported curated locales" in capsys.readouterr().err


def test_curated_paths_accept_case_separator_and_bare_likely_variants(tmp_path):
    from phonebox.cli.commands.compare import _curated_paths

    paths = _curated_paths(tmp_path, ["IT-it", "pt"])
    assert paths == {
        "it_IT": tmp_path / "it_ipa.tsv",
        "pt": tmp_path / "pt_ipa.tsv",
    }


def test_sweep_canonicalizes_explicit_identity_and_preserves_path(
    monkeypatch, tmp_path
):
    path = tmp_path / "chosen.lex"
    seen = []

    def prepare(locale, supplied_path, **kwargs):
        seen.append((locale, supplied_path))
        return object(), [], [], {}

    monkeypatch.setattr(g2p_sweep, "prepare_sweep_data", prepare)
    rows = g2p_sweep.run_g2p_sweep(
        {"IT-it": path}, locales=["it_IT"], letter_spans=[], lm_orders=[]
    )
    assert seen == [("it_IT", path)]
    assert rows == {"it_IT": {}}


def test_explicit_path_selection_resolves_bare_likely_but_not_regional_fallback(
    tmp_path,
):
    from phonebox.eval.locale_registry import select_locale_paths

    path = tmp_path / "italian.lex"
    assert select_locale_paths({"IT-it": path}, ["it"]) == {"it": path}
    with pytest.raises(ValueError, match="no path supplied"):
        select_locale_paths({"it": path}, ["it_IT"])


def test_compare_all_resolves_bare_selector_against_regional_paths(
    monkeypatch, tmp_path
):
    seen = []

    def compare(**kwargs):
        seen.append(kwargs)
        return {"locale": kwargs["locale"], "results": []}

    monkeypatch.setattr(g2p_compare_all, "run_compare", compare)
    lexicon = tmp_path / "it.lex"
    summaries = g2p_compare_all.run_compare_all(
        lexicons={"it_IT": lexicon},
        baseline_models=None,
        no_config_joins=True,
        locales=["it"],
    )
    assert seen[0]["locale"] == "it"
    assert seen[0]["lexicon"] == lexicon
    assert summaries[0]["locale"] == "it"


def test_compare_all_cli_reports_missing_file_without_traceback(tmp_path, capsys):
    status = main(
        [
            "compare",
            "all",
            "--lexicon-dir",
            str(tmp_path),
            "--g2p-dir",
            str(tmp_path),
            "--locales",
            "it",
        ]
    )
    captured = capsys.readouterr()
    assert status == 2
    assert "lexicon not found" in captured.err
    assert "Traceback" not in captured.err


def test_sweep_cli_honors_selector_with_explicit_path(monkeypatch, tmp_path):
    seen = []

    def sweep(paths, **kwargs):
        seen.append((paths, kwargs["locales"]))
        return {"it": {}}

    monkeypatch.setattr(g2p_sweep, "run_g2p_sweep", sweep)
    lexicon = tmp_path / "chosen.lex"
    output = tmp_path / "sweep.md"
    status = main(
        [
            "compare",
            "sweep",
            "--lexicon",
            f"it_IT={lexicon}",
            "--locales",
            "it",
            "--output",
            str(output),
        ]
    )
    assert status == 0
    assert seen == [({"it_IT": lexicon}, ["it"])]


def test_sweep_runs_real_preparation_training_and_evaluation(tmp_path):
    lexicon = tmp_path / "tiny.tsv"
    lexicon.write_text(
        "\n".join(f"a{chr(98 + index)}\ta {chr(98 + index)}" for index in range(20))
        + "\n",
        encoding="utf-8",
    )
    rows = g2p_sweep.run_g2p_sweep(
        {"xx": lexicon},
        locales=["xx"],
        letter_spans=[1],
        lm_orders=[1],
        em_iterations=1,
        max_test=2,
    )
    metrics = rows["xx"][(1, 1)]
    assert metrics["n_test"] == 2
    assert 0 <= metrics["wer_pct"] <= 100


def test_unit_analysis_returns_structured_config_membership(monkeypatch, tmp_path):
    class Vec:
        config = {"join": {"letters": ["c h"], "ipa": ["t s"]}}

        def __init__(self, **kwargs):
            pass

        def multigram_config(self):
            return {"max_letter_span": 3, "max_phone_span": 2}

    class Aligner:
        q = {(("c", "h"), ("t", "s")): 0.7, (("a",), ("a",)): 0.9}

    class Model:
        aligner = Aligner()

    monkeypatch.setattr(multigram_units, "Vectorizer", Vec)
    monkeypatch.setattr(
        multigram_units, "load_lexicon", lambda path: [("ch", ["t", "s"]), ("a", ["a"])]
    )
    monkeypatch.setattr(
        multigram_units, "split_lexicon", lambda pairs, **kwargs: ([], pairs)
    )
    monkeypatch.setattr(
        multigram_units, "cook_pair", lambda vec, word, phones: (word, phones)
    )
    monkeypatch.setattr(
        multigram_units, "train_multigram", lambda *args, **kwargs: Model()
    )
    result = multigram_units.analyze_multigram_units("xx", tmp_path / "custom.lex")
    assert result.max_letter_span == 3
    assert result.units[0].configured_letter_join
    assert result.units[0].configured_phone_join


def test_unit_analysis_runs_real_training(tmp_path):
    lexicon = tmp_path / "tiny.tsv"
    lexicon.write_text(
        "\n".join(f"a{chr(98 + index)}\ta {chr(98 + index)}" for index in range(20))
        + "\n",
        encoding="utf-8",
    )
    result = multigram_units.analyze_multigram_units(
        "xx", lexicon, em_iterations=1, max_test=2, parallel_align=False
    )
    assert result.training_pairs == 18
    assert isinstance(result.units, tuple)


def test_accuracy_returns_counts_and_percentages(monkeypatch):
    class Model:
        class Vectorizer:
            @staticmethod
            def cook_phones(phones):
                return phones

        vectorizer = Vectorizer()

        def __init__(self, **kwargs):
            pass

        def load_prondict(self, lines):
            list(lines)

        def align(self):
            pass

        def train(self):
            pass

        def pronounce(self, word):
            return [word.upper()]

    monkeypatch.setattr(accuracy, "G2PDecisionTree", Model)
    result = accuracy.evaluate_accuracy(
        [("a", ["A"]), ("b", ["B"]), ("c", ["C"]), ("d", ["D"])],
        train_fraction=0.5,
    )
    assert (result.training_entries, result.test_entries) == (2, 2)
    assert result.word_accuracy == result.phone_accuracy == 100


def test_accuracy_runs_real_training_with_phoneset_cooking():
    entries = [(f"a{chr(98 + index)}", ["AH0", chr(66 + index)]) for index in range(20)]
    result = accuracy.evaluate_accuracy(
        entries, train_fraction=0.9, width=3, trainer="native"
    )
    assert (result.training_entries, result.test_entries) == (18, 2)
    assert result.total_phones == 4


def test_experiments_use_explicit_paths(monkeypatch, tmp_path, capsys):
    lexicon = tmp_path / "chosen.lex"
    model = tmp_path / "chosen.g2p.gz"
    seen = []

    def compare(**kwargs):
        seen.append(kwargs)
        return {
            "locale": "it_IT",
            "train_normalize_policy": kwargs.get("train_normalize_policy"),
            "results": [
                {
                    "model": "G2PDecisionTree",
                    "wer_pct": 1.0,
                    "per_pct": 1.0,
                    "per_equiv_pct": 1.0,
                },
                {
                    "model": "MultigramG2P",
                    "wer_pct": 1.0,
                    "per_pct": 1.0,
                    "per_equiv_pct": 1.0,
                },
            ],
        }

    monkeypatch.setattr(experiments, "run_compare", compare)
    monkeypatch.setattr(experiments, "load_lexicon", lambda path: [])
    monkeypatch.setattr(experiments, "split_lexicon", lambda pairs, **kwargs: ([], []))
    monkeypatch.setattr(
        experiments,
        "audit_normalize_delta",
        lambda *args: {
            "entries_changed": 0,
            "train_entries": 0,
            "phone_token_changes": 0,
        },
    )
    manifest = experiments.run_experiments(
        [experiments.ExperimentSpec("it_IT", lexicon, model)],
        output_dir=tmp_path / "out",
        skip_error_analysis=True,
    )
    assert seen[0]["lexicon"] == lexicon
    assert seen[0]["baseline_model"] == model
    assert manifest[0]["locale"] == "it_IT"
    assert capsys.readouterr() == ("", "")


def test_compare_report_names_supplied_baseline_model(tmp_path):
    output = tmp_path / "report.md"
    summary = {
        "locale": "it_IT",
        "lexicon": "chosen.lex",
        "n_entries": 2,
        "n_test": 1,
        "n_multi_pron": 0,
        "baseline_model": "/models/chosen.g2p.gz",
        "no_config_joins": False,
        "phone_equiv": False,
        "results": [
            {
                "model": name,
                "train_s": 0.0,
                "wer_pct": 0.0,
                "wer_relaxed_pct": 0.0,
                "per_pct": 0.0,
                "per_equiv_pct": 0.0,
                "pos_acc_pct": 100.0,
            }
            for name in ("G2PDecisionTree", "MultigramG2P")
        ],
    }
    g2p_compare_all.write_compare_all(
        output, [summary], config=g2p_compare_all.CompareAllConfig()
    )
    text = output.read_text(encoding="utf-8")
    assert "1:1 model: `/models/chosen.g2p.gz`" in text
    assert "1:1 model: `train-split" not in text


def test_evaluate_is_quiet_by_default(capsys):
    from phonebox.eval.g2p_compare import evaluate

    def broken(_word):
        raise RuntimeError("expected failure")

    result = evaluate("broken", broken, [("word", ["W"])])
    assert result["n_test"] == 1
    assert capsys.readouterr() == ("", "")


def test_experiments_resolve_bare_locale_without_changing_paths(monkeypatch, tmp_path):
    lexicon = tmp_path / "chosen.lex"
    model = tmp_path / "chosen.g2p.gz"
    seen = []

    def compare(**kwargs):
        seen.append(kwargs)
        return {
            "locale": kwargs["locale"],
            "train_normalize_policy": kwargs.get("train_normalize_policy"),
            "results": [
                {
                    "model": model_name,
                    "wer_pct": 0.0,
                    "per_pct": 0.0,
                    "per_equiv_pct": 0.0,
                }
                for model_name in ("G2PDecisionTree", "MultigramG2P")
            ],
        }

    monkeypatch.setattr(experiments, "run_compare", compare)
    monkeypatch.setattr(experiments, "load_lexicon", lambda path: [])
    monkeypatch.setattr(experiments, "split_lexicon", lambda pairs, **kwargs: ([], []))
    monkeypatch.setattr(
        experiments,
        "audit_normalize_delta",
        lambda *args: {
            "entries_changed": 0,
            "train_entries": 0,
            "phone_token_changes": 0,
        },
    )
    manifest = experiments.run_experiments(
        [experiments.ExperimentSpec("it", lexicon, model)],
        output_dir=tmp_path / "out",
        skip_error_analysis=True,
    )
    assert seen[0]["locale"] == "it_IT"
    assert seen[0]["lexicon"] == lexicon
    assert seen[0]["baseline_model"] == model
    assert manifest[0]["locale"] == "it_IT"
