"""Adapters preserve the declared population and reject ambiguous tool output."""

import json
from pathlib import Path

import pytest

from phonebox.eval.benchmark import (
    _cart_convergence,
    _multigram_convergence,
    _predictions,
    run_benchmark,
)
from phonebox.eval.benchmark_data import _assemble


def dataset():
    return _assemble(
        "tiny",
        {
            "train": [("ab", ["A", "B"]), ("ac", ["A", "C"]), ("é", ["E"])],
            "dev": [("ba", ["B", "A"])],
            "test": [("ca", ["C", "A"]), ("ca", ["A"])],
        },
        {"phoneset": "ipa"},
    )


@pytest.mark.parametrize("system", ["cart", "multigram"])
def test_actual_native_training_identity_and_population(tmp_path, system):
    report = run_benchmark(dataset(), system, tmp_path / system)
    assert report["metrics"]["n_test"] == 1
    assert report["training"]["supplied_entries"] == 3
    assert report["training"]["retained_entries"] == 3
    assert report["training"]["model_bytes"] > 0
    assert report["training"]["dictionary_entries"] == 0
    convergence = report["training"]["convergence"]
    assert convergence["max_iterations"] == 100
    assert convergence["converged"] is True
    assert convergence["cap_censored"] is False
    assert convergence["stop_reason"] == "converged"
    trace_key = "history" if system == "cart" else "observed_pre_update_loglik_history"
    assert convergence["iterations"] == len(convergence[trace_key])
    if system == "cart":
        from phonebox.core.g2p_model import G2PDecisionTree

        reloaded = G2PDecisionTree()
        reloaded.load_model(str(tmp_path / system / "model.g2p.gz"))
        assert reloaded.exceptions == {}
    else:
        from phonebox.core.multigram_g2p import MultigramG2P
        from phonebox.core.multigram_lm import MultigramLM

        assert report["settings"]["g2p_version"] == MultigramG2P.VERSION
        assert report["settings"]["lm_version"] == MultigramLM.VERSION
        assert report["settings"]["scoring"] == MultigramG2P.SCORING
        assert report["settings"]["em_max_iterations"] == 100
        saved = json.loads((tmp_path / system / "model.g2p.units.json").read_text())
        assert saved["exceptions"] == {}
    assert report["settings"]["letter_preprocessing"]["source"] == {
        "norm_rules": None,
        "g2p_rules": None,
    }
    assert report["settings"]["use_dict_fallback"] is False
    assert report["timings"]["prediction_seconds"] >= 0
    assert str(tmp_path) not in json.dumps(report)


@pytest.mark.parametrize(
    "text", ["x\tA\nx\tA\n", "unknown\tA\n", "x\t0.25\tA\n", "x\t0.25\n", "garbage\n"]
)
def test_ambiguous_external_output_refused(tmp_path, text):
    path = tmp_path / "predictions"
    path.write_text(text)
    with pytest.raises(ValueError):
        _predictions(path, {"x"}, {"A"})


def test_missing_output_does_not_shrink_evaluation(tmp_path):
    from phonebox.eval.benchmark import _metrics

    path = tmp_path / "predictions"
    path.write_text("y\tB\n")
    parsed = _predictions(path, {"x", "y"}, {"A", "B"})
    result = _metrics(parsed.__getitem__, [("x", ["A"]), ("y", ["B"])])
    assert result["n_test"] == 2
    assert result["prediction_errors"] == 1
    assert result["empty_predictions"] == 1
    assert result["wer_relaxed_pct"] == 50
    assert result["per_variant_pct"] == 50


def test_sequitur_actual_subprocess_selection_and_test_isolation(tmp_path):
    executable = tmp_path / "sequitur"
    executable.write_text("""#!/usr/bin/env python3
import os, pathlib, sys
assert os.environ['OMP_NUM_THREADS'] == '1'
args = sys.argv[1:]
if '--write-model' in args:
    path = pathlib.Path(args[args.index('--write-model')+1])
    order = int(path.name.split('-')[1])
    assert ('--ramp-up' in args) == (order > 1)
    path.write_text(str(order))
    print('iteration: 20')
    print('LL devel: -12.0')
    print('iteration converged.')
else:
    order = int(pathlib.Path(args[args.index('--model')+1]).read_text())
    words = pathlib.Path(args[args.index('--apply')+1]).read_text().splitlines()
    for word in reversed(words):
        if word == 'ba':
            phones = 'B A' if order >= 2 else 'A'
        else:
            phones = 'C A'
        print(word+'\\t'+phones)
""")
    executable.chmod(0o755)
    report = run_benchmark(
        dataset(), "sequitur", tmp_path / "run", sequitur_executable=executable
    )
    assert report["settings"]["selected_order"] == 2
    assert report["metrics"]["wer_relaxed_pct"] == 0
    assert report["training"]["retained_entries"] is None
    assert report["provenance"]["tool"]["source_revision"] is None
    assert len(report["provenance"]["tool"]["sha256"]) == 64
    assert str(tmp_path) not in json.dumps(report)


def test_split_overlap_guard_fires_even_when_recorded_counts_still_match():
    from phonebox.eval.benchmark import _validate_dataset
    from phonebox.eval.benchmark_data import _split_digest

    prepared = dataset()
    _validate_dataset(prepared)
    # Replace the single dev entry with a train entry and re-record the dev
    # digest: every recorded count and hash still matches, so only the
    # population-overlap guard can refuse this.
    prepared.dev[:] = prepared.train[:1]
    prepared.metadata["prepared_sha256"]["dev"] = _split_digest(prepared.dev)
    with pytest.raises(ValueError, match="leaked"):
        _validate_dataset(prepared)


def test_split_leakage_and_reused_work_directory_refused(tmp_path):
    prepared = dataset()
    prepared.dev[:] = prepared.train[:1]
    with pytest.raises(ValueError, match="recorded|leaked"):
        run_benchmark(prepared, "cart", tmp_path / "leak")
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep").write_text("keep")
    with pytest.raises(ValueError, match="empty"):
        run_benchmark(dataset(), "cart", occupied)
    assert (occupied / "keep").read_text() == "keep"


def test_mutated_pairs_cannot_reuse_prepared_provenance(tmp_path):
    prepared = dataset()
    prepared.train[0] = ("db", ["A", "B"])
    with pytest.raises(ValueError, match="recorded prepared hash"):
        run_benchmark(prepared, "cart", tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_wrong_recorded_population_refused(tmp_path):
    prepared = dataset()
    prepared.metadata["counts"]["train"]["words"] += 1
    with pytest.raises(ValueError, match="recorded counts"):
        run_benchmark(prepared, "cart", tmp_path / "rejected")


def test_invalid_tool_receipt_refused_before_execution(tmp_path):
    executable = tmp_path / "sequitur"
    executable.write_text("#!/bin/sh\nexit 99\n")
    executable.chmod(0o755)
    receipt = executable.with_name(executable.name + ".provenance.json")
    receipt.write_text(json.dumps({"build": {"location": str(tmp_path)}}))
    with pytest.raises(ValueError, match="deployment path"):
        run_benchmark(
            dataset(), "sequitur", tmp_path / "run", sequitur_executable=executable
        )
    assert not list((tmp_path / "run").iterdir())


@pytest.mark.parametrize(
    "receipt",
    [
        '{"version":NaN}',
        '{"version":2}',
        '{"build":{"command":"make PREFIX=/temporary/location"}}',
    ],
)
def test_invalid_receipt_schema_is_rejected(tmp_path, receipt):
    from phonebox.eval.benchmark import _tool_identity

    executable = tmp_path / "tool"
    executable.write_text("executable")
    provenance = tmp_path / "receipt.json"
    provenance.write_text(receipt)
    with pytest.raises(ValueError):
        _tool_identity(executable, provenance)


def test_phonetisaurus_pipeline_uses_no_test_references_for_training(tmp_path):
    prefix = tmp_path / "prefix"
    binaries = prefix / "bin"
    binaries.mkdir(parents=True)
    script = """#!/usr/bin/env python3
import os, pathlib, sys
assert os.environ['BLIS_NUM_THREADS'] == '1'
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
def option(name):
    return next(value.split('=',1)[1] for value in args if value.startswith(name+'='))
if name == 'phonetisaurus-align':
    assert '--seq1_max=2' in args and '--seq2_max=2' in args
    assert 'ca\\t' not in pathlib.Path(option('--input')).read_text()
    pathlib.Path(option('--ofile')).write_text('aligned')
elif name == 'estimate-ngram':
    assert args[args.index('-o')+1] == '8'
    pathlib.Path(args[args.index('-wl')+1]).write_text('arpa')
elif name == 'phonetisaurus-arpa2wfst':
    pathlib.Path(option('--ofile')).write_text('fst')
else:
    assert '--nbest=1' in args and '--print_scores=false' in args
    for word in pathlib.Path(option('--wordlist')).read_text().splitlines():
        print(word+'\\tC A')
"""
    for name in (
        "phonetisaurus-align",
        "estimate-ngram",
        "phonetisaurus-arpa2wfst",
        "phonetisaurus-g2pfst",
    ):
        executable = binaries / name
        executable.write_text(script)
        executable.chmod(0o755)
        receipt = {
            "version": "mitlm-version"
            if name == "estimate-ngram"
            else "phonetisaurus-version",
            "source_revision": "mitlm-revision"
            if name == "estimate-ngram"
            else "phonetisaurus-revision",
            "build": {"compiler": "test compiler"},
        }
        Path(str(executable) + ".provenance.json").write_text(json.dumps(receipt))
    (prefix / "provenance.json").write_text(
        json.dumps(
            {
                "version": "wrong-shared-version",
                "source_revision": "wrong-shared-revision",
            }
        )
    )
    report = run_benchmark(
        dataset(), "phonetisaurus", tmp_path / "run", phonetisaurus_prefix=prefix
    )
    assert report["metrics"]["wer_relaxed_pct"] == 0
    assert report["training"]["retained_entries"] is None
    assert report["training"]["model_bytes"] == 3
    assert len(report["provenance"]["tools"]) == 4
    tools = report["provenance"]["tools"]
    assert tools["estimate-ngram"]["version"] == "mitlm-version"
    assert tools["estimate-ngram"]["source_revision"] == "mitlm-revision"
    for name in tools:
        if name != "estimate-ngram":
            assert tools[name]["version"] == "phonetisaurus-version"
            assert tools[name]["source_revision"] == "phonetisaurus-revision"
    assert str(tmp_path) not in json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "final_stop", ["iteration converged.", "maximum number of iterations reached."]
)
def test_sequitur_restarts_capped_order_before_test(tmp_path, final_stop):
    executable = tmp_path / "sequitur"
    executable.write_text(
        """#!/usr/bin/env python3
import pathlib, sys
args = sys.argv[1:]
if '--write-model' in args:
    path = pathlib.Path(args[args.index('--write-model')+1])
    order = int(path.name.split('-')[1])
    cap = int(args[args.index('--max-iterations')+1])
    assert int(args[args.index('--min-iterations')+1]) == 20
    if order > 1:
        previous = pathlib.Path(args[args.index('--model')+1])
        assert int(previous.read_text()) == order - 1
        assert previous.name.endswith('-extended')
    else:
        assert '--model' not in args
    path.write_text(str(order))
    print('iteration: '+str(cap-1))
    print('LL devel: -12.0')
    print('maximum number of iterations reached.' if cap == 100 else FINAL_STOP)
else:
    words = pathlib.Path(args[args.index('--apply')+1])
    if words.name == 'test.words':
        assert (words.parent/'model-3-extended').exists()
    for word in words.read_text().splitlines():
        print(word+'\\t'+('B A' if word == 'ba' else 'C A'))
""".replace("FINAL_STOP", repr(final_stop))
    )
    executable.chmod(0o755)
    report = run_benchmark(
        dataset(), "sequitur", tmp_path / "run", sequitur_executable=executable
    )
    settings = report["settings"]
    assert [
        (row["order"], row["max_iterations"]) for row in settings["training_attempts"]
    ] == [(1, 100), (1, 200), (2, 100), (2, 200), (3, 100), (3, 200)]
    assert settings["iteration_limited_orders"] == (
        [1, 2, 3] if final_stop.startswith("maximum") else []
    )
    assert len(list((tmp_path / "run").glob("model-*"))) == 6


@pytest.mark.parametrize(
    "log",
    [
        "",
        "iteration failed.",
        "iteration converged.",
        "iteration: 0\nLL devel: 1e999\niteration converged.",
    ],
)
def test_sequitur_stop_evidence_fails_closed(tmp_path, log):
    from phonebox.eval.benchmark import _sequitur_stop

    output = tmp_path / "log"
    output.write_text(log)
    with pytest.raises(ValueError, match="Sequitur"):
        _sequitur_stop(output)


@pytest.mark.parametrize(
    "budgets", [(20, 10, 200), (20, 100, 50), (0, 100, 200), (True, 100, 200)]
)
def test_sequitur_invalid_budgets_before_side_effects(tmp_path, budgets):
    with pytest.raises(ValueError, match="iteration budgets"):
        run_benchmark(
            dataset(),
            "sequitur",
            tmp_path / "run",
            sequitur_min_iterations=budgets[0],
            sequitur_max_iterations=budgets[1],
            sequitur_extension_iterations=budgets[2],
        )
    assert not (tmp_path / "run").exists()


def test_native_receipts_distinguish_iteration_caps(monkeypatch):
    from phonebox.core.em_align import EMAlign
    from phonebox.core.multigram_align import MultigramAligner
    from phonebox.core.vectorizer import Vectorizer

    cart = EMAlign(
        locale=Vectorizer(phoneset_name="ipa"),
        max_iterations=1,
        parallel=False,
        verbose=False,
    )
    cart.load_prondict(["ab A B", "ac A C"])
    monkeypatch.setattr(cart, "align_once", lambda iteration: (1, 0.5))
    cart.align()
    cart_receipt = _cart_convergence(cart)
    assert cart_receipt["history"] == [{"iteration": 1, "changed": 1, "ratio": 0.5}]
    mg = MultigramAligner(max_iterations=1)
    mg.fit([(["a"], ["A"])])
    mg_receipt = _multigram_convergence(mg)
    assert mg_receipt["last_relative_change"] is None
    for receipt in (cart_receipt, mg_receipt):
        assert receipt["iterations"] == 1
        assert receipt["stop_reason"] == "iteration_limit"
        assert receipt["cap_censored"] is True
        assert receipt["converged"] is False


def test_native_receipts_refuse_missing_observations():
    from phonebox.core.em_align import EMAlign
    from phonebox.core.multigram_align import MultigramAligner

    with pytest.raises(ValueError, match="convergence evidence"):
        _cart_convergence(EMAlign())
    with pytest.raises(ValueError, match="convergence evidence"):
        _multigram_convergence(MultigramAligner())


@pytest.mark.parametrize("over_cap", [False, True])
def test_native_receipts_refuse_incomplete_or_over_cap_traces(monkeypatch, over_cap):
    from phonebox.core.em_align import EMAlign
    from phonebox.core.multigram_align import MultigramAligner

    count = 3 if over_cap else 1
    history = [{"iteration": i + 1, "changed": 1, "ratio": 0.5} for i in range(count)]
    monkeypatch.setattr(EMAlign, "alignment_history", property(lambda self: history))
    monkeypatch.setattr(
        MultigramAligner, "loglik_history", property(lambda self: [-1.0] * count)
    )
    with pytest.raises(ValueError, match="incomplete or over-cap"):
        _cart_convergence(EMAlign(max_iterations=2))
    with pytest.raises(ValueError, match="incomplete or over-cap"):
        _multigram_convergence(MultigramAligner(max_iterations=2))


@pytest.mark.parametrize("last_value", ["nan", "inf", "-inf", "invalid", ""])
def test_sequitur_rejects_bad_likelihood_after_finite_observation(tmp_path, last_value):
    from phonebox.eval.benchmark import _sequitur_stop

    output = tmp_path / "log"
    output.write_text(
        "iteration: 0\nLL devel: -12\niteration: 1\n"
        f"LL devel: {last_value}\niteration converged.\n"
    )
    with pytest.raises(ValueError, match="development evidence"):
        _sequitur_stop(output)


def test_sequitur_rejects_missing_iteration_likelihood(tmp_path):
    from phonebox.eval.benchmark import _sequitur_stop

    output = tmp_path / "log"
    output.write_text(
        "iteration: 0\nLL devel: -12\niteration: 1\niteration converged.\n"
    )
    with pytest.raises(ValueError, match="development evidence"):
        _sequitur_stop(output)
