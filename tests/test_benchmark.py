"""Adapters preserve the declared population and reject ambiguous tool output."""

import json

import pytest

from phonebox.eval.benchmark import _predictions, run_benchmark
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
    if system == "cart":
        from phonebox.core.g2p_model import G2PDecisionTree

        reloaded = G2PDecisionTree()
        reloaded.load_model(str(tmp_path / system / "model.g2p.gz"))
        assert reloaded.exceptions == {}
    else:
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
    order = int(path.name.rsplit('-',1)[1])
    assert ('--ramp-up' in args) == (order > 1)
    path.write_text(str(order))
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
    report = run_benchmark(
        dataset(), "phonetisaurus", tmp_path / "run", phonetisaurus_prefix=prefix
    )
    assert report["metrics"]["wer_relaxed_pct"] == 0
    assert report["training"]["retained_entries"] is None
    assert report["training"]["model_bytes"] == 3
    assert len(report["provenance"]["tools"]) == 4
    assert str(tmp_path) not in json.dumps(report, allow_nan=False)
