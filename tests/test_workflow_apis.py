"""Public workflows preserve inputs and agree with installed CLI adapters."""

import json
import subprocess
import sys

import pytest

from phonebox import G2P, MultigramG2P, discover_joins, train_g2p, train_multigram
from phonebox.bundler import bundle_g2p
from phonebox.cli.main import main


@pytest.fixture
def dictionary(tmp_path):
    path = tmp_path / "tiny.dict"
    path.write_text(
        ";;; header\nship SH IH1 P # note\nshop SH AA1 P\ncat K AE1 T\nbat B AE1 T\n"
    )
    return path


def test_multigram_library_optional_export_and_cli_parity(dictionary, tmp_path, capsys):
    before = set(tmp_path.iterdir())
    result = train_multigram(
        dictionary, locale="EN-us", phoneset="cmu", em_iterations=1
    )
    assert result.units_path is None and result.lm_path is None
    assert set(tmp_path.iterdir()) == before
    assert capsys.readouterr().out == ""
    stem = tmp_path / "api.g2p"
    exported = train_multigram(
        dictionary, locale="en", phoneset="cmu", em_iterations=1, output=stem
    )
    assert exported.units_path.is_file() and exported.lm_path.is_file()
    assert not stem.exists()
    assert exported.metrics["aligned_entries"] == 4
    assert MultigramG2P.load(stem).pronounce("ship") == exported.model.pronounce("ship")
    cli_stem = tmp_path / "cli.g2p"
    assert (
        main(
            [
                "train-multigram",
                "--locale",
                "en",
                "--phoneset",
                "cmu",
                "--lexicon",
                str(dictionary),
                "--em-iterations",
                "1",
                "-o",
                str(cli_stem),
            ]
        )
        == 0
    )
    assert MultigramG2P.load(cli_stem).pronounce("ship") == exported.model.pronounce(
        "ship"
    )
    capsys.readouterr()
    assert main(["pronounce", "ship", "-m", str(cli_stem)]) == 0
    assert capsys.readouterr().out == "ship\tSH IH1 P\n"


def test_join_library_no_write_effective_settings_and_cli_parity(
    dictionary, tmp_path, capsys
):
    before = set(tmp_path.iterdir())
    result = discover_joins(dictionary, locale="en", max_iterations=1)
    assert result.output_path is None and set(tmp_path.iterdir()) == before
    assert result.settings["max_letter_span"] == 3
    assert result.settings["max_phone_span"] == 2
    assert capsys.readouterr().out == ""
    output = tmp_path / "joins.json"
    assert (
        main(
            [
                "suggest-joins",
                "--lexicon",
                str(dictionary),
                "--locale",
                "en",
                "--max-iterations",
                "1",
                "-o",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text()) == json.loads(json.dumps(result.to_dict()))


@pytest.mark.parametrize("alias_kind", ["same", "symlink", "hardlink"])
def test_join_and_multigram_preserve_aliased_dictionary(
    dictionary, tmp_path, alias_kind
):
    alias = dictionary
    if alias_kind != "same":
        alias = tmp_path / "alias.dict"
        if alias_kind == "symlink":
            alias.symlink_to(dictionary)
        else:
            alias.hardlink_to(dictionary)
    before = dictionary.read_bytes()
    with pytest.raises(ValueError, match="differ"):
        discover_joins(dictionary, output=alias)
    with pytest.raises(ValueError, match="differ"):
        train_multigram(dictionary, locale="en", output=alias)
    assert dictionary.read_bytes() == before


def test_multigram_sidecar_collision_preserves_input(dictionary, tmp_path):
    stem = tmp_path / "model.g2p"
    stem.with_suffix(".g2p.units.json").hardlink_to(dictionary)
    before = dictionary.read_bytes()
    with pytest.raises(ValueError, match="differ"):
        train_multigram(dictionary, locale="en", output=stem)
    assert dictionary.read_bytes() == before
    assert not stem.with_suffix(".g2p.lm.json").exists()


def test_bundle_alias_and_isolated_runner(dictionary, tmp_path):
    model = tmp_path / "tree.g2p.gz"
    trained = train_g2p(
        dictionary, locale="en", phoneset="cmu", prune=False, output=model
    )
    before = model.read_bytes()
    alias = tmp_path / "alias.g2p.gz"
    alias.hardlink_to(model)
    with pytest.raises(ValueError, match="differ"):
        bundle_g2p(str(model), str(alias))
    assert model.read_bytes() == before
    runner = tmp_path / "runner.py"
    bundle_g2p(str(model), str(runner))
    run = subprocess.run(
        [sys.executable, "-I", "-S", str(runner), "ship"],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0
    assert run.stdout == "ship\t" + " ".join(trained.model.pronounce("ship")) + "\n"


@pytest.mark.parametrize("n", [0, -1])
def test_public_nbest_rejects_nonpositive_even_for_exception(
    dictionary, tmp_path, capsys, n
):
    result = train_g2p(
        dictionary,
        locale="en",
        phoneset="cmu",
        prune=False,
        output=tmp_path / "tree.g2p.gz",
    )
    facade = G2P._from_trained_model(result.model)
    with pytest.raises(ValueError, match="positive"):
        facade.pronounce_nbest("ship", n)
    assert (
        main(["pronounce", "ship", "-m", str(result.output_path), "--nbest", str(n)])
        == 2
    )
    assert capsys.readouterr().out == ""


def test_normalize_missing_file_returns_expected_status(tmp_path, capsys):
    assert main(["normalize", "-f", str(tmp_path / "missing")]) == 2
    assert "Error:" in capsys.readouterr().err


@pytest.mark.parametrize("target", ["source", "model"])
def test_score_output_alias_preserves_inputs(dictionary, tmp_path, capsys, target):
    model = tmp_path / "tree.g2p.gz"
    train_g2p(dictionary, locale="en", phoneset="cmu", prune=False, output=model)
    source = tmp_path / "prons.jsonl"
    source.write_text('{"word":"ship","prons":["SH IH1 P"]}\n')
    selected = source if target == "source" else model
    alias = tmp_path / "alias.jsonl"
    alias.hardlink_to(selected)
    before = source.read_bytes(), model.read_bytes()
    assert main(["score-prons", str(source), "-m", str(model), "-o", str(alias)]) == 2
    assert (source.read_bytes(), model.read_bytes()) == before
    assert "Traceback" not in capsys.readouterr().err


def test_triage_output_alias_preserves_source(tmp_path):
    source = tmp_path / "review.tsv"
    source.write_text('{"word":"hello","prons":{"HH EH1 L OW":0.001}}\n')
    before = source.read_bytes()
    assert main(["find-suspicious", str(source), "--triage", "-o", str(tmp_path)]) == 2
    assert source.read_bytes() == before


def test_analysis_missing_inputs_return_expected_status(tmp_path, capsys):
    assert main(["find-suspicious", str(tmp_path / "missing"), "--zeros"]) == 2
    assert (
        main(
            [
                "score-prons",
                str(tmp_path / "missing"),
                "-m",
                str(tmp_path / "absent-model"),
            ]
        )
        == 2
    )
    assert "Traceback" not in capsys.readouterr().err


@pytest.mark.parametrize("collision", ["input-json", "input-markdown", "json-markdown"])
def test_cmudict_preflights_aliases_before_training(tmp_path, monkeypatch, collision):
    from phonebox.cli.commands import cmudict_compare

    lexicon = tmp_path / "input.dict"
    lexicon.write_text("cat K AE1 T\n")
    result = tmp_path / "result.json"
    markdown = tmp_path / "report.md"
    if collision == "input-json":
        result.hardlink_to(lexicon)
    elif collision == "input-markdown":
        markdown.symlink_to(lexicon)
    else:
        result.write_text("existing result")
        markdown.hardlink_to(result)

    def must_not_train(*args, **kwargs):
        pytest.fail("alias guard must run before training")

    monkeypatch.setattr(cmudict_compare, "run_cmudict_comparison", must_not_train)
    before = lexicon.read_bytes()
    assert (
        main(
            [
                "compare",
                "cmudict",
                "--lexicon",
                str(lexicon),
                "--refresh",
                str(result),
                "--markdown",
                str(markdown),
            ]
        )
        == 2
    )
    assert lexicon.read_bytes() == before


def test_cmudict_write_failure_is_expected_status(tmp_path, monkeypatch, capsys):
    from phonebox.cli.commands import cmudict_compare

    lexicon = tmp_path / "tiny.dict"
    lexicon.write_text("cat K AE1 T\n")
    result = tmp_path / "result-directory"
    result.mkdir()
    monkeypatch.setattr(cmudict_compare, "run_cmudict_comparison", lambda *a, **k: {})
    assert (
        main(
            [
                "compare",
                "cmudict",
                "--lexicon",
                str(lexicon),
                "--refresh",
                str(result),
                "--markdown",
                str(tmp_path / "report.md"),
            ]
        )
        == 2
    )
    assert "Traceback" not in capsys.readouterr().err


def test_multigram_optional_stress_and_saved_rewrite_policy(dictionary, tmp_path):
    stem = tmp_path / "rewritten.g2p"
    result = train_multigram(
        dictionary,
        locale="en",
        phoneset="cmu",
        em_iterations=1,
        remove_stress=True,
        no_config_joins=True,
        spelling_rewrites={"i": "e"},
        output=stem,
    )
    loaded = MultigramG2P.load(stem)
    assert loaded.preprocessor.cook_letters("ship", g2p=True) == ["s", "h", "e", "p"]
    assert loaded.pronounce("ship") == result.model.pronounce("ship")
    assert all(
        not phone.endswith(("0", "1", "2")) for phone in loaded.pronounce("ship")
    )


def test_invalid_workflow_span_reports_bad_input(dictionary, tmp_path):
    assert (
        main(
            [
                "train-multigram",
                "--lexicon",
                str(dictionary),
                "--locale",
                "en",
                "--max-letter-span",
                "0",
                "-o",
                str(tmp_path / "mg"),
            ]
        )
        == 2
    )
    assert (
        main(
            [
                "suggest-joins",
                "--lexicon",
                str(dictionary),
                "--max-phone-span",
                "0",
                "-o",
                str(tmp_path / "joins.json"),
            ]
        )
        == 2
    )
    assert not (tmp_path / "mg.units.json").exists()
    assert not (tmp_path / "joins.json").exists()


def test_impossible_spans_fail_before_writing_and_mixed_inputs_train(tmp_path, capsys):
    dictionary = tmp_path / "impossible.dict"
    dictionary.write_text("a A B C D E\n", encoding="utf-8")
    stem = tmp_path / "model"
    units, lm = MultigramG2P.export_paths(stem)
    report = tmp_path / "joins.json"
    for path in (units, lm, report):
        path.write_bytes(b"preserve existing artifact")
    before = {path: path.read_bytes() for path in (dictionary, units, lm, report)}
    with pytest.raises(ValueError, match="no admissible input pairs"):
        train_multigram(
            dictionary,
            locale="en",
            output=stem,
            max_letter_span=1,
            max_phone_span=1,
            em_iterations=1,
        )
    with pytest.raises(ValueError, match="no admissible input pairs"):
        discover_joins(
            dictionary,
            locale="en",
            output=report,
            max_letter_span=1,
            max_phone_span=1,
            max_iterations=1,
        )
    common = [
        "--lexicon",
        str(dictionary),
        "--locale",
        "en",
        "--max-letter-span",
        "1",
        "--max-phone-span",
        "1",
    ]
    assert (
        main(["train-multigram", *common, "--em-iterations", "1", "-o", str(stem)]) == 2
    )
    assert (
        main(["suggest-joins", *common, "--max-iterations", "1", "-o", str(report)])
        == 2
    )
    assert "no admissible input pairs" in capsys.readouterr().err
    assert {path: path.read_bytes() for path in before} == before
    dictionary.write_text("a A B C D E\nb B\n", encoding="utf-8")
    result = train_multigram(
        dictionary, locale="en", max_letter_span=1, max_phone_span=1, em_iterations=1
    )
    assert result.model.pronounce("b") == ["B"]
    assert (
        discover_joins(
            dictionary, max_letter_span=1, max_phone_span=1, max_iterations=1
        ).n_entries
        == 2
    )


def test_aligner_admission_respects_minimum_phone_span():
    from phonebox.core.multigram_align import MultigramAligner

    with pytest.raises(ValueError, match="no admissible input pairs"):
        MultigramAligner(max_letter_span=1, max_phone_span=1, min_phone_span=1).fit(
            [(("a", "b"), ("A",))]
        )
    with pytest.raises(ValueError, match="no admissible input pairs"):
        MultigramAligner().fit([((), ("A",))])
    aligner = MultigramAligner(max_phone_span=0, max_iterations=1).fit([(("a",), ())])
    assert aligner.q


@pytest.mark.parametrize("command", ["train-multigram", "suggest-joins"])
def test_verbose_cli_emits_progress_in_fresh_process(dictionary, tmp_path, command):
    output = tmp_path / command
    args = [
        sys.executable,
        "-m",
        "phonebox.cli.main",
        command,
        "--lexicon",
        str(dictionary),
        "--locale",
        "en",
        "-o",
        str(output),
    ]
    quiet = subprocess.run(args, capture_output=True, text=True)
    verbose = subprocess.run([*args, "--verbose"], capture_output=True, text=True)
    assert quiet.returncode == verbose.returncode == 0
    assert "MultigramAligner:" not in quiet.stderr
    assert "MultigramAligner:" in verbose.stderr
    assert "iter 1: LL=" in verbose.stderr


def test_default_workflow_apis_remain_quiet(dictionary, capsys):
    train_multigram(dictionary, locale="en", em_iterations=1)
    discover_joins(dictionary, locale="en", max_iterations=1)
    captured = capsys.readouterr()
    assert not captured.out and not captured.err
