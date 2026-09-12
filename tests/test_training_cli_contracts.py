"""Real training, dimensions, error boundaries, and input preservation."""

import importlib.util
import json
from io import StringIO

import pytest

from phonebox import G2P, Vectorizer, train_g2p, train_g2p_from_config
from phonebox.cli.main import main
from phonebox.core.em_align import EMAlign
from phonebox.core.g2p_model import G2PDecisionTree


@pytest.mark.parametrize("position", ["first", "last"])
@pytest.mark.parametrize("header", [False, True])
def test_vector_headers_match_stream_and_list_and_are_not_training_rows(
    position, header, alignments, tmp_path
):
    vec = Vectorizer(width=3, target_position=position)
    lines = alignments.read_text().splitlines()
    expected = vec.vectorize_file(lines, header=header)
    stream = StringIO()
    vec.vectorize_file(lines, stream, header=header)
    assert stream.getvalue().splitlines() == expected
    assert (expected[0].split() == vec.default_cols) == header
    path = tmp_path / "vectors.txt"
    path.write_text(stream.getvalue())
    data = vec.load_vectors_file(str(path))
    assert " ".join(vec.default_cols) not in data
    assert sum(data.values()) == len(expected) - int(header)
    # A genuine numeric-feature row must not be discarded as a header prefix.
    row = "P 0 1 9" if position == "first" else "0 1 9 P"
    path.write_text(row + "\n")
    assert vec.load_vectors_file(str(path)) == {row: 1}


def test_primary_training_rejects_inference_seed(dictionary, tmp_path, capsys):
    seed = tmp_path / "seed.g2p.gz"
    train_g2p(
        dictionary, locale="en", phoneset="cmu", width=3, prune=False, output=seed
    )
    loaded = G2PDecisionTree(model=str(seed))
    assert loaded.vectorizer.width == 3
    output = tmp_path / "existing.g2p.gz"
    output.write_bytes(b"preserve\n")
    options = {
        "locale": "it",
        "phoneset": "ipa",
        "width": 7,
        "model": str(seed),
        "prune": False,
        "output": output,
    }
    with pytest.raises(ValueError, match="model.*inference"):
        train_g2p(dictionary, **options)
    config = dict(dictionary=str(dictionary), **options)
    with pytest.raises(ValueError, match="model.*inference"):
        train_g2p_from_config(config)
    config["output"] = str(output)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    assert main(["train", "--config", str(path)]) == 2
    assert "inference" in capsys.readouterr().err
    assert output.read_bytes() == b"preserve\n"


@pytest.mark.parametrize("width", [-3, 0, 2])
def test_vectorizer_requires_positive_odd_width(width):
    with pytest.raises(ValueError, match="positive.*odd"):
        Vectorizer(width=width)


@pytest.mark.parametrize("command", ["align", "vectorize", "train"])
def test_negative_width_rejected_before_replacing_output(
    command, dictionary, alignments, tmp_path, capsys
):
    output = tmp_path / "existing.g2p.gz"
    output.write_bytes(b"preserve existing output\n")
    if command == "align":
        args = ["align", str(dictionary)]
    elif command == "vectorize":
        args = ["vectorize", str(alignments)]
    else:
        args = ["model", "train", "en", "--alignments", str(alignments)]
    assert main([*args, "--width", "-3", "-o", str(output)]) == 2
    assert output.read_bytes() == b"preserve existing output\n"
    assert "positive" in capsys.readouterr().err


@pytest.fixture
def dictionary(tmp_path):
    path = tmp_path / "tiny.dict"
    path.write_text(
        "cat K AE1 T\nbat B AE1 T\nhat HH AE1 T\nmat M AE1 T\n", encoding="utf-8"
    )
    return path


@pytest.fixture
def alignments(dictionary, tmp_path):
    path = tmp_path / "aligned.txt"
    train_g2p(dictionary, locale="en", phoneset="cmu", prune=False, alignments_out=path)
    return path


@pytest.mark.parametrize("target_first", [False, True])
def test_width_three_vectorization_and_prepared_training_roundtrip(
    alignments, tmp_path, target_first
):
    vectors = tmp_path / "width3.vec"
    flag = ["--target-first"] if target_first else []
    assert (
        main(["vectorize", str(alignments), "--width", "3", *flag, "-o", str(vectors)])
        == 0
    )
    assert all(len(line.split()) == 4 for line in vectors.read_text().splitlines())
    vec = Vectorizer(width=3, target_position="first" if target_first else "last")
    assert vectors.read_text().splitlines()[0].split() != vec.default_cols
    model = tmp_path / "width3.g2p.gz"
    assert (
        main(
            [
                "model",
                "train",
                "en",
                "--vectors",
                str(vectors),
                "--width",
                "3",
                *flag,
                "-o",
                str(model),
            ]
        )
        == 0
    )
    loaded = G2P(model=model, use_dict_fallback=False)
    assert loaded._dt.vectorizer.width == 3
    assert loaded.pronounce("cat") == ["K", "AE1", "T"]


def test_prepared_dimension_mismatch_is_rejected_before_training(
    alignments, tmp_path, capsys
):
    vec = Vectorizer(width=3)
    vectors = tmp_path / "width3.vec"
    with alignments.open() as infile:
        vectors.write_text("\n".join(vec.vectorize_file(infile)) + "\n")
    output = tmp_path / "bad.g2p.gz"
    assert (
        main(["model", "train", "en", "--vectors", str(vectors), "-o", str(output)])
        == 2
    )
    assert "width" in capsys.readouterr().err
    assert not output.exists()
    model = G2PDecisionTree(width=3)
    with pytest.raises(ValueError, match="width"):
        model.load_vectors_data([["a", "b"]], ["A"], [1])


@pytest.mark.parametrize("command", ["align", "vectorize", "export-vectors"])
@pytest.mark.parametrize("alias", ["same", "hardlink", "symlink"])
def test_output_alias_preserves_training_input(
    dictionary, alignments, tmp_path, command, alias
):
    source = alignments if command == "vectorize" else dictionary
    output = source if alias == "same" else tmp_path / "alias"
    if alias == "hardlink":
        output.hardlink_to(source)
    elif alias == "symlink":
        output.symlink_to(source)
    before = source.read_bytes()
    args = (
        [command, str(source)]
        if command != "export-vectors"
        else ["dict", command, str(source)]
    )
    assert main([*args, "-o", str(output)]) == 2
    assert source.read_bytes() == before


@pytest.mark.parametrize("content", ["", ";;; header\n# comment\n", "a A B\n"])
def test_no_admitted_data_is_a_shared_value_error(tmp_path, content):
    path = tmp_path / "empty.dict"
    path.write_text(content)
    em = EMAlign(Vectorizer(locale="en", phoneset_name="cmu"), parallel=False)
    for line in content.splitlines():
        em.add_line(line)
    assert not em.init_data
    with pytest.raises(ValueError, match="admissible"):
        em.align()
    assert (
        main(
            [
                "train",
                "--locale",
                "en",
                "--lexicon",
                str(path),
                "-o",
                str(tmp_path / "primary.g2p.gz"),
            ]
        )
        == 2
    )
    assert (
        main(
            [
                "recipe",
                str(path),
                "tts",
                "--no-prune",
                "-o",
                str(tmp_path / "recipe.g2p.gz"),
            ]
        )
        == 2
    )


def test_unknown_config_option_has_shared_validation(dictionary, tmp_path, capsys):
    config = {
        "dictionary": str(dictionary),
        "locale": "en",
        "output": str(tmp_path / "model.g2p.gz"),
        "unknown_option": 1,
    }
    with pytest.raises(
        ValueError, match="Unknown training config options: unknown_option"
    ):
        train_g2p_from_config(config)
    path = tmp_path / "unknown.json"
    path.write_text(json.dumps(config))
    assert main(["train", "--config", str(path)]) == 2
    assert "unknown_option" in capsys.readouterr().err


def test_expected_prepared_input_and_optional_backend_errors(
    alignments, tmp_path, capsys
):
    empty = tmp_path / "empty.txt"
    empty.write_text("")
    assert (
        main(
            [
                "model",
                "train",
                "en",
                "--alignments",
                str(empty),
                "-o",
                str(tmp_path / "empty.g2p.gz"),
            ]
        )
        == 2
    )
    if importlib.util.find_spec("sklearn") is None:
        assert (
            main(
                [
                    "model",
                    "train",
                    "en",
                    "--alignments",
                    str(alignments),
                    "--trainer",
                    "sklearn",
                    "-o",
                    str(tmp_path / "sklearn.g2p.gz"),
                ]
            )
            == 2
        )
        assert "phonebox[sklearn]" in capsys.readouterr().err


@pytest.mark.parametrize("iterations", ["0", "-1"])
def test_benchmark_requires_positive_iterations(tmp_path, iterations):
    assert (
        main(
            [
                "model",
                "benchmark",
                str(tmp_path / "missing.g2p.gz"),
                "--iterations",
                iterations,
            ]
        )
        == 2
    )


def test_benchmark_expected_model_errors(tmp_path):
    missing = tmp_path / "missing.g2p.gz"
    assert main(["model", "benchmark", str(missing), "--iterations", "1"]) == 2
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("not json")
    assert main(["model", "benchmark", str(corrupt), "--iterations", "1"]) == 2


def test_literal_dash_output_is_not_a_stdout_alias(dictionary, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "-"
    source.write_bytes(dictionary.read_bytes())
    before = source.read_bytes()
    assert main(["align", "-", "-o", "-"]) == 2
    assert source.read_bytes() == before


def test_dict_export_vectors_preserves_stdout_and_forwards_width(dictionary, capsys):
    assert main(["dict", "export-vectors", str(dictionary), "--width", "3"]) == 0
    rows = capsys.readouterr().out.splitlines()
    assert rows
    assert all(len(row.split()) == 4 for row in rows)


def test_config_rejects_core_phoneset_alias_with_public_option_guidance(
    dictionary, tmp_path, capsys
):
    config = {
        "dictionary": str(dictionary),
        "locale": "en",
        "phoneset_name": "cmu",
        "prune": False,
    }
    with pytest.raises(ValueError, match="phoneset_name.*phoneset"):
        train_g2p_from_config(config)
    config["output"] = str(tmp_path / "alias.g2p.gz")
    path = tmp_path / "alias.json"
    path.write_text(json.dumps(config))
    assert main(["train", "--config", str(path)]) == 2
    error = capsys.readouterr().err
    assert "phoneset_name" in error and "phoneset" in error
