"""Real inventory validation and command parity regressions."""

import json

import pytest

from phonebox.cli.main import main
from phonebox.dictionary import Dictionary
from phonebox.validation import (
    format_lexicon_validation,
    validate_lexicon,
    validate_lexicon_file,
)


def test_shared_parser_counts_and_variants():
    result = validate_lexicon(
        [
            "# ignored\n",
            "cat K AE1 T # comment\n",
            "cat(2)\tK AE2 T\n",
            "cat(3) K AE2 T\n",
            "bad\n",
        ],
        ["K", "AE1", "AE2", "T", "Z"],
    )
    assert (result.entries, result.unique_words, result.unique_pronunciations) == (
        3,
        1,
        2,
    )
    assert result.duplicate_entries == 1
    assert result.distinct_phones == 4
    assert result.unused_phones == ("Z",)
    assert not result.has_errors(strict=True)


def test_mixed_normalization_retains_all_forms_and_counts():
    result = validate_lexicon(["one é\n", "two e\u0301\n", "three e\u0301\n"], ["é"])
    assert result.distinct_phones == 2
    assert result.phones[0].occurrences == 3
    assert dict(result.phones[0].raw_counts) == {"é": 1, "e\u0301": 2}
    assert len(result.normalization_mismatches) == 1
    assert result.normalization_mismatches[0].lexicon_phone == "e\u0301"
    assert result.normalization_mismatches[0].occurrences == 2
    assert result.has_errors()
    both = validate_lexicon(["one é\n", "two e\u0301\n"], ["é", "e\u0301"])
    assert not both.has_errors(strict=True)


def test_missing_phone_aggregates_normalization_and_bounded_samples():
    result = validate_lexicon(["one é\n", "two e\u0301\n", "two é\n"], [], show_words=1)
    assert result.missing_phones[0].occurrences == 3
    assert result.missing_phones[0].words == ("one",)
    assert not result.has_errors()
    assert result.has_errors(strict=True)
    assert "(3 uses" in format_lexicon_validation(result)


@pytest.mark.parametrize("spec", [[1], [None], [""], ["A B"], "AB"])
def test_invalid_phone_spec_is_value_error(spec):
    with pytest.raises(ValueError, match="phoneset"):
        validate_lexicon([], spec)


@pytest.mark.parametrize("limit", [-1, 1.5, True])
def test_invalid_sample_limit(limit):
    with pytest.raises(ValueError, match="show_words"):
        validate_lexicon([], [], show_words=limit)


def test_exact_non_nfc_totals():
    result = validate_lexicon(["e\u0301 A\n"] * 25, ["A"], show_words=0)
    assert result.non_nfc_word_entries == 25
    assert result.non_nfc_words == ()
    assert "25 lexicon word entries" in format_lexicon_validation(result)


def test_cli_library_parity_and_bad_inputs(tmp_path, capsys):
    lexicon = tmp_path / "tiny.dict"
    spec = tmp_path / "phones.json"
    lexicon.write_text("one é\ntwo e\u0301\n", encoding="utf-8")
    spec.write_text(json.dumps(["é"]), encoding="utf-8")
    result = validate_lexicon_file(lexicon, ["é"])
    args = ["check", "--lexicon", str(lexicon), "--phoneset", str(spec)]
    assert main(args) == 1
    assert format_lexicon_validation(result) in capsys.readouterr().out
    spec.write_text("[1]", encoding="utf-8")
    assert main(args) == 2
    assert "phoneset elements" in capsys.readouterr().err
    spec.write_text('["é", "e\\u0301"]', encoding="utf-8")
    assert main(args + ["--show-words", "-1"]) == 2
    assert "show_words" in capsys.readouterr().err
    assert main(args) == 0
    capsys.readouterr()
    lexicon.unlink()
    assert main(args) == 2
    assert "Error:" in capsys.readouterr().err


def test_missing_phone_cli_warning_vs_strict(tmp_path, capsys):
    lexicon = tmp_path / "tiny.dict"
    spec = tmp_path / "phones.json"
    lexicon.write_text("word X\n", encoding="utf-8")
    spec.write_text("[]", encoding="utf-8")
    args = ["check", "--lexicon", str(lexicon), "--phoneset", str(spec)]
    assert main(args) == 0
    assert main(args + ["--strict"]) == 1
    assert "missing" in capsys.readouterr().out


@pytest.mark.parametrize("has_notice", [False, True])
def test_cmudict_manifest_only_advertises_available_notice(tmp_path, has_notice):
    from phonebox.dictionary import CMUDICT_LICENSE_URL

    folder = tmp_path / "cmudict"
    folder.mkdir()
    (folder / "cmudict.dict").write_text("cat K AE T\n", encoding="utf-8")
    if has_notice:
        (folder / "LICENSE").write_bytes(b"copyright and redistribution notice\n")
    Dictionary.create_manifest(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    source = manifest["sources"][0]
    assert source["license_url"] == CMUDICT_LICENSE_URL
    assert source["license"] == (
        "CMUdict license (see LICENSE)" if has_notice else "CMUdict license"
    )
    assert ("license_file" in source) == has_notice
    if has_notice:
        assert (tmp_path / source["license_file"]).is_file()


@pytest.mark.parametrize("existing_notice", [False, True])
def test_cmudict_failed_license_download_never_succeeds(
    tmp_path, monkeypatch, existing_notice
):
    import urllib.error
    from io import BytesIO

    folder = tmp_path / "cmudict"
    folder.mkdir()
    notice = folder / "LICENSE"
    original = b"previous copyright notice\n"
    if existing_notice:
        notice.write_bytes(original)

    def response(url, *, timeout):
        if url.endswith("/LICENSE"):
            raise urllib.error.URLError("notice unavailable")
        return BytesIO(b"cat K AE T\n")

    monkeypatch.setattr("phonebox.dictionary.urllib.request.urlopen", response)
    with pytest.raises(RuntimeError, match="required.*LICENSE"):
        Dictionary.fetch("cmudict", data_dir=tmp_path)
    assert notice.exists() == existing_notice
    if existing_notice:
        assert notice.read_bytes() == original


def test_cmudict_fetch_downloads_and_preserves_license_bytes(tmp_path, monkeypatch):
    from io import BytesIO

    from phonebox.dictionary import CMUDICT_LICENSE_URL, CMUDICT_REPO

    notice = b"Copyright fixture\nRedistribution conditions and disclaimer fixture.\n"
    urls = []

    def response(url, *, timeout):
        urls.append(url)
        return BytesIO(notice if url.endswith("/LICENSE") else b"cat K AE T\n")

    monkeypatch.setattr("phonebox.dictionary.urllib.request.urlopen", response)
    dictionary = Dictionary.fetch("cmudict", data_dir=tmp_path)
    assert len(dictionary) == 1
    assert (tmp_path / "cmudict" / "LICENSE").read_bytes() == notice
    assert f"{CMUDICT_REPO}/LICENSE" in urls
    assert all(url.startswith(CMUDICT_REPO + "/") for url in urls)
    Dictionary.create_manifest(tmp_path)
    source = json.loads((tmp_path / "manifest.json").read_text())["sources"][0]
    assert source["license_url"] == CMUDICT_LICENSE_URL
    assert (tmp_path / source["license_file"]).read_bytes() == notice
