"""Exercise real dataset preparation using tiny, hash-pinned local caches."""

import hashlib
import io
import json

import pytest

from phonebox.eval import benchmark_data as data


def source(filename: str, content: bytes) -> data.SourceFile:
    return data.SourceFile(
        filename,
        "https://example.invalid/" + filename,
        hashlib.sha256(content).hexdigest(),
        "fixture-revision",
        "fixture-license",
        "https://example.invalid/license",
        ("Fixture",),
    )


def install_task(tmp_path, monkeypatch, splits):
    cache = tmp_path / "italian"
    cache.mkdir()
    sources = {}
    for split, content in splits.items():
        raw = content.encode()
        spec = source(f"ita_{split}.tsv", raw)
        (cache / spec.filename).write_bytes(raw)
        sources[split] = spec
    monkeypatch.setitem(data._TASK_FILES, "italian", sources)


def task_splits(train="si\ts i\nsì\ts i\ne\u0301\te\né\te\n"):
    return {"train": train, "dev": "casa\tk a z a\n", "test": "Città\tt͡ʃ i t a\n"}


def test_task_identity_tokens_dedup_provenance_and_digest(tmp_path, monkeypatch):
    install_task(tmp_path, monkeypatch, task_splits())
    result = data.load_dataset("italian", tmp_path)
    assert result.train == [("si", ["s", "i"]), ("sì", ["s", "i"]), ("é", ["e"])]
    assert result.test == [("Città", ["t͡ʃ", "i", "t", "a"])]
    assert result.metadata["counts"]["train"] == {
        "source_entries": 4,
        "prepared_entries": 3,
        "words": 3,
        "duplicates_removed": 1,
    }
    assert result.metadata["locale"] == "it_IT"
    serialized = json.dumps(
        result.train, ensure_ascii=False, separators=(",", ":")
    ).encode()
    assert (
        result.metadata["prepared_sha256"]["train"]
        == hashlib.sha256(serialized).hexdigest()
    )
    assert str(tmp_path) not in json.dumps(
        result.to_dict(), ensure_ascii=False, allow_nan=False
    )
    assert result.to_dict() == data.load_dataset("italian", tmp_path).to_dict()


@pytest.mark.parametrize(
    "train",
    [
        "",
        "word AA\n",
        "word\tAA\textra\n",
        "\tAA\n",
        "word\t\n",
        "two words\tAA\n",
        "\n",
    ],
)
def test_task_strict_tsv_rejects_malformed_splits(tmp_path, monkeypatch, train):
    install_task(tmp_path, monkeypatch, task_splits(train))
    with pytest.raises(ValueError):
        data.load_dataset("italian", tmp_path)


def test_nfc_cross_split_leakage_rejected_even_with_distinct_phones(
    tmp_path, monkeypatch
):
    splits = task_splits("e\u0301\tA\n")
    splits["test"] = "é\tB\n"
    install_task(tmp_path, monkeypatch, splits)
    with pytest.raises(ValueError, match="NFC spelling overlap"):
        data.load_dataset("italian", tmp_path)


def test_invalid_cache_fails_without_network_or_replacement(tmp_path, monkeypatch):
    install_task(tmp_path, monkeypatch, task_splits())
    target = tmp_path / "italian" / "ita_train.tsv"
    target.write_text("CHANGED")

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid cache must fail closed before network")

    monkeypatch.setattr(data.urllib.request, "urlopen", forbidden)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        data.load_dataset("italian", tmp_path)
    assert target.read_text() == "CHANGED"


@pytest.mark.parametrize("good", [False, True])
def test_fetch_checks_download_before_atomic_install(tmp_path, monkeypatch, good):
    spec = source("tiny.tsv", b"word\tA\n")

    class Response(io.BytesIO):
        pass

    monkeypatch.setattr(
        data.urllib.request,
        "urlopen",
        lambda *a, **k: Response(b"word\tA\n" if good else b"BAD"),
    )
    if good:
        assert data._fetch(spec, tmp_path).read_bytes() == b"word\tA\n"
    else:
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            data._fetch(spec, tmp_path)
        assert not (tmp_path / spec.filename).exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == (
        [spec.filename] if good else []
    )


def test_cmu_all_variants_case_aliases_stress_and_stable_group_splits(
    tmp_path, monkeypatch
):
    raw = (
        ";;; comment\n# comment\nCat K AE1 T\ncat(7) K AE0 T # variant\ncat(9) K AH0 T\n"
        + "".join(f"word{i} W ER1 D\n" for i in range(100))
    ).encode()
    spec = source("cmudict.dict", raw)
    monkeypatch.setattr(data, "_CMUDICT_SOURCE", spec)
    cache = tmp_path / "cmudict"
    cache.mkdir()
    (cache / spec.filename).write_bytes(raw)
    preserved = data.load_dataset("cmudict", tmp_path)
    stripped = data.load_dataset("cmudict", tmp_path, remove_stress=True)
    assert [
        preserved.metadata["counts"][s]["words"] for s in ["test", "dev", "train"]
    ] == [10, 9, 82]
    assert all(
        {w for w, _ in getattr(preserved, s)} == {w for w, _ in getattr(stripped, s)}
        for s in ["train", "dev", "test"]
    )
    locations = [
        s
        for s in ["train", "dev", "test"]
        if any(w == "cat" for w, _ in getattr(preserved, s))
    ]
    assert len(locations) == 1
    split = locations[0]
    assert len([p for w, p in getattr(preserved, split) if w == "cat"]) == 3
    assert [p for w, p in getattr(stripped, split) if w == "cat"] == [
        ["K", "AE", "T"],
        ["K", "AH", "T"],
    ]
    assert (
        sum(c["duplicates_removed"] for c in stripped.metadata["counts"].values()) == 1
    )
    assert data.load_dataset("cmudict", tmp_path, True).to_dict() == stripped.to_dict()
    assert (
        json.loads(json.dumps(stripped.to_dict(), allow_nan=False))["metadata"][
            "phoneset"
        ]
        == "cmu"
    )


def test_unknown_dataset_and_non_cmu_stress_rejected_before_fetch(tmp_path):
    with pytest.raises(ValueError, match="Unknown dataset"):
        data.load_dataset("unknown", tmp_path)
    with pytest.raises(ValueError, match="only to cmudict"):
        data.load_dataset("italian", tmp_path, True)
    with pytest.raises(ValueError, match="boolean"):
        data.load_dataset("cmudict", tmp_path, 1)  # type: ignore[arg-type]
    assert not tmp_path.exists() or not list(tmp_path.iterdir())


def test_tiny_cmu_cannot_produce_empty_training_population(tmp_path, monkeypatch):
    raw = b"a A\nb B\n"
    spec = source("cmudict.dict", raw)
    monkeypatch.setattr(data, "_CMUDICT_SOURCE", spec)
    cache = tmp_path / "cmudict"
    cache.mkdir()
    (cache / spec.filename).write_bytes(raw)
    with pytest.raises(ValueError, match="empty prepared train"):
        data.load_dataset("cmudict", tmp_path)


def test_transport_failure_leaves_no_partial_cache(tmp_path, monkeypatch):
    spec = source("tiny.tsv", b"word\tA\n")

    def unavailable(*args, **kwargs):
        raise OSError("fixture transport failure")

    monkeypatch.setattr(data.urllib.request, "urlopen", unavailable)
    with pytest.raises(OSError, match="fixture transport failure"):
        data._fetch(spec, tmp_path)
    assert not list(tmp_path.iterdir())


def test_prepared_phone_lists_are_detached_from_input():
    original = [("word", ["A"])]
    prepared, counts = data._prepare_split(original)
    prepared[0][1].append("B")
    assert original == [("word", ["A"])]
    assert counts["prepared_entries"] == 1
