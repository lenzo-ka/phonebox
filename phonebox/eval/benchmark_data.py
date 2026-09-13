"""Pinned public benchmark data with shared, model-independent preparation.

SIGMORPHON 2021 data is CC BY-SA 3.0, separately from this module's license.
See its source README and Ashby et al. (2021), DOI 10.18653/v1/2021.sigmorphon-1.13,
and Lee et al. (2020), https://aclanthology.org/2020.lrec-1.521/.
Raw datasets are fetched into caller-owned caches, never bundled here.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unicodedata
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from phonebox.constants import DOWNLOAD_TIMEOUT_SECONDS
from phonebox.eval.cmudict_compare import (
    CMUDICT_COMMIT,
    CMUDICT_LICENSE_URL,
    CMUDICT_SHA256,
    CMUDICT_URL,
    sha256_file,
)
from phonebox.experiments.split import split_lexicon_by_key
from phonebox.lexicon import parse_dict_line, strip_phone_stress

Pair = tuple[str, list[str]]
SIGMORPHON_COMMIT = "821bcdece5a47820872215969f275004b0abe80c"
SIGMORPHON_ROOT = (
    f"https://raw.githubusercontent.com/sigmorphon/2021-task1/{SIGMORPHON_COMMIT}"
)
SIGMORPHON_LICENSE_URL = f"https://github.com/sigmorphon/2021-task1/blob/{SIGMORPHON_COMMIT}/README.md#licensing"
TASK_CITATIONS = (
    "Ashby et al. (2021), Results of the Second SIGMORPHON Shared Task on "
    "Multilingual Grapheme-to-Phoneme Conversion, https://aclanthology.org/2021.sigmorphon-1.13/",
    "Lee et al. (2020), Massively multilingual pronunciation mining with WikiPron, "
    "https://aclanthology.org/2020.lrec-1.521/",
)


@dataclass(frozen=True)
class SourceFile:
    """An immutable download identity and its separate data licensing facts."""

    filename: str
    url: str
    sha256: str
    revision: str
    license: str
    license_url: str
    citations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return source provenance without a caller's local cache path."""
        return {
            "filename": self.filename,
            "url": self.url,
            "sha256": self.sha256,
            "revision": self.revision,
            "license": self.license,
            "license_url": self.license_url,
            "citations": list(self.citations),
        }


@dataclass(frozen=True)
class PreparedDataset:
    """Detached train/dev/test pairs and JSON-compatible preparation provenance.

    Spellings retain NFC identity and case. Phone tokens remain separate and
    unchanged except for explicitly requested CMU stress stripping. No locale
    rules, letter joins, dictionary exceptions or model-specific cooking apply.
    """

    name: str
    train: list[Pair]
    dev: list[Pair]
    test: list[Pair]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize all prepared pairs and provenance with strict JSON numbers."""
        return {
            "name": self.name,
            "train": [[word, list(phones)] for word, phones in self.train],
            "dev": [[word, list(phones)] for word, phones in self.dev],
            "test": [[word, list(phones)] for word, phones in self.test],
            "metadata": self.metadata,
        }


def _task_source(language: str, level: str, split: str, digest: str) -> SourceFile:
    filename = f"{language}_{split}.tsv"
    return SourceFile(
        filename,
        f"{SIGMORPHON_ROOT}/data/{level}/{filename}",
        digest,
        SIGMORPHON_COMMIT,
        "CC-BY-SA-3.0",
        SIGMORPHON_LICENSE_URL,
        TASK_CITATIONS,
    )


_TASK_FILES = {
    "italian": {
        "train": _task_source(
            "ita",
            "low",
            "train",
            "e12573b2f640aa799b5f87ba265a61f62a6212217170aa915875d651a3e0ea1b",
        ),
        "dev": _task_source(
            "ita",
            "low",
            "dev",
            "de30becaa09b9121730a07faa80feb08c864405556ab34938d9aaf89f678d521",
        ),
        "test": _task_source(
            "ita",
            "low",
            "test",
            "66f7049496757be938615dddbfd6fd5c846728c477888900fc20bfcc9084bd0c",
        ),
    },
    "french": {
        "train": _task_source(
            "fre",
            "medium",
            "train",
            "e9de4f5125d2478e50506c5ed278f69d5fb17d8d213f99f845305c24e28a55c2",
        ),
        "dev": _task_source(
            "fre",
            "medium",
            "dev",
            "3b42f14f318367dd94faa6d59e8fafabd845a3998800b6ce71a7c67e999490da",
        ),
        "test": _task_source(
            "fre",
            "medium",
            "test",
            "6202c2b7b8d5fbd76a4d40373f9564a9946671dab617438b534bd47c64071c92",
        ),
    },
}
_CMUDICT_SOURCE = SourceFile(
    "cmudict.dict",
    CMUDICT_URL,
    CMUDICT_SHA256,
    CMUDICT_COMMIT,
    "BSD-2-Clause",
    CMUDICT_LICENSE_URL,
    (
        "CMU Pronouncing Dictionary, Carnegie Mellon University and contributors, https://github.com/cmusphinx/cmudict",
    ),
)


def _fetch(source: SourceFile, cache_dir: Path) -> Path:
    """Verify existing caches or atomically install a verified new download."""
    target = cache_dir / source.filename
    if target.exists():
        _verify_hash(target, source)
        return target
    cache_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{source.filename}.", dir=cache_dir
    )
    try:
        with (
            os.fdopen(descriptor, "wb") as output,
            urllib.request.urlopen(  # noqa: S310
                source.url, timeout=DOWNLOAD_TIMEOUT_SECONDS
            ) as response,
        ):
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        _verify_hash(Path(temporary), source)
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target


def _verify_hash(path: Path, source: SourceFile) -> None:
    actual = sha256_file(path)
    if actual != source.sha256:
        raise ValueError(
            f"{source.filename} SHA-256 mismatch: expected {source.sha256}, got {actual}; "
            "remove the invalid cache file and fetch again"
        )


def _parse_task_tsv(path: Path) -> list[Pair]:
    """Read exactly two TSV fields; phones are whitespace-delimited tokens."""
    pairs: list[Pair] = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            columns = line.rstrip("\r\n").split("\t")
            if (
                len(columns) != 2
                or not columns[0]
                or any(character.isspace() for character in columns[0])
                or not columns[1].split()
            ):
                raise ValueError(
                    f"{path.name}:{number}: expected spelling TAB phone tokens"
                )
            pairs.append((columns[0], columns[1].split()))
    if not pairs:
        raise ValueError(f"{path.name}: empty task split")
    return pairs


def _prepare_split(pairs: Sequence[Pair]) -> tuple[list[Pair], dict[str, int]]:
    result: list[Pair] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for spelling, phones in pairs:
        word = unicodedata.normalize("NFC", spelling)
        if not word or not phones or any(not token for token in phones):
            raise ValueError(
                "Prepared examples require nonempty spelling and phone tokens"
            )
        identity = (word, tuple(phones))
        if identity not in seen:
            seen.add(identity)
            result.append((word, list(phones)))
    return result, {
        "source_entries": len(pairs),
        "prepared_entries": len(result),
        "words": len({word for word, _ in result}),
        "duplicates_removed": len(pairs) - len(result),
    }


def _split_digest(pairs: Sequence[Pair]) -> str:
    """Hash ordered pairs as UTF-8 compact JSON, with no trailing newline."""
    payload = json.dumps(
        pairs, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _assemble(
    name: str, splits: dict[str, list[Pair]], metadata: dict[str, Any]
) -> PreparedDataset:
    prepared: dict[str, list[Pair]] = {}
    counts = {}
    for split, pairs in splits.items():
        prepared[split], counts[split] = _prepare_split(pairs)
        if not prepared[split]:
            raise ValueError(f"{name}: empty prepared {split} split")
    keys = {split: {word for word, _ in pairs} for split, pairs in prepared.items()}
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        if overlap := keys[left] & keys[right]:
            raise ValueError(
                f"{name}: NFC spelling overlap between {left}/{right}: {min(overlap)!r}"
            )
    return PreparedDataset(
        name,
        prepared["train"],
        prepared["dev"],
        prepared["test"],
        {
            **metadata,
            "schema_version": 1,
            "deduplication": "NFC spelling and full transformed phone sequence; stable first occurrence within each split",
            "counts": counts,
            "prepared_sha256": {
                split: _split_digest(pairs) for split, pairs in prepared.items()
            },
            "prepared_hash_encoding": "UTF-8 compact JSON ordered [spelling, phone-token-list] pairs; no newline",
            "spelling_overlap": {"train_dev": 0, "train_test": 0, "dev_test": 0},
        },
    )


def load_dataset(
    name: str, cache_dir: str | Path, remove_stress: bool = False
) -> PreparedDataset:
    """Load cmudict/french/italian with verified bytes and shared preparation.

    French/Italian retain published SIGMORPHON 2021 splits and phone tokens;
    remove_stress=True is rejected for these already-prepared IPA task datasets.
    CMUdict lowercases NFC spellings, optionally applies the shared CMU stress
    transform, and splits complete spelling groups with seed1729: test10% capped
    at10000 words, then dev10% of remaining groups. Duplicate pairs are removed
    only after transformations and splitting; all pronunciation variants remain
    in the same split. An invalid existing cache fails closed without replacing
    it. Data license notices/citations are included in metadata, not relicensed.
    """
    if name not in {"cmudict", *_TASK_FILES}:
        raise ValueError("Unknown dataset; choose cmudict, french or italian")
    if not isinstance(remove_stress, bool):
        raise ValueError("remove_stress must be a boolean")
    cache = Path(cache_dir) / name
    if name in _TASK_FILES:
        if remove_stress:
            raise ValueError(
                "Task IPA phone tokens are preserved; remove_stress applies only to cmudict"
            )
        sources = _TASK_FILES[name]
        splits = {
            split: _parse_task_tsv(_fetch(source, cache))
            for split, source in sources.items()
        }
        return _assemble(
            name,
            splits,
            {
                "locale": "fr_FR" if name == "french" else "it_IT",
                "phoneset": "ipa",
                "sources": {
                    split: source.to_dict() for split, source in sources.items()
                },
                "preparation": {
                    "spelling": "NFC; case preserved",
                    "phones": "original tokens",
                    "remove_stress": False,
                },
                "split": {
                    "method": "published SIGMORPHON2021",
                    "revision": SIGMORPHON_COMMIT,
                },
            },
        )
    source = _CMUDICT_SOURCE
    pairs: list[Pair] = []
    with _fetch(source, cache).open(encoding="utf-8") as stream:
        for line in stream:
            if parsed := parse_dict_line(line):
                word, phones = parsed
                pairs.append(
                    (
                        unicodedata.normalize("NFC", word.lower()),
                        [strip_phone_stress(p, "cmu") for p in phones]
                        if remove_stress
                        else phones,
                    )
                )
    test, remaining = split_lexicon_by_key(
        pairs, key=lambda word: word, seed=1729, test_fraction=0.1, max_test=10000
    )
    dev, train = split_lexicon_by_key(
        remaining,
        key=lambda word: word,
        seed=1729,
        test_fraction=0.1,
        max_test=len(remaining),
    )
    return _assemble(
        name,
        {"train": train, "dev": dev, "test": test},
        {
            "locale": "en_US",
            "phoneset": "cmu",
            "sources": {"lexicon": source.to_dict()},
            "preparation": {
                "spelling": "NFC; lowercase",
                "phones": "shared CMU stress transform"
                if remove_stress
                else "original tokens",
                "remove_stress": remove_stress,
            },
            "split": {
                "method": "complete NFC lowercase spelling groups",
                "seed": 1729,
                "test_fraction": 0.1,
                "max_test_words": 10000,
                "dev_fraction_of_remaining_groups": 0.1,
                "rounding": "floor; minimum one held-out group per split; reject empty train",
            },
        },
    )
