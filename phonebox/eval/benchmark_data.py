"""Pinned public benchmark data with shared, model-independent preparation.

WikiPron data has its own Wiktionary terms (CC BY-SA 4.0 and GFDL), not
WikiPron software's Apache license or this module's license. See the pinned
WikiPron README and Lee et al. (2020), https://aclanthology.org/2020.lrec-1.521/.
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
WIKIPRON_COMMIT = "d282e848a211ea31cfd730f0ced8bc8cdab9e83d"
WIKIPRON_ROOT = f"https://raw.githubusercontent.com/CUNY-CL/wikipron/{WIKIPRON_COMMIT}"
WIKIPRON_LICENSE_URL = "https://en.wiktionary.org/wiki/Wiktionary:Copyrights"
WIKIPRON_CITATIONS = (
    "Lee et al. (2020), Massively multilingual pronunciation mining with WikiPron, "
    "https://aclanthology.org/2020.lrec-1.521/",
    "Wiktionary contributors, https://en.wiktionary.org/",
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


def _wikipron_source(filename: str, directory: str, digest: str) -> SourceFile:
    return SourceFile(
        filename,
        f"{WIKIPRON_ROOT}/data/{directory}/{filename}",
        digest,
        WIKIPRON_COMMIT,
        "CC-BY-SA-4.0 (Wiktionary terms; GFDL alternative)",
        WIKIPRON_LICENSE_URL,
        WIKIPRON_CITATIONS,
    )


_WIKIPRON_FILES = {
    "italian": _wikipron_source(
        "ita_latn_broad_filtered.tsv",
        "scrape/tsv",
        "231c78d1fb89f1f03ad7420b69003023b84863d877ce82ee72d2590fa470b78a",
    ),
    "french": _wikipron_source(
        "fra_latn_broad_filtered.tsv",
        "scrape/tsv",
        "6f0fe8d7a50f4eb494eb478e5673cb72542484dccd048f46f3b12f75f4d51628",
    ),
}
_WIKIPRON_WHITELISTS = {
    "italian": _wikipron_source(
        "ita_broad.phones",
        "phones/phones",
        "d05afac80a5873a329d3e314f5c3c2fb780e211e68bd65a2f4b1e3268b60f71b",
    ),
    "french": _wikipron_source(
        "fra_broad.phones",
        "phones/phones",
        "2795d312b417e2181b5c38cdf0541e8af0d5bb00209e2328acdf95c23fb5e66e",
    ),
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


def _parse_tsv(path: Path) -> list[Pair]:
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
        raise ValueError(f"{path.name}: empty TSV lexicon")
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


def _spelling_group(word: str) -> str:
    """Keep NFC-casefold aliases together without altering model input."""
    return unicodedata.normalize("NFC", unicodedata.normalize("NFC", word).casefold())


def _load_wikipron(name: str, cache: Path) -> PreparedDataset:
    source = _WIKIPRON_FILES[name]
    whitelist = _WIKIPRON_WHITELISTS[name]
    raw = _parse_tsv(_fetch(source, cache))
    allowed = {
        token
        for line in _fetch(whitelist, cache).read_text(encoding="utf-8").splitlines()
        if (token := line.split("#", 1)[0].strip())
    }
    if not allowed:
        raise ValueError("WikiPron phone whitelist is empty")
    unknown = {phone for _, phones in raw for phone in phones} - allowed
    if unknown:
        raise ValueError(f"WikiPron phones outside pinned whitelist: {sorted(unknown)}")
    pairs = [
        (unicodedata.normalize("NFC", word), phones)
        for word, phones in raw
        if name != "french" or "‿" not in phones
    ]
    original_words = {unicodedata.normalize("NFC", word) for word, _ in raw}
    retained_words = {word for word, _ in pairs}
    test, remaining = split_lexicon_by_key(
        pairs, key=_spelling_group, seed=1729, test_fraction=0.1, max_test=10000
    )
    dev, train = split_lexicon_by_key(
        remaining,
        key=_spelling_group,
        seed=1729,
        test_fraction=0.1,
        max_test=len(remaining),
    )
    splits = {"train": train, "dev": dev, "test": test}
    groups = {
        split: {_spelling_group(word) for word, _ in rows}
        for split, rows in splits.items()
    }
    if any(
        groups[a] & groups[b]
        for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))
    ):
        raise ValueError("WikiPron casefold spelling overlap between splits")
    return _assemble(
        name,
        splits,
        {
            "locale": "fr_FR" if name == "french" else "it_IT",
            "phoneset": "ipa",
            "sources": {
                "lexicon": source.to_dict(),
                "phone_whitelist": whitelist.to_dict(),
            },
            "preparation": {
                "spelling": "NFC; case preserved",
                "phones": "original tokens",
                "remove_stress": False,
                "excluded_variant_tokens": ["‿"] if name == "french" else [],
            },
            "quality": {
                "upstream_filter": "phone-token whitelist; not a correctness guarantee",
                "upstream_settings_url": f"{WIKIPRON_ROOT}/data/scrape/lib/scrape.py",
                "upstream_settings": {"stress": False, "syllable_boundaries": False},
                "raw_entries": len(raw),
                "raw_words": len(original_words),
                "excluded_annotated_entries": len(raw) - len(pairs),
                "excluded_only_annotated_words": len(original_words - retained_words),
                "retained_entries_before_dedup": len(pairs),
                "retained_words_before_dedup": len(retained_words),
                "exclusion_reason": "whole linking-annotated variants excluded from segment-only target"
                if name == "french"
                else "none",
            },
            "split": {
                "method": "complete NFC-casefold spelling groups; original case retained",
                "seed": 1729,
                "test_fraction": 0.1,
                "max_test_words": 10000,
                "dev_fraction_of_remaining_groups": 0.1,
                "rounding": "floor; minimum one held-out group per split; reject empty train",
                "groups": {split: len(keys) for split, keys in groups.items()},
                "casefold_overlap": {"train_dev": 0, "train_test": 0, "dev_test": 0},
            },
        },
    )


def load_dataset(
    name: str, cache_dir: str | Path, remove_stress: bool = False
) -> PreparedDataset:
    """Load verified CMUdict or full filtered WikiPron French/Italian lexicons.

    WikiPron preserves NFC spelling/case and original segmented IPA tokens.
    Entire French variants bearing the linking annotation ``‿`` are excluded;
    clean alternatives remain. NFC-casefold aliases share a split while retaining
    their original spelling. Its upstream phone whitelist is checked, but does
    not establish lexicon correctness. Additional stress stripping is CMU-only.
    All datasets use seed1729: test10% capped at10000 spelling groups, then
    dev10% of remaining groups. CMUdict retains its existing lowercase grouping.
    Stable pair dedup follows transformations; no model-specific cooking applies.
    Invalid cached bytes fail closed. Raw data is not bundled or relicensed.
    """
    if name not in {"cmudict", *_WIKIPRON_FILES}:
        raise ValueError("Unknown dataset; choose cmudict, french or italian")
    if not isinstance(remove_stress, bool):
        raise ValueError("remove_stress must be a boolean")
    cache = Path(cache_dir) / name
    if name in _WIKIPRON_FILES:
        if remove_stress:
            raise ValueError(
                "IPA phone tokens are preserved; remove_stress applies only to cmudict"
            )
        return _load_wikipron(name, cache)
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
