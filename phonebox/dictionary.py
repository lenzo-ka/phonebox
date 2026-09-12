#!/usr/bin/env python
"""
Dictionary class for managing pronunciation dictionaries.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections import defaultdict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TextIO

from .constants import (
    DEFAULT_TRAIN_PHONESET,
    DEFAULT_TRAIN_PRUNE,
    DEFAULT_TRAIN_REMOVE_STRESS,
    DEFAULT_TRAIN_TEST_SPLIT,
    DEFAULT_TRAIN_VALIDATION_SPLIT,
    DICT_ENCODING,
    DOWNLOAD_TIMEOUT_SECONDS,
)
from .core.decision_tree import DecisionTree
from .lexicon import parse_dict_line, strip_phone_stress
from .utils.io import paths_refer_to_same_file
from .utils.logging_config import get_logger

logger = get_logger(__name__)


# Download and upstream-notice links share the same public source revision.
CMUDICT_PROJECT = "cmusphinx/cmudict"
CMUDICT_REVISION = "master"
CMUDICT_REPOSITORY = f"https://github.com/{CMUDICT_PROJECT}"
CMUDICT_REPO = f"https://raw.githubusercontent.com/{CMUDICT_PROJECT}/{CMUDICT_REVISION}"
CMUDICT_LICENSE_URL = f"{CMUDICT_REPOSITORY}/blob/{CMUDICT_REVISION}/LICENSE"


def strip_stress(phoneme: str) -> str:
    """Remove stress markers from a phoneme."""
    return strip_phone_stress(phoneme, "cmu")


def phone_mapping_transform(
    mapping: Mapping[str, str | list[str]],
) -> Callable[[list[str]], list[str]]:
    """Build a validated literal phone-to-phone-sequence transform."""
    cooked: dict[str, list[str]] = {}
    for source, target in mapping.items():
        if not isinstance(source, str) or not source:
            raise ValueError("phone mapping keys must be nonempty strings")
        values = [target] if isinstance(target, str) else target
        if (
            not isinstance(values, list)
            or not values
            or not all(isinstance(phone, str) and phone for phone in values)
        ):
            raise ValueError(
                "phone mapping values must be nonempty strings or lists of them"
            )
        cooked[source] = values

    def transform(phones: list[str]) -> list[str]:
        return [mapped for phone in phones for mapped in cooked.get(phone, [phone])]

    return transform


class Dictionary:
    """
    Pronunciation dictionary with processing and training capabilities.

    Examples:
        # Load existing dictionary
        dict = Dictionary('data/cmudict/cmudict.dict', dict_format='cmudict')

        # Process it
        dict.process(remove_stress=True, output='processed.dict')

        # Train g2p model
        model = dict.train_g2p_model('en_US')

        # Or all in one
        dict = Dictionary.fetch('cmudict')
        model = dict.process(remove_stress=True).train_g2p_model('en_US')
    """

    def __init__(
        self,
        path: str | Path | None = None,
        dict_format: str = "auto",
        locale: str = "en_US",
    ):
        """
        Initialize a Dictionary.

        Args:
            path: Path to dictionary file (None for empty dictionary).
                Accepts ``str`` or ``pathlib.Path``; coerced internally.
            dict_format: Dictionary format ('cmudict', 'auto')
            locale: Language locale (e.g., 'en_US', 'en_IN')
        """
        self.path = Path(path) if path else None
        self.dict_format = dict_format
        self.locale = locale
        self._entries: list[tuple[str, list[str]]] = []
        self._loaded = False

    def load(self) -> Dictionary:
        """
        Load dictionary from file.

        Returns:
            Self for chaining
        """
        if not self.path or not self.path.exists():
            raise FileNotFoundError(f"Dictionary file not found: {self.path}")

        self._entries = []
        with open(self.path, encoding=DICT_ENCODING) as f:
            for line in f:
                result = parse_dict_line(line)
                if result:
                    word, phones = result
                    self._entries.append((word, phones))

        self._loaded = True
        return self

    @property
    def entries(self) -> list[tuple[str, list[str]]]:
        """Get dictionary entries as list of (word, phonemes) tuples."""
        if not self._loaded and self.path:
            self.load()
        return self._entries

    @property
    def size(self) -> int:
        """Number of entries in dictionary."""
        return len(self.entries)

    def _normalize_dict(
        self,
        infile: TextIO,
        outfile: TextIO,
        lowercase: bool,
        remove_stress: bool,
        deduplicate: bool,
        sort_output: bool,
        phone_transform: Callable[[list[str]], list[str]] | None,
        phoneset: str,
    ) -> int:
        """Internal method to normalize dictionary."""
        word_pronunciations: dict[str, list[str]] = defaultdict(list)

        for line in infile:
            result = parse_dict_line(line)
            if result is None:
                continue

            base_word, phonemes = result

            if lowercase:
                base_word = base_word.lower()

            if phone_transform is not None:
                phonemes = phone_transform(phonemes)
                if (
                    not isinstance(phonemes, list)
                    or not phonemes
                    or not all(isinstance(phone, str) and phone for phone in phonemes)
                ):
                    raise TypeError(
                        "phone_transform must return a list of nonempty strings"
                    )
            if remove_stress:
                phonemes = [strip_phone_stress(p, phoneset) for p in phonemes]

            pronunciation = " ".join(phonemes)

            if not deduplicate or pronunciation not in word_pronunciations[base_word]:
                word_pronunciations[base_word].append(pronunciation)

        entries_written = 0
        words = sorted(word_pronunciations) if sort_output else word_pronunciations

        for base_word in words:
            pronunciations = word_pronunciations[base_word]

            for idx, pronunciation in enumerate(pronunciations):
                word_variant = base_word if idx == 0 else f"{base_word}({idx + 1})"
                print(f"{word_variant} {pronunciation}", file=outfile)
                entries_written += 1

        return entries_written

    def process(
        self,
        remove_stress: bool = False,
        lowercase: bool = False,
        deduplicate: bool = True,
        sort_output: bool = True,
        output: str | Path | None = None,
        phone_transform: Callable[[list[str]], list[str]] | None = None,
        phone_mapping: Mapping[str, str | list[str]] | None = None,
        phoneset: str = "cmu",
    ) -> Dictionary:
        """
        Process dictionary with various transformations.

        Args:
            remove_stress: Remove stress markers from phonemes
            lowercase: Convert words to lowercase
            deduplicate: Remove duplicate pronunciations
            sort_output: Sort output alphabetically
            output: Output file path (default: create temp file)
            phone_transform: Optional pronunciation mapping applied before
                deduplication and dense variant numbering.
            phone_mapping: Literal phone replacements. Each value is one phone
                or a nonempty list of phones. Mutually exclusive with
                ``phone_transform``.
            phoneset: Phoneset tag selecting stress syntax. Unknown tags preserve
                phone tokens when ``remove_stress`` is enabled.

        Returns:
            New Dictionary instance pointing to processed file
        """
        if not self.path:
            raise ValueError("Cannot process dictionary without input file")
        if phone_transform is not None and phone_mapping is not None:
            raise ValueError("choose phone_transform or phone_mapping, not both")
        if phone_mapping is not None:
            phone_transform = phone_mapping_transform(phone_mapping)

        if output is None:
            suffix = "_nostress" if remove_stress else "_processed"
            output = self.path.parent / f"{self.path.stem}{suffix}{self.path.suffix}"

        output = Path(output)
        if paths_refer_to_same_file(self.path, output):
            raise ValueError("input and output dictionary must be different files")

        with (
            open(self.path, encoding=DICT_ENCODING) as infile,
            open(output, "w", encoding=DICT_ENCODING) as outfile,
        ):
            self._normalize_dict(
                infile,
                outfile,
                lowercase=lowercase,
                remove_stress=remove_stress,
                deduplicate=deduplicate,
                sort_output=sort_output,
                phone_transform=phone_transform,
                phoneset=phoneset,
            )

        return Dictionary(path=output, dict_format=self.dict_format, locale=self.locale)

    def train_g2p_model(
        self,
        locale: str | None = None,
        output: str | Path | None = None,
        phoneset: str = DEFAULT_TRAIN_PHONESET,
        remove_stress: bool = DEFAULT_TRAIN_REMOVE_STRESS,
        config: str | None = None,
        prune: bool = DEFAULT_TRAIN_PRUNE,
        validation_split: float = DEFAULT_TRAIN_VALIDATION_SPLIT,
        test_split: float = DEFAULT_TRAIN_TEST_SPLIT,
        alignments_out: str | Path | None = None,
        **kwargs,
    ) -> DecisionTree:
        """
        Train a g2p model from this dictionary.

        Args:
            locale: Locale for the model (default: use dictionary locale)
            output: Where to save the model (None = don't save)
            phoneset: Phoneset name ('cmu' or 'xsampa')
            remove_stress: Remove stress markers during training
            config: Optional config file path (overrides other args)
            prune: Post-prune the trained tree on the validation split
            validation_split: Fraction held out for pruning (e.g. 0.05)
            test_split: Fraction held out for held-out test evaluation
            **kwargs: Additional arguments for DecisionTree

        Returns:
            Trained DecisionTree model
        """
        # Load config if provided (overrides function args)
        # Training-call params (prune/validation_split/test_split) are kept
        # separate from DecisionTree(...) constructor kwargs so they reach
        # train_from_dict instead of blowing up __init__.
        _train_call_keys = {"prune", "validation_split", "test_split"}
        if config:
            from .config_loader import load_config

            config_dict = load_config(config)
            locale = config_dict.get("locale", locale)
            phoneset = config_dict.get("phoneset", phoneset)
            remove_stress = config_dict.get("remove_stress", remove_stress)
            output = config_dict.get("output", output)
            alignments_out = config_dict.get("alignments_out", alignments_out)
            prune = config_dict.get("prune", prune)
            validation_split = config_dict.get("validation_split", validation_split)
            test_split = config_dict.get("test_split", test_split)

            # Merge config into kwargs, but skip params already passed explicitly
            _handled = {
                "locale",
                "phoneset",
                "remove_stress",
                "output",
                "alignments_out",
                "dictionary",
            } | _train_call_keys
            for key, value in config_dict.items():
                if key not in _handled and key not in kwargs:
                    kwargs[key] = value
        if not self.path:
            raise ValueError("Cannot train from dictionary without file")

        locale = locale or self.locale

        from .training import train_g2p

        result = train_g2p(
            self.path,
            locale=locale,
            phoneset=phoneset,
            remove_stress=remove_stress,
            output=output,
            alignments_out=alignments_out,
            validation_split=validation_split,
            test_split=test_split,
            prune=prune,
            **kwargs,
        )
        return result.model

    @staticmethod
    def _download_file(url: str, output_path: Path) -> bool:
        """Download a file from URL. Returns True on success, False on failure."""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(
                url, timeout=DOWNLOAD_TIMEOUT_SECONDS
            ) as response:
                content = response.read()
            with open(output_path, "wb") as f:
                f.write(content)
            return True
        except (urllib.error.URLError, OSError) as e:
            logger.error("Error downloading %s: %s", url, e)
            return False

    @staticmethod
    def create_manifest(data_dir: Path, verbose: bool = False) -> None:
        """Create manifest file documenting downloaded CMUdict dictionary."""
        manifest: dict[str, list[dict[str, Any]]] = {"dictionaries": [], "sources": []}

        cmudict_dir = data_dir / "cmudict"
        if cmudict_dir.exists() and (cmudict_dir / "cmudict.dict").exists():
            source: dict[str, Any] = {
                "license": "CMUdict license",
                "license_url": CMUDICT_LICENSE_URL,
                "name": "CMUdict",
                "path": "cmudict/",
                "repository": CMUDICT_REPOSITORY,
            }
            if (cmudict_dir / "LICENSE").is_file():
                source["license"] = "CMUdict license (see LICENSE)"
                source["license_file"] = "cmudict/LICENSE"
            manifest["sources"].append(source)
            manifest["dictionaries"].append(
                {
                    "file": "cmudict/cmudict.dict",
                    "has_stress": True,
                    "language": "en_US",
                    "phoneset": "arpabet",
                    "source": "cmudict",
                }
            )

        manifest_path = data_dir / "manifest.json"
        with open(manifest_path, "w", encoding=DICT_ENCODING) as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        if verbose:
            logger.info("Manifest created: %s", manifest_path)
            logger.info("  Sources: %d", len(manifest["sources"]))
            logger.info("  Dictionaries: %d", len(manifest["dictionaries"]))

    @staticmethod
    def print_status(data_dir: Path) -> None:
        """Print status of downloaded CMUdict dictionary."""
        print("\nCMUdict Status:")
        print("=" * 60)

        cmudict_path = data_dir / "cmudict" / "cmudict.dict"
        if cmudict_path.exists():
            size_mb = cmudict_path.stat().st_size / (1024 * 1024)
            print(f"[OK] CMUdict: {cmudict_path} ({size_mb:.2f} MB)")
        else:
            print("[--] CMUdict: Not downloaded")

        manifest_path = data_dir / "manifest.json"
        if manifest_path.exists():
            print(f"\n[OK] Manifest: {manifest_path}")
        else:
            print("\n[--] Manifest: Not created")

        print("=" * 60)

    @classmethod
    def fetch(
        cls,
        source: str,
        data_dir: Path = Path("data"),
        verbose: bool = False,
    ) -> Dictionary:
        """
        Fetch CMUdict and its required copyright/license notice.

        Downloads follow the upstream master branch. A failed dictionary or
        LICENSE download raises RuntimeError; optional support files may fail.

        Args:
            source: Source name (must be 'cmudict')
            data_dir: Base data directory
            verbose: Print progress

        Returns:
            Dictionary instance

        Examples:
            dict = Dictionary.fetch('cmudict')
        """
        data_dir = Path(data_dir)

        if source == "cmudict":
            cmudict_dir = data_dir / "cmudict"
            cmudict_dir.mkdir(parents=True, exist_ok=True)

            files = [
                "LICENSE",
                "README",
                "cmudict.dict",
                "cmudict.phones",
                "cmudict.symbols",
            ]
            for filename in files:
                url = f"{CMUDICT_REPO}/{filename}"
                output_path = cmudict_dir / filename
                ok = cls._download_file(url, output_path)
                if not ok and filename in {"LICENSE", "cmudict.dict"}:
                    raise RuntimeError(
                        f"Failed to download required CMUdict file from {url}"
                    )

            path = cmudict_dir / "cmudict.dict"
            locale = "en_US"
            dict_format = "cmudict"
        else:
            raise ValueError(f"Unknown source: {source}. Only 'cmudict' is supported.")

        return cls(path=path, dict_format=dict_format, locale=locale)

    def __repr__(self) -> str:
        status = f"loaded, {self.size} entries" if self._loaded else "not loaded"
        return f"Dictionary(path={self.path}, locale={self.locale}, {status})"

    def __len__(self) -> int:
        return self.size
