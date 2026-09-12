#!/usr/bin/env python

from __future__ import annotations

import json
import re
import unicodedata as ud
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

from phonebox.utils.logging_config import get_logger

from ..constants import (
    AETHER,
    CONTEXT_WINDOW_SIZE,
    DEFAULT_CASED,
    DEFAULT_LOCALE,
    DEFAULT_PHONESET,
    EPSILON,
    FILE_ENCODING,
    JOIN_CHAR,
)
from ..lexicon import parse_dict_line, strip_phone_stress
from ..locale_resolution import (
    canonical_locale,
    orthographic_compatibility,
    resolve_locale,
)
from ..portable_normalization import join_seq, make_join_re
from .legacy_preprocessing import known_legacy_g2p_rules

logger = get_logger(__name__)

LETTER_PREPROCESSING_VERSION = 1

try:
    from ..utils.icu_utils import RuleTransliterator

    HAS_ICU = True
except ImportError:
    HAS_ICU = False


class Vectorizer:
    """Turn a letter + phone sequence into a sequence of observation vectors."""

    def __init__(
        self,
        locale: str | None = None,
        phoneset_name: str | None = None,
        remove_stress: bool = False,
        remove_accents: bool = False,
        filter_non_letters: bool = False,
        cased: bool = DEFAULT_CASED,
        width: int = CONTEXT_WINDOW_SIZE,
        aether: str = AETHER,
        epsilon: str = EPSILON,
        join_char: str = JOIN_CHAR,
        verbose: bool = False,
        norm_xlit: bool = False,
        spelling_rewrites: dict[str, str] | None = None,
        letter_preprocessing: dict | None = None,
        target_position: str = "last",
    ) -> None:
        """Configure how letter/phone sequences become observation vectors.

        Args:
            locale: Locale tag whose ``config.json`` supplies joinings and
                transliteration rules (None for per-character, config-free use).
            phoneset_name: Selects stress-stripping rules and the locale join
                key; defaults to ``DEFAULT_PHONESET``.
            remove_stress: Strip phoneset stress markers from phones.
            remove_accents: Fold accents off Latin letters.
            filter_non_letters: Drop non-letter characters (except ' - .).
            cased: Keep letter case (default lowercases everything).
            width: Context window size; must be odd (center letter + equal
                left/right context).
            aether: Padding symbol for positions off the ends of the word.
            epsilon: Symbol emitted for a silent letter (no phone).
            join_char: Glues multi-unit graphemes/phones into one token.
            verbose: Log transliteration traces.
            norm_xlit: Apply the locale's normalization transliterator.
            target_position: ``"last"`` or ``"first"`` — where the predicted
                column sits in each emitted vector.
        """
        self.locale = None
        self.policy_locale: str | None = None
        self.phoneset_name = phoneset_name
        self.remove_stress = remove_stress
        self.remove_accents = remove_accents
        self.filter_non_letters = filter_non_letters
        self.cased = cased
        self.width = width

        self.aether = aether
        self.epsilon = epsilon
        self.join_char = join_char
        self.verbose = verbose
        self.liaison_pad: str | None = None
        self.norm_xlit = norm_xlit
        self.spelling_rewrites = dict(spelling_rewrites or {})
        self.letter_preprocessing: dict | None = None
        self.target_position = target_position  # "first" or "last"

        if target_position not in ["first", "last"]:
            raise ValueError(
                f"target_position must be 'first' or 'last', not {target_position}"
            )

        if self.phoneset_name is None:
            self.phoneset_name = DEFAULT_PHONESET
        if self.width <= 0 or not self.width % 2:
            raise ValueError(f"width must be positive and odd, not {self.width}")
        self._pad = int((self.width - 1) / 2)
        self.padding = [self.aether] * self._pad

        self.counts: dict[str, defaultdict[str, int]] = {
            "letters": defaultdict(int),
            "phones": defaultdict(int),
        }

        self.config: dict | None = None
        self.norm_transliterator: RuleTransliterator | None = None
        self.g2p_transliterator: RuleTransliterator | None = None
        self.lett_join_re: re.Pattern[str] | None = None
        self.phon_join_re: re.Pattern[str] | None = None
        if letter_preprocessing is None:
            self.setup_locale(locale)
            self.spelling_rewrites = self._validate_spelling_rewrites(
                self.spelling_rewrites
            )
        else:
            self.locale = self.canonical_locale_for(locale or DEFAULT_LOCALE)
            self.policy_locale = self.locale
            self.load_letter_preprocessing(letter_preprocessing)

    @property
    def default_cols(self) -> list[str]:
        """All column names including target column P, based on width."""
        cols = [str(i) for i in range(self.width)]
        if self.target_position == "first":
            return ["P"] + cols
        return cols + ["P"]

    def multigram_config(self) -> dict[str, int]:
        """Return the locale's optional ``multigram`` config section.

        Keys recognised so far: ``max_letter_span``, ``max_phone_span``.
        Returns an empty dict if the locale doesn't override defaults.
        """
        if not self.config:
            return {}
        section = self.config.get("multigram") or {}
        if not isinstance(section, dict):
            return {}
        return {k: int(v) for k, v in section.items() if isinstance(v, (int, float))}

    @property
    def feature_names(self) -> list[str]:
        """Feature column names (excludes target column P)."""
        return [str(i) for i in range(self.width)]

    def parse_vector(self, vector_str: str) -> tuple[list[str], str]:
        """Parse a vector string into (features, target).

        Args:
            vector_str: Space-separated vector string

        Returns:
            Tuple of (X_features, y_target)
        """
        parts = vector_str.strip().split()
        if self.target_position == "first":
            return parts[1:], parts[0]
        return parts[:-1], parts[-1]

    def setup_locale(self, locale: str | None) -> None:
        """Resolve *locale* and load its transliterators and join config."""
        self.locale = self.canonical_locale_for(locale or DEFAULT_LOCALE)
        # setup_locale may switch an existing Vectorizer (notably when loading
        # a snapshotless model). Clear every locale-owned value before loading
        # the target so missing rules or joins cannot leak from the old locale.
        self.policy_locale = None
        self.liaison_pad = None
        self.config = None
        self.norm_transliterator = None
        self.g2p_transliterator = None
        self.lett_join_re = None
        self.phon_join_re = None

        locale_dir = self._resolve_locale_dir()
        if not locale_dir:
            return

        # French marks liaison context with a padding symbol so the tree can
        # condition on a following word boundary. This belongs to the selected
        # policy, not an unresolved request that merely starts with ``fr``.
        if self.policy_locale and self.policy_locale.startswith("fr"):
            self.liaison_pad = "#"

        self._load_transliterators(locale_dir)
        self._load_config(locale_dir)

    def _resolve_locale_dir(self) -> Path | None:
        """Find locale config directory, falling back to default if needed."""
        module_dir = Path(__file__).parent.parent
        data_dir = module_dir / "config" / "locales"
        available = [path.name for path in data_dir.iterdir() if path.is_dir()]
        resolution = resolve_locale(
            self.locale, available, orthographic_compatibility()
        )
        resolved = resolution.resolved
        locale_dir = data_dir / resolved if resolved else data_dir / self.locale
        self.policy_locale = resolved or "default"
        default_dir = data_dir / "default"

        if locale_dir.exists():
            return locale_dir

        if default_dir.exists():
            logger.warning("=" * 80)
            logger.warning("Locale directory not found: %s", locale_dir)
            logger.warning("Falling back to default configuration at: %s", default_dir)
            logger.warning(
                "This may result in suboptimal g2p behavior for locale: %s", self.locale
            )
            logger.warning("=" * 80)
            return default_dir

        logger.error(
            "Neither locale directory '%s' nor default directory '%s' found!",
            locale_dir,
            default_dir,
        )
        return None

    def _load_transliterators(self, locale_dir: Path) -> None:
        """Load transliterator rules if available."""
        data_dir = locale_dir.parent

        if self.norm_xlit:
            norm_xlit = data_dir / "norm.xlit"
            if norm_xlit.exists():
                self.norm_transliterator = RuleTransliterator(path=str(norm_xlit))

        g2p_xlit = locale_dir / "g2p.xlit"
        if g2p_xlit.exists():
            self.g2p_transliterator = RuleTransliterator(path=str(g2p_xlit))

    def _load_config(self, locale_dir: Path) -> None:
        """Load locale config and setup joining regexes."""
        with open(locale_dir / "config.json", encoding=FILE_ENCODING) as infile:
            self.config = json.load(infile)

        joinings = self.config.get("join", {})
        if joinings.get("letters"):
            self.lett_join_re = self.make_re(joinings["letters"])
        if joinings.get(self.phoneset_name):
            self.phon_join_re = self.make_re(joinings[self.phoneset_name])

    def make_re(self, join_list: list[str]) -> re.Pattern[str]:
        return make_join_re(join_list)

    def disable_config_joins(self) -> None:
        """Disable locale ``config.json`` letter/phone joins (xlit still applies)."""
        self.lett_join_re = None
        self.phon_join_re = None

    def _use_legacy_locale_preprocessing(self) -> bool:
        """Restore known pre-snapshot rules for this saved model's locale."""
        rules = known_legacy_g2p_rules(self.locale)
        if rules is None:
            return False
        self.g2p_transliterator = RuleTransliterator(rules=rules)
        if not self.g2p_transliterator:
            raise ValueError("invalid packaged legacy g2p transliterator rules")
        return True

    def export_config(self):
        """Serialize the cook-affecting state into the config persisted with a
        model and reloaded by ``load_model`` (the phoneset join key is renamed
        to ``"phones"`` so load is phoneset-agnostic)."""
        if self.config is None:
            # Return a minimal config if none was loaded
            config = {"join": {"letters": [], self.phoneset_name: []}}
        else:
            config = deepcopy(self.config)
        joiners = config.pop("join", {"letters": [], self.phoneset_name: []})
        # Persist every flag that affects cook_letters/cook_phones so we can
        # round-trip the vectorizer state on load_model.
        config["remove_accents"] = self.remove_accents
        config["filter_non_letters"] = self.filter_non_letters
        config["locale"] = self.locale
        config["policy_locale"] = self.policy_locale
        config["phoneset_name"] = self.phoneset_name
        config["cased"] = self.cased
        config["remove_stress"] = self.remove_stress
        config["width"] = self.width
        config["aether"] = self.aether
        config["epsilon"] = self.epsilon
        config["join_char"] = self.join_char
        config["liaison_pad"] = self.liaison_pad
        config["padding"] = "".join(self.padding)
        config["counts"] = deepcopy(self.counts)
        for key in list(joiners):
            if key == "letters":
                continue
            value = joiners.pop(key)
            if key == self.phoneset_name:
                joiners["phones"] = value
        config["join"] = joiners
        config["letter_preprocessing"] = self.export_letter_preprocessing()
        return config

    def export_letter_preprocessing(self) -> dict:
        """Return an exact, versioned snapshot of raw-word preprocessing."""
        norm_rules = (
            self.norm_transliterator.rules
            if self.norm_transliterator is not None
            else None
        )
        g2p_rules = (
            self.g2p_transliterator.rules
            if self.g2p_transliterator is not None
            else None
        )

        joins: list[str] = []
        if self.lett_join_re is not None and self.config:
            joins = list((self.config.get("join") or {}).get("letters", []))
        return {
            "version": LETTER_PREPROCESSING_VERSION,
            "source": {"norm_rules": norm_rules, "g2p_rules": g2p_rules},
            "join_char": self.join_char,
            "letter_joins": joins,
            "cased": self.cased,
            "remove_accents": self.remove_accents,
            "filter_non_letters": self.filter_non_letters,
            "spelling_rewrites": dict(self.spelling_rewrites),
        }

    def _validate_spelling_rewrites(self, requested: dict[str, str]) -> dict[str, str]:
        """Validate every key with all post-cooking rewrites disabled."""
        previous = self.spelling_rewrites
        self.spelling_rewrites = {}
        try:
            for source, replacement in requested.items():
                if not isinstance(source, str) or not isinstance(replacement, str):
                    raise ValueError("spelling rewrites must map strings to strings")
                cooked = self.cook_letters(source, g2p=True)
                if len(cooked) != 1 or len(cooked[0]) != 1:
                    raise ValueError(
                        f"spelling rewrite source {source!r} does not cook to one character"
                    )
                if cooked[0] != source:
                    raise ValueError(
                        f"spelling rewrite source {source!r} changes during locale cooking; "
                        f"use the cooked character {cooked[0]!r}"
                    )
        finally:
            self.spelling_rewrites = previous
        return dict(requested)

    def load_letter_preprocessing(self, snapshot: dict) -> None:
        """Restore preprocessing from model metadata without locale lookup."""
        if snapshot.get("version") != LETTER_PREPROCESSING_VERSION:
            raise ValueError("unsupported letter preprocessing version")
        source = snapshot.get("source")
        if not isinstance(source, dict):
            raise ValueError("malformed letter preprocessing source")
        missing_source_keys = {"norm_rules", "g2p_rules"} - source.keys()
        if missing_source_keys:
            missing = ", ".join(sorted(missing_source_keys))
            raise ValueError(f"missing letter preprocessing source fields: {missing}")
        required = {
            "join_char": str,
            "letter_joins": list,
            "cased": bool,
            "remove_accents": bool,
            "filter_non_letters": bool,
            "spelling_rewrites": dict,
        }
        for key, expected_type in required.items():
            if not isinstance(snapshot.get(key), expected_type):
                raise ValueError(f"malformed letter preprocessing field: {key}")
        for key in ("norm_rules", "g2p_rules"):
            if source.get(key) is not None and not isinstance(source.get(key), str):
                raise ValueError(f"malformed letter preprocessing source: {key}")
        joins = snapshot["letter_joins"]
        if not all(isinstance(item, str) for item in joins):
            raise ValueError("malformed letter preprocessing field: letter_joins")
        rewrites = snapshot["spelling_rewrites"]
        if not all(
            isinstance(key, str) and len(key) == 1 and isinstance(value, str)
            for key, value in rewrites.items()
        ):
            raise ValueError("malformed letter preprocessing field: spelling_rewrites")
        from ..utils.icu_utils import HAS_ICU, RuleTransliterator

        norm_rules = source.get("norm_rules")
        g2p_rules = source.get("g2p_rules")
        if (norm_rules or g2p_rules) and not HAS_ICU:
            raise RuntimeError(
                "loading saved letter preprocessing requires the ICU runtime; "
                "install phonebox with its declared runtime dependencies"
            )
        norm_transliterator = (
            RuleTransliterator(rules=norm_rules) if norm_rules else None
        )
        g2p_transliterator = RuleTransliterator(rules=g2p_rules) if g2p_rules else None
        if norm_rules and not norm_transliterator:
            raise ValueError("invalid saved norm transliterator rules")
        if g2p_rules and not g2p_transliterator:
            raise ValueError("invalid saved g2p transliterator rules")
        previous_state = (
            self.norm_transliterator,
            self.g2p_transliterator,
            self.cased,
            self.remove_accents,
            self.filter_non_letters,
            self.spelling_rewrites,
            self.join_char,
            self.lett_join_re,
            self.config,
            self.letter_preprocessing,
        )
        try:
            self.norm_transliterator = norm_transliterator
            self.g2p_transliterator = g2p_transliterator
            self.cased = snapshot["cased"]
            self.remove_accents = snapshot["remove_accents"]
            self.filter_non_letters = snapshot["filter_non_letters"]
            self.join_char = snapshot["join_char"]
            self.lett_join_re = make_join_re(joins)
            saved_join_config = deepcopy((self.config or {}).get("join", {}))
            saved_join_config["letters"] = list(joins)
            self.config = {"join": saved_join_config}
            self.spelling_rewrites = self._validate_spelling_rewrites(
                dict(snapshot["spelling_rewrites"])
            )
            self.letter_preprocessing = deepcopy(snapshot)
        except Exception:
            (
                self.norm_transliterator,
                self.g2p_transliterator,
                self.cased,
                self.remove_accents,
                self.filter_non_letters,
                self.spelling_rewrites,
                self.join_char,
                self.lett_join_re,
                self.config,
                self.letter_preprocessing,
            ) = previous_state
            raise

    @staticmethod
    def canonical_locale_for(locale):
        return canonical_locale(locale)

    def vectorize_word(self, word: str) -> list[list[str]]:
        """Returns letter vectors for one line (word)"""
        letters = list(word)
        if self.liaison_pad:
            letters.append(self.liaison_pad)

        return list(self.next_letter_vector(letters, g2p=True))

    def next_letter_vector(
        self, letters: str | list[str], g2p: bool = False, cooked: bool = False
    ):
        """Yield one padded context window (width letters) per cooked letter."""
        letters = list(letters) if cooked else self.cook_letters(letters, g2p=g2p)
        lett_count = len(letters)  # Use cooked length
        lets = self.padding + letters + self.padding  # Use cooked letters

        for n in range(lett_count):
            yield lets[n : n + self.width]

    def next_vector(
        self, letters: list[str], phones: list[str], letters_cooked: bool = False
    ):
        """Iterator for output letter/phone vectors"""
        for n, vec in enumerate(
            self.next_letter_vector(letters, g2p=True, cooked=letters_cooked)
        ):
            if self.target_position == "first":
                yield [phones[n]] + vec
            else:  # "last"
                yield vec + [phones[n]]

    def letters_and_phones(
        self,
        line: str,
        letters_spaced: bool = False,
        phones_cooked: bool = False,
        letters_cooked: bool = False,
    ) -> tuple[list[str] | None, list[str] | None]:
        """Get lists of letters and phones from dictionary line.

        Args:
            line: Dictionary or alignment line
            letters_spaced: If True, letters are space-separated (alignment format)
            phones_cooked: If True, skip cooking phones (already cooked in alignment files)
            letters_cooked: If True, skip cooking letters (already cooked in alignments)
        """

        line = ud.normalize("NFC", line.strip())

        if not line:
            return None, None

        # Dictionary lines use the shared parser. Alignment lines retain their
        # separate grammar because a literal '#' may be a French liaison token.
        if not letters_spaced:
            parsed = parse_dict_line(line)
            if parsed is None:
                return None, None
            letters_str, phones = parsed
            letters = (
                list(letters_str)
                if letters_cooked
                else self.cook_letters(list(letters_str), g2p=True)
            )
            cooked = phones if phones_cooked else self.cook_phones(phones)
            return letters, cooked
        if "\t" in line:
            parts = line.split("\t")
            if len(parts) < 2:
                return None, None
            letters_str, phones_str, *_ = parts
        else:
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                return None, None
            letters_str, phones_str = parts

        letters_list = letters_str.split(" ") if letters_spaced else list(letters_str)
        if not letters_cooked:
            letters_list = self.cook_letters(letters_list, g2p=True)

        phones_list = phones_str.split()
        if not phones_cooked:
            phones_list = self.cook_phones(phones_list)

        return letters_list, phones_list

    def next_alignment_vector(self, alignment: str):
        """Accumulate counts and return next alignment vector"""
        # Alignment files have pre-cooked phones (already aligned 1:1 with letters)
        letters, phones = self.letters_and_phones(
            alignment, letters_spaced=True, phones_cooked=True, letters_cooked=True
        )
        for L in letters:
            self.counts["letters"][L] += 1
        for P in phones:
            self.counts["phones"][P] += 1

        for vector in self.next_vector(letters, phones, letters_cooked=True):
            yield " ".join(vector)

    def cook_letters(self, letters: str | list[str], g2p: bool = False) -> list[str]:
        """
        Canonicalize letter sequence with joinings.
        If g2p is True, does additional normalization
        that shouldn't be done when doing dictionary lookup.
        """
        if self.join_char:
            # On the letter side join_char glues graphemes into one token, so
            # strip it (cook_phones instead turns it into a separating space).
            cooked = [p.replace(self.join_char, "") for p in letters]
        else:
            cooked = list(letters)
        if not self.cased:
            cooked = [letter.lower() for letter in cooked]

        after = orig = "".join(cooked)

        # Remove accents if requested (orthographic normalization)
        if self.remove_accents:
            after = self._remove_accents(after)

        # Filter non-letters if requested (keep letters, hyphens, apostrophes)
        if self.filter_non_letters:
            after = self._filter_non_letters(after)

        if self.norm_transliterator:
            after = self.norm_transliterator.translit(after)
        if g2p and self.g2p_transliterator:
            after = self.g2p_transliterator.translit(after)
        if g2p and self.spelling_rewrites:
            after = "".join(self.spelling_rewrites.get(char, char) for char in after)
        if self.verbose and after != orig:
            logger.debug("xlit: %s %s", orig, after)
        cooked = list(after)

        return self.join_seq(self.lett_join_re, cooked)

    @staticmethod
    def _remove_accents(text: str) -> str:
        """
        Remove accents from Latin letters.

        Uses Unicode normalization:
        1. NFD: Decompose (café → c+a+f+e+́)
        2. Remove combining marks (category Mn)
        3. NFC: Recompose

        Examples:
            café → cafe
            naïve → naive
            résumé → resume

        Args:
            text: Input text

        Returns:
            Text with accents removed
        """
        # Decompose to separate base letters from combining marks
        nfd = ud.normalize("NFD", text)

        # Remove combining marks (Mn = Nonspacing_Mark category)
        without_marks = "".join(char for char in nfd if ud.category(char) != "Mn")

        # Recompose to canonical form
        return ud.normalize("NFC", without_marks)

    @staticmethod
    def _filter_non_letters(text: str) -> str:
        """
        Keep only letters, hyphens, apostrophes, and periods.

        Works in NFD space so accented letters are preserved properly.
        Combining marks are kept with their base letters.

        Examples:
            test3 → test
            word(2) → word
            don't → don't
            café → café

        Args:
            text: Input text

        Returns:
            Text with non-letter characters removed (except -, ', .)
        """
        # Work in NFD so combining marks stay with base letters
        nfd = ud.normalize("NFD", text)
        # Keep letters, combining marks (Mn/Mc), and word punctuation
        filtered = "".join(
            c
            for c in nfd
            if c.isalpha() or ud.category(c) in ("Mn", "Mc") or c in "-'."
        )
        return ud.normalize("NFC", filtered)

    def cook_phones(self, phones: list[str]) -> list[str]:
        """
        Canonicalize phone list but joining phones
        and optionally remove stress
        """
        if self.join_char:
            cooked = [p.replace(self.join_char, " ") for p in phones]
        else:
            cooked = list(phones)
        if self.remove_stress:
            # Strip phoneset-specific stress markers; unknown phonesets pass
            # through unchanged (the caller asked for stripping but we have
            # no rule for this phoneset).
            cooked = [strip_phone_stress(p, self.phoneset_name) for p in cooked]

        return self.join_seq(self.phon_join_re, cooked)

    def join_seq(self, regex, seq: list[str]) -> list[str]:
        """Combine joiners in list and return updated list"""
        return join_seq(regex, seq, self.join_char)

    def uncook(self, cooked: list[str]) -> list[str]:
        """Split joined phones. Does not restore stress."""
        raw = []
        for token in cooked:
            if token == self.epsilon:
                continue
            raw.extend(token.split(self.join_char))

        return raw

    def vectorize_file(self, infile, outfile=None, header: bool = False) -> list[str]:
        """Get or write all the vectors from alignments"""
        out: list[str] = []
        if header:
            if outfile:
                print(*self.default_cols, file=outfile)
            else:
                out.append(" ".join(self.default_cols))

        for line in infile:
            for vector in self.next_alignment_vector(line):
                if outfile:
                    print(vector, file=outfile)
                else:
                    out.append(vector)

        return out

    def load_vectors_file(self, path: str):
        """Load vectors from file into a count dict.

        Args:
            path: Path to vectors file

        Returns:
            Counter mapping vector string to count
        """
        with open(path, encoding=FILE_ENCODING) as f:
            return Counter(
                line.strip()
                for line in f
                if line.strip()
                and not line.startswith("#")
                and line.split() != self.default_cols
            )

    def parse_vectors_to_data(
        self, vectors_dict
    ) -> tuple[list[list[str]], list[str], list[int]]:
        """Parse vectors dict into X, y, counts for training.

        Args:
            vectors_dict: Dict mapping vector string to count

        Returns:
            Tuple of (X, y, counts) lists
        """
        X = []
        y = []
        counts = []

        for vector, count in vectors_dict.items():
            X_val, y_val = self.parse_vector(vector)
            X.append(X_val)
            y.append(y_val)
            counts.append(count)

        return X, y, counts
