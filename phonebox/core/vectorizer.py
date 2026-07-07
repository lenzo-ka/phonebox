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

# Per-phoneset stress markers; applied only when ``remove_stress=True``.
# Add more known phonesets here as needed. Unknown phonesets get no-op
# stress stripping (the phoneset is just a tag, see DEFAULT_PHONESET).
_STRESS_STRIPPERS: dict[str, re.Pattern[str]] = {
    "cmu": re.compile(r"[012]"),
    "xsampa": re.compile(r'["%]'),
}

logger = get_logger(__name__)

try:
    from ..utils.icu_utils import RuleTransliterator

    HAS_ICU = True
except ImportError:
    HAS_ICU = False


def make_join_re(join_list: list[str]) -> re.Pattern[str] | None:
    """Compile a regex that matches any space-separated joining sequence.

    Returns None when the list is empty so callers can short-circuit.
    """
    if not join_list:
        return None
    items = sorted(join_list, key=len, reverse=True)
    items = [re.escape(x) for x in items]
    # The trailing boundary is a lookahead so it is not consumed: adjacent
    # join groups sharing a delimiter space (``"a b c d"`` with ``"a b"`` and
    # ``"c d"``) both match in a single left-to-right ``sub`` pass.
    return re.compile(r" (" + r"|".join(items) + r")(?= )")


def join_seq(
    regex: re.Pattern[str] | None, seq: list[str], join_char: str
) -> list[str]:
    """Collapse adjacent tokens in *seq* matched by *regex* using *join_char*."""
    if not regex:
        return list(seq)
    as_str = " " + " ".join(seq) + " "
    joined = regex.sub(lambda m: " " + m.group(1).replace(" ", join_char), as_str)
    return joined.split()


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
        self.target_position = target_position  # "first" or "last"

        if target_position not in ["first", "last"]:
            raise ValueError(
                f"target_position must be 'first' or 'last', not {target_position}"
            )

        if self.phoneset_name is None:
            self.phoneset_name = DEFAULT_PHONESET
        if not self.width % 2:
            raise ValueError(f"width must be odd, not {self.width}")
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
        self.setup_locale(locale)

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

        # French marks liaison context with a padding symbol so the tree can
        # condition on a following word boundary.
        if self.locale.startswith("fr"):  # type: ignore[attr-defined]
            self.liaison_pad = "#"

        locale_dir = self._resolve_locale_dir()
        if not locale_dir:
            return

        self._load_transliterators(locale_dir)
        self._load_config(locale_dir)

    def _resolve_locale_dir(self) -> Path | None:
        """Find locale config directory, falling back to default if needed."""
        module_dir = Path(__file__).parent.parent
        data_dir = module_dir / "config" / "locales"
        locale_dir = data_dir / self.locale
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
        return config

    @staticmethod
    def canonical_locale_for(locale):
        if len(locale) == 5:
            lang = locale[:2]
            region = locale[-2:]
            locale = lang.lower() + "_" + region.upper()
        return locale

    def vectorize_word(self, word: str) -> list[list[str]]:
        """Returns letter vectors for one line (word)"""
        letters = list(word)
        if self.liaison_pad:
            letters.append(self.liaison_pad)

        return list(self.next_letter_vector(letters, g2p=True))

    def next_letter_vector(self, letters: str | list[str], g2p: bool = False):
        """Yield one padded context window (width letters) per cooked letter."""
        letters = self.cook_letters(letters, g2p=g2p)
        lett_count = len(letters)  # Use cooked length
        lets = self.padding + letters + self.padding  # Use cooked letters

        for n in range(lett_count):
            yield lets[n : n + self.width]

    def next_vector(self, letters: list[str], phones: list[str]):
        """Iterator for output letter/phone vectors"""
        for n, vec in enumerate(self.next_letter_vector(letters, g2p=True)):
            if self.target_position == "first":
                yield [phones[n]] + vec
            else:  # "last"
                yield vec + [phones[n]]

    def letters_and_phones(
        self, line: str, letters_spaced: bool = False, phones_cooked: bool = False
    ) -> tuple[list[str] | None, list[str] | None]:
        """Get lists of letters and phones from dictionary line.

        Args:
            line: Dictionary or alignment line
            letters_spaced: If True, letters are space-separated (alignment format)
            phones_cooked: If True, skip cooking phones (already cooked in alignment files)
        """

        line = ud.normalize("NFC", line.strip())

        # Remove trailing comment if present. Skip this in alignment-line
        # mode (letters_spaced=True) because fr_FR's liaison_pad IS the
        # literal '#' character and appears as a letter token in EM
        # alignments (e.g. "o e u f #\tœ ε f ε"); stripping at '#' there
        # would drop the tab and lose the phone side entirely, returning
        # (None, None) and crashing load_alignments downstream.
        if not letters_spaced and "#" in line:
            line = line.split("#")[0].strip()

        # Skip empty lines after comment removal
        if not line:
            return None, None

        # Handle both tab and space separated formats
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

        # Strip CMUdict-style variant suffix `(N)` from the word the same way
        # `dictionary.parse_dict_line` does. The xlit's `[^-.'[:L:]] Remove`
        # rule normally drops these parens+digits as a side effect, but only
        # when PyICU is available; without that, lexicon entries like
        # `œufs(2)` leak `(`, `2`, `)` into training letters and exception
        # keys. This makes the behavior consistent regardless of ICU.
        if not letters_spaced:
            letters_str = re.sub(r"\(\d+\)$", "", letters_str)

        letters_list = letters_str.split(" ") if letters_spaced else list(letters_str)
        letters_list = self.cook_letters(letters_list, g2p=True)

        phones_list = phones_str.split()
        if not phones_cooked:
            phones_list = self.cook_phones(phones_list)

        return letters_list, phones_list

    def next_alignment_vector(self, alignment: str):
        """Accumulate counts and return next alignment vector"""
        # Alignment files have pre-cooked phones (already aligned 1:1 with letters)
        letters, phones = self.letters_and_phones(
            alignment, letters_spaced=True, phones_cooked=True
        )
        for L in letters:
            self.counts["letters"][L] += 1
        for P in phones:
            self.counts["phones"][P] += 1

        for vector in self.next_vector(letters, phones):
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
            stripper = _STRESS_STRIPPERS.get(self.phoneset_name)
            if stripper is not None:
                cooked = [stripper.sub("", p) for p in cooked]

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
        if outfile:  # header
            print(*self.default_cols, file=outfile)
        elif header:
            out.append(" ".join(str(c) for c in self.default_cols))

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
                and not line.startswith("0 1")
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
