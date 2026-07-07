#!/usr/bin/env python
"""1:1 EM alignment of letters to phones for the CART (decision-tree) pipeline.

Each training word is aligned so every letter maps to exactly one phone (a
silent letter maps to the epsilon symbol); multi-phone letters are handled
upstream by the vectorizer's join step. EM iterates: enumerate the possible
silent-letter placements per word (``possible_alignments``), score them under
the current letter->phone model, pick the best (``_find_best_alignment``), then
re-estimate the model (``make_model``). This is the simpler 1:1 counterpart to
``MultigramAligner``, which learns many-to-many joint units instead.
"""

from __future__ import annotations

import math
import signal
import unicodedata as ud
from collections.abc import Iterator
from itertools import combinations
from multiprocessing import Pool, cpu_count
from time import time

from phonebox.utils.logging_config import get_logger

from ..constants import (
    CHUNKS_PER_WORKER,
    DEFAULT_CASED,
    DEFAULT_MAX_COMBINATIONS,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MIN_CHANGE_RATIO,
    EPSILON_PROBABILITY_BOOST,
    MIN_CHUNK_SIZE,
    PARALLEL_ENTRY_THRESHOLD,
)
from ..utils.io import is_dict_comment
from .vectorizer import Vectorizer

logger = get_logger(__name__)


# Worker initialization for proper Ctrl-C handling
def _worker_init():
    """Initialize worker process to ignore SIGINT (parent will handle)"""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _align_chunk(args):
    """
    Worker function to align a chunk of entries.

    Args:
        args: Tuple of (chunk_entries, model, log_base)

    Returns:
        List of (entry_index, new_alignment, changed) tuples
    """
    chunk_entries, model, log_base = args
    results = []

    for entry_idx, entry in chunk_entries:
        letters, phones, state, possibles = entry

        # Skip if too short
        if len(letters) < len(phones):
            results.append((entry_idx, state, False))
            continue

        was = " ".join(state)
        best_score = None
        best = None

        # Find best alignment
        for candidate in possibles:
            score = 0 if log_base else 1
            for n, letter in enumerate(letters):
                prob = model[letter][candidate[n]]
                if log_base:
                    score += math.log(prob, log_base)
                else:
                    score *= prob

            if best is None or best_score < score:
                best = candidate
                best_score = score

        now = " ".join(best)
        changed = now != was
        results.append((entry_idx, best, changed))

    return results


class EMAlign:
    """
    Align letter/sound sequences for g2p using EM
    """

    def __init__(
        self,
        locale: str | Vectorizer | None = None,
        path: str | None = None,
        phoneset_name: str | None = None,
        max_iterations: int | None = None,
        min_change_ratio: float | None = None,
        max_combinations: int | None = None,
        remove_stress: bool = False,
        cased: bool = DEFAULT_CASED,
        verbose: bool = True,
        parallel: bool = True,
        num_workers: int | None = None,
    ) -> None:
        """
        Initialize EM alignment.

        Args:
            locale: Language locale (e.g., 'en_US') or Vectorizer instance
            path: Path to pronunciation dictionary to load immediately
            phoneset_name: Phoneset tag (e.g. 'cmu', 'xsampa', 'ipa'); the
                tag selects stress-stripping rules and locale ``join`` keys.
            max_iterations: Maximum EM iterations (default: 100)
            min_change_ratio: Convergence threshold - stop when change ratio
                drops below this value (default: 0.0 = run until no changes)
            max_combinations: Maximum alignment combinations per word.
                Words exceeding this limit are skipped (default: 0 = unlimited)
            remove_stress: Remove stress markers from phonemes
            cased: Case-sensitive letter processing
            verbose: Enable verbose logging output
            parallel: Enable parallel processing for alignment
            num_workers: Number of parallel workers (default: cpu_count - 1)
        """
        self.max_iterations = (
            max_iterations if max_iterations is not None else DEFAULT_MAX_ITERATIONS
        )
        self.min_change_ratio = (
            min_change_ratio
            if min_change_ratio is not None
            else DEFAULT_MIN_CHANGE_RATIO
        )
        self.max_combinations = (
            max_combinations
            if max_combinations is not None
            else DEFAULT_MAX_COMBINATIONS
        )
        self.verbose = verbose
        self.parallel = parallel
        self.num_workers = num_workers if num_workers else max(1, cpu_count() - 1)
        self.init_data: list = []
        self.model: dict[str, dict[str, float]] | None = None
        self.em_data: list[list] | None = None
        self.seen: set[str] = set()

        if isinstance(locale, str):
            self.vectorizer = Vectorizer(
                locale=locale,
                phoneset_name=phoneset_name,
                remove_stress=remove_stress,
                cased=cased,
                verbose=self.verbose,
            )
        elif isinstance(locale, Vectorizer):
            self.vectorizer = locale

        if path:
            self.load_path(path)  # type: ignore[attr-defined]

    def load_prondict(self, infile) -> None:
        """Load dictionary file data"""
        start = time()
        logger.info("Loading pronunciations...")
        n = good = 0
        for line in infile:
            if is_dict_comment(line):
                continue
            n += 1
            good += self.add_line(line)

        if self.verbose:
            bad = n - good
            logger.info("done. (%.2f sec)", time() - start)
            if n > 0:
                logger.info(
                    "%d lines with %d accepted (%.3f%%), %d rejected (%.3f%%)",
                    n,
                    good,
                    100 * good / n,
                    bad,
                    100 * bad / n,
                )
            else:
                logger.info("No lines processed")

    def add_line(self, line: str) -> int:
        """Add a dictionary line (token + pron) to our data"""
        letters, phones = self.vectorizer.letters_and_phones(line)

        # Skip invalid lines
        if letters is None or phones is None:
            return 0
        key = " ".join(letters) + "::" + " ".join(phones)
        if key in self.seen:
            return 0  # remove duplicates (after normalization)
        self.seen.add(key)

        # Any locale that declares a liaison_pad (currently fr_*) gets it
        # appended as a sentinel letter so the EM aligner can learn the
        # word-final liaison context. Don't hard-code fr_FR here so other
        # liaison-aware locales (fr_CA, fr_BE) work out of the box.
        if self.vectorizer.liaison_pad:
            letters.append(self.vectorizer.liaison_pad)

        lett_len = len(letters)
        phon_len = len(phones)

        diff = lett_len - phon_len
        if diff < 0:
            if self.verbose:
                logger.debug(
                    "short(%d): %s :: %s",
                    diff,
                    " ".join(letters),
                    " ".join(phones),
                )
            return 0

        if self.max_combinations:
            num_combinations = math.comb(lett_len, len(phones))
            if num_combinations > self.max_combinations:
                if self.verbose:
                    logger.debug(
                        "toomany(C[%d,%d]=%d): %s :: %s",
                        lett_len,
                        phon_len,
                        num_combinations,
                        " ".join(letters),
                        " ".join(phones),
                    )
                return 0

        self.init_data.append([letters, phones])

        return 1

    def initialize(self, constrain: bool = True) -> None:
        """Set up initial model state"""
        start = time()

        if self.verbose:
            logger.info("     entry count: %d", len(self.init_data))
            logger.info("  max_iterations: %d", self.max_iterations)
            logger.info("min_change_ratio: %s", self.min_change_ratio)
            logger.info("max_combinations: %d", self.max_combinations)
            logger.info("Initializing EM...")

        left = set()
        right = set()
        for letters, phones in self.init_data:
            left.update(letters)
            right.update(phones)

        right.add(self.vectorizer.epsilon)

        model: dict[str, dict[str, float]] = {}
        for letter in left:
            model[letter] = dict.fromkeys(right, 1)
            if constrain:
                if letter in right:
                    model[letter][letter] += 1
                if not ud.category(letter[0]).startswith("L"):
                    model[letter][self.vectorizer.epsilon] += EPSILON_PROBABILITY_BOOST
            total = sum(model[letter].values())
            model[letter] = {phone: val / total for phone, val in model[letter].items()}

        init_data = []
        possibles_count = 0
        for entry in self.init_data:
            letters, phones = entry

            state = list(phones)
            if len(state) < len(letters):
                state.extend([self.vectorizer.epsilon] * (len(letters) - len(state)))

            possibles = self.possible_alignments(entry)
            possibles_count += len(possibles)
            init_data.append([letters, phones, state, possibles])

        self.model = model
        self.em_data = init_data
        entry_count = len(init_data)
        if self.verbose:
            logger.info("done. (%.2f sec)", time() - start)
            if entry_count:
                logger.info(
                    "%d entries with %d combinations (%.2f avg)",
                    entry_count,
                    possibles_count,
                    possibles_count / entry_count,
                )

        # Rough RSS pressure check: each alignment list is ~50 B of Python
        # overhead (lst header + cached pointers; phone strings are interned
        # so they don't multiply). With a 7-letter window, a single big
        # entry can dominate. If the total looks alarming, surface a hint.
        BYTES_PER_ALIGNMENT = 50
        from ..constants import INIT_DATA_MEMORY_WARN_BYTES

        approx_bytes = possibles_count * BYTES_PER_ALIGNMENT
        if approx_bytes > INIT_DATA_MEMORY_WARN_BYTES:
            gb = approx_bytes / 1024 / 1024 / 1024
            logger.warning(
                "EMAlign init_data is approximately %.1f GB across %d "
                "alignment lists. If this exceeds available RAM, set "
                "max_combinations to a smaller value (currently %d) and "
                "retrain. See phonebox/constants.py for the default and "
                "the rationale.",
                gb,
                possibles_count,
                self.max_combinations,
            )

    def possible_alignments(self, row):
        """Produce all possible alignments where letters <= phones"""
        letters, phones = row

        num_eps = len(letters) - len(phones)
        if num_eps <= 0:
            return [phones]

        indx = list(range(len(letters)))
        result = []

        # c picks which letter positions are silent (epsilon). Reverse the
        # phones so pop() (from the tail) hands them out left-to-right into the
        # remaining, non-epsilon positions.
        for c in combinations(indx, num_eps):
            phon = list(reversed(phones))
            pron = []
            for n, _letter in enumerate(letters):
                if n in c:
                    pron.append(self.vectorizer.epsilon)
                else:
                    pron.append(phon.pop())
            result.append(pron)

        return result

    def make_model(self):
        """Make a model given the current alignments"""
        model = {}
        phone_set = set()
        for entry in self.em_data:
            letters, phones, state, possibles = entry
            for n, letter in enumerate(letters):
                phone = state[n]
                phone_set.add(phone)
                model[letter][phone] = model.setdefault(letter, {}).get(phone, 0) + 1

        for letter in list(model):
            for phone in phone_set:
                if phone not in model[letter]:
                    model[letter][phone] = 1
                else:
                    model[letter][phone] += 1

            # Apostrophe and hyphen are almost always silent; bias them toward
            # epsilon so EM doesn't steal a real phone onto them.
            if letter in ["'", "-"]:
                model[letter][self.vectorizer.epsilon] += EPSILON_PROBABILITY_BOOST

            total = sum(model[letter].values())
            model[letter] = {phone: val / total for phone, val in model[letter].items()}

        self.model = model

    def align_once(self, label="-", log_base=None):
        """Find the best alignment given the model"""
        start = time()
        entry_count = len(self.em_data)

        # Choose alignment strategy
        use_parallel = self.parallel and entry_count > PARALLEL_ENTRY_THRESHOLD

        if use_parallel:
            try:
                changed = self._align_parallel(log_base)
            except KeyboardInterrupt:
                logger.warning("Alignment interrupted by user")
                raise
            except Exception as e:
                logger.warning(
                    "Parallel alignment failed (%s), falling back to serial", e
                )
                self.parallel = False
                changed = self._align_serial(log_base)
        else:
            changed = self._align_serial(log_base)

        ratio = changed / entry_count
        if self.verbose:
            logger.info(
                "%s\t%d\t%.1f%%\t%.2f", label, changed, 100 * ratio, time() - start
            )

        return changed, ratio

    def _align_serial(self, log_base=None):
        """Serial alignment of all entries."""
        changed = 0

        for entry in self.em_data:
            letters, phones, state, possibles = entry
            if len(letters) < len(phones):
                if self.verbose:
                    logger.debug("%d < %d: skipping", len(letters), len(phones))
                continue

            was = " ".join(state)
            best = self._find_best_alignment(letters, possibles, log_base)

            if " ".join(best) != was:
                entry[2] = best
                changed += 1

        return changed

    def _find_best_alignment(self, letters, possibles, log_base=None):
        """Find best alignment candidate for a word."""
        best_score = None
        best = None

        for candidate in possibles:
            score = 0 if log_base else 1
            for n, letter in enumerate(letters):
                prob = self.model[letter][candidate[n]]
                if log_base:
                    score += math.log(prob, log_base)
                else:
                    score *= prob

            if best is None or best_score < score:
                best = candidate
                best_score = score

        return best

    def _align_parallel(self, log_base=None):
        """
        Parallel version of alignment using multiprocessing.
        Returns number of changed entries.
        """
        entry_count = len(self.em_data)
        changed = 0

        # Prepare indexed entries for chunking
        indexed_entries = list(enumerate(self.em_data))

        # Split into chunks for workers
        chunk_size = max(
            MIN_CHUNK_SIZE, entry_count // (self.num_workers * CHUNKS_PER_WORKER)
        )
        chunks = []
        for i in range(0, entry_count, chunk_size):
            chunk = indexed_entries[i : i + chunk_size]
            chunks.append((chunk, self.model, log_base))

        # Process chunks in parallel with Ctrl-C handling
        try:
            with Pool(processes=self.num_workers, initializer=_worker_init) as pool:
                # Map-Reduce: process chunks and collect results
                all_results = []
                for chunk_results in pool.imap_unordered(_align_chunk, chunks):
                    all_results.extend(chunk_results)

                # Update entries with results
                for entry_idx, new_alignment, did_change in all_results:
                    if did_change:
                        self.em_data[entry_idx][2] = new_alignment
                        changed += 1
                    # Even if unchanged, could update but it's the same value
                    # so we skip for efficiency

        except KeyboardInterrupt:
            # Pool context manager will handle cleanup
            raise

        return changed

    def align(self, init: bool = True) -> None:
        """Iterate over the aligning, model building"""
        start = time()
        if init:
            self.initialize()

        if self.verbose:
            logger.info("iter\tchanged\tpercent\telapsed")

        for n in range(self.max_iterations):
            changed, ratio = self.align_once(n + 1)
            self.make_model()
            if ratio < self.min_change_ratio or not changed:
                break

        if self.verbose:
            logger.info("total time: %.1f", time() - start)

    def next_alignment(self) -> Iterator[str]:
        """Iterator over alignments"""
        for entry in self.em_data:
            letters, phones, state, possibles = entry
            yield " ".join(letters) + "\t" + " ".join(state)

    def write(self, outfile) -> None:
        """Write out alignments"""
        for line in self.next_alignment():
            print(line, file=outfile)
