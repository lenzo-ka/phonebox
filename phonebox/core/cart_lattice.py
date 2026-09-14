"""Reusable joint-unit support with CART-scored coherent path decoding."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from types import MappingProxyType

from .joint_decode import _decode_prepared
from .multigram_align import Unit
from .multigram_lm import decode_unit_id, unit_id


class CartDecompositionLattice:
    """Owned, reusable unit inventory; no q prior or sequence LM is added.

    A path is scored by the product of its projected per-position CART target
    probabilities. Support enforces joint-unit boundaries, but the classifier
    remains a marginal model. This is exact Viterbi, not pronunciation-level
    marginalization or an assertion of learned cross-position correlations.
    """

    def __init__(self, units: Iterable[Unit], *, join_char: str, epsilon: str):
        if not join_char or not epsilon:
            raise ValueError("lattice requires nonempty join and epsilon markers")
        inventory = []
        seen = set()
        index: dict[str, list[Unit]] = {}
        targets = {}
        ids = {}
        for entry in units:
            if (
                not isinstance(entry, (list, tuple))
                or len(entry) != 2
                or any(not isinstance(side, (list, tuple)) for side in entry)
            ):
                raise ValueError("invalid decomposition unit structure")
            left, right = entry
            unit = (tuple(left), tuple(right))
            left, right = unit
            if not left or any(
                not isinstance(token, str) or not token for token in (*left, *right)
            ):
                raise ValueError("invalid decomposition unit")
            if any(join_char in token or token == epsilon for token in (*left, *right)):
                raise ValueError("reserved decomposition unit syntax")
            if unit in seen:
                continue
            seen.add(unit)
            inventory.append(unit)
            index.setdefault(left[0], []).append(unit)
            targets[unit] = join_char.join(right) if right else epsilon
            ids[unit] = unit_id(unit)
            if decode_unit_id(ids[unit]) != unit:
                raise ValueError("decomposition unit does not round trip")
        self.units = tuple(inventory)
        self.join_char = join_char
        self.epsilon = epsilon
        self._index = MappingProxyType(
            {key: tuple(value) for key, value in index.items()}
        )
        self._targets = MappingProxyType(targets)
        self._ids = MappingProxyType(ids)

    def decode(
        self,
        letters: Sequence[str],
        distributions: Sequence[str | Mapping[str, float]],
    ) -> list[str] | None:
        """Return the best legal pronunciation, or None when no path exists.

        Deterministic CART leaves are point distributions. Missing labels have
        zero probability; no floor, q multiplier, beam or silent fallback is
        introduced. Candidate order determines exact-score ties consistently.
        """
        if len(letters) != len(distributions):
            raise ValueError("lattice distributions must match letter positions")
        logs = []
        for distribution in distributions:
            if isinstance(distribution, str):
                distribution = {distribution: 1.0}
            values = {}
            for target, probability in distribution.items():
                if (
                    not isinstance(target, str)
                    or isinstance(probability, bool)
                    or not isinstance(probability, (int, float))
                ):
                    raise ValueError("invalid CART target distribution")
                if not math.isfinite(probability) or not 0 <= probability <= 1:
                    raise ValueError("invalid CART target probability")
                if probability:
                    values[target] = math.log(probability)
            total = math.fsum(distribution.values())
            if not math.isclose(total, 1.0, abs_tol=1e-6, rel_tol=1e-6):
                raise ValueError("CART target probabilities must sum to one")
            # Tiny probabilities omitted by Cartlet, or fractional accumulation,
            # can leave small mass deficits. Normalization adds one constant to
            # every complete path, so its ranking is unchanged.
            if total != 1.0:
                normalization = math.log(total)
                values = {
                    target: score - normalization for target, score in values.items()
                }
            logs.append(values)

        def score(position: int, unit: Unit) -> float:
            value = logs[position].get(self._targets[unit], -math.inf)
            for offset in range(1, len(unit[0])):
                value += logs[position + offset].get(self.epsilon, -math.inf)
            return value

        return _decode_prepared(list(letters), self._index, None, 0, self._ids, score)
