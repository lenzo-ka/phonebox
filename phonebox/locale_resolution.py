"""ICU-derived locale identity and resource resolution without runtime ICU."""

from __future__ import annotations

import json
import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files

_TAG_RE = re.compile(
    r"^(?P<language>[A-Za-z]{2,3})"
    r"(?:[_-](?P<script>[A-Za-z]{4}))?"
    r"(?:[_-](?P<region>[A-Za-z]{2}|[0-9]{3}))?"
    r"(?P<variants>(?:[_-][A-Za-z0-9]{4,8})*)$"
)


@dataclass(frozen=True)
class LocaleResolution:
    """A canonical request, selected resource, and how it matched.

    ``resolved`` and ``match`` are both ``None`` when no resource matches;
    otherwise ``match`` is ``exact``, ``likely``, or ``compatible``.
    """

    requested: str
    resolved: str | None
    match: str | None


@lru_cache(maxsize=1)
def _metadata() -> dict:
    resource = files("phonebox.config").joinpath("exemplars.json")
    data = json.loads(resource.read_text(encoding="utf-8"))
    metadata = data.get("locale_resolution")
    if not isinstance(metadata, dict):
        raise RuntimeError("exemplar artifact lacks locale resolution metadata")
    return metadata


def canonical_locale(locale: str) -> str:
    """Return a normalized locale identity without adding likely subtags."""
    if not isinstance(locale, str) or not locale:
        raise ValueError("locale must be a non-empty string")
    if locale.casefold() == "default":
        return "default"
    match = _TAG_RE.fullmatch(locale)
    if match is None:
        raise ValueError(f"malformed locale identifier: {locale!r}")
    language = match.group("language").lower()
    language = _metadata().get("language_aliases", {}).get(language, language)
    components = [language]
    if script := match.group("script"):
        components.append(script.title())
    if region := match.group("region"):
        components.append(region.upper())
    variants = match.group("variants")
    if variants:
        components.extend(part.upper() for part in re.split(r"[_-]", variants) if part)
    return "_".join(components)


def likely_locale(locale: str) -> str | None:
    """Return the pinned likely identity for a bare language, if available."""
    requested = canonical_locale(locale)
    if "_" in requested or requested == "default":
        return None
    value = _metadata().get("likely", {}).get(requested)
    return value if isinstance(value, str) else None


def locale_candidates(
    locale: str, *, include_compatible: bool = False
) -> tuple[str, ...]:
    """Return ordered resource candidates, independent of an available set."""
    requested = canonical_locale(locale)
    candidates = [requested]
    likely = likely_locale(requested)
    if likely:
        candidates.append(likely)
        parts = likely.split("_")
        if len(parts) >= 3 and len(parts[1]) == 4:
            candidates.append("_".join((parts[0], *parts[2:])))
    if include_compatible:
        compatible = _metadata().get("orthographic_compatible", {}).get(requested, [])
        candidates.extend(compatible)
    return tuple(dict.fromkeys(candidates))


def resolve_locale(
    locale: str,
    available: Collection[str],
    compatible: Mapping[str, Sequence[str]] | None = None,
) -> LocaleResolution:
    """Resolve a locale against one caller-supplied resource namespace."""
    requested = canonical_locale(locale)
    requested_parts = requested.split("_")
    requested_language = requested_parts[0]
    requested_script = (
        requested_parts[1]
        if len(requested_parts) > 1 and len(requested_parts[1]) == 4
        else None
    )
    canonical_available: dict[str, str] = {}
    for item in available:
        canonical = canonical_locale(item)
        previous = canonical_available.get(canonical)
        if previous is not None and previous != item:
            raise ValueError(f"ambiguous available locale identity: {canonical}")
        canonical_available[canonical] = item
    if requested in canonical_available:
        return LocaleResolution(requested, canonical_available[requested], "exact")
    for candidate in locale_candidates(requested)[1:]:
        if candidate in canonical_available:
            return LocaleResolution(requested, canonical_available[candidate], "likely")
    candidates = compatible.get(requested, ()) if compatible is not None else ()
    for candidate in candidates:
        canonical = canonical_locale(candidate)
        candidate_parts = canonical.split("_")
        candidate_script = (
            candidate_parts[1]
            if len(candidate_parts) > 1 and len(candidate_parts[1]) == 4
            else None
        )
        if candidate_parts[0] != requested_language:
            continue
        if requested_script is not None and candidate_script != requested_script:
            continue
        if canonical in canonical_available:
            return LocaleResolution(
                requested, canonical_available[canonical], "compatible"
            )
    return LocaleResolution(requested, None, None)


def orthographic_compatibility() -> dict[str, tuple[str, ...]]:
    """Return pinned compatible spelling-policy candidates."""
    raw = _metadata().get("orthographic_compatible", {})
    return {key: tuple(value) for key, value in raw.items()}


__all__ = [
    "LocaleResolution",
    "canonical_locale",
    "likely_locale",
    "locale_candidates",
    "orthographic_compatibility",
    "resolve_locale",
]
