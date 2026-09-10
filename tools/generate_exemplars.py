#!/usr/bin/env python3
"""Generate Phonebox's packaged ICU locale exemplar inventory."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

FORMAT_VERSION = 2
DEFAULT_OUTPUT = Path("phonebox/config/exemplars.json")
KINDS = ("standard", "auxiliary")
EXPECTED_VERSIONS = {
    "icukit": "0.4.0",
    "icukit-pyicu": "78.3.0",
    "ICU": "78.3",
    "Unicode": "17.0",
}


def validate_versions(actual: dict[str, str]) -> None:
    """Reject generation with a different wrapper, backend, ICU, or Unicode."""
    if actual != EXPECTED_VERSIONS:
        expected = ", ".join(
            f"{name}={value}" for name, value in EXPECTED_VERSIONS.items()
        )
        found = ", ".join(
            f"{name}={actual.get(name, 'missing')}" for name in EXPECTED_VERSIONS
        )
        raise RuntimeError(
            f"generator version mismatch: expected {expected}; found {found}; "
            "install the pinned dev requirements"
        )


def inventory(unicode_set: Any) -> dict[str, Any]:
    characters: list[str] = []
    ranges: list[list[int]] = []
    for i in range(unicode_set.getRangeCount()):
        start = ord(unicode_set.getRangeStart(i))
        end = ord(unicode_set.getRangeEnd(i))
        if start == end:
            characters.append(chr(start))
        else:
            ranges.append([start, end])
    return {
        "c": "".join(characters),
        "r": ranges,
        "s": sorted(map(str, unicode_set.strings())),
    }


def generate() -> dict[str, Any]:
    try:
        import icu
        import icukit
    except ImportError as error:
        raise RuntimeError(
            "generation requires icukit==0.4.0 and icukit-pyicu==78.3.0"
        ) from error
    actual_versions = {
        "icukit": icukit.__version__,
        "icukit-pyicu": version("icukit-pyicu"),
        "ICU": icu.ICU_VERSION,
        "Unicode": icu.UNICODE_VERSION,
    }
    validate_versions(actual_versions)
    locale_values = {}
    unique = {}
    failures = []
    for locale in sorted(icukit.list_locales()):
        values = []
        for kind in KINDS:
            try:
                value = inventory(
                    icu.UnicodeSet(icukit.get_exemplar_characters(locale, kind))
                )
            except Exception as error:
                failures.append(f"{locale}/{kind}: {error}")
                continue
            encoded = json.dumps(
                value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            unique[encoded] = value
            values.append(encoded)
        if len(values) == 2:
            locale_values[locale] = values
    if failures:
        raise RuntimeError("failed to generate exemplar data:\n" + "\n".join(failures))
    encoded_inventories = sorted(unique)
    inventory_ids = {value: index for index, value in enumerate(encoded_inventories)}
    locale_profiles = {
        locale: tuple(inventory_ids[v] for v in values)
        for locale, values in locale_values.items()
    }
    profiles = sorted(set(locale_profiles.values()))
    profile_ids = {value: index for index, value in enumerate(profiles)}
    locale_ids = sorted(locale_profiles)
    languages = sorted({locale.split("_")[0] for locale in locale_ids})
    alias_inputs = (
        "".join(chars)
        for chars in itertools.product("abcdefghijklmnopqrstuvwxyz", repeat=2)
    )
    language_aliases = {
        language: canonical
        for language in alias_inputs
        if (canonical := icukit.canonicalize_locale(language)) != language
        and canonical.isalpha()
        and 2 <= len(canonical) <= 3
    }
    likely = {language: icukit.add_likely_subtags(language) for language in languages}
    configured = sorted(
        path.name
        for path in (Path(__file__).parents[1] / "phonebox/config/locales").iterdir()
        if path.is_dir() and path.name != "default"
    )
    configured_profiles = {
        locale: locale_profiles[locale]
        for locale in configured
        if locale in locale_profiles
    }
    compatible = {}
    for locale, profile in locale_profiles.items():
        parts = locale.split("_")
        if len(parts) > 1 and len(parts[1]) == 4:
            continue
        language = locale.split("_", 1)[0]
        matches = [
            candidate
            for candidate, candidate_profile in configured_profiles.items()
            if candidate.split("_", 1)[0] == language and candidate_profile == profile
        ]
        likely_parts = likely[language].split("_")
        preferred = "_".join((likely_parts[0], *likely_parts[2:]))
        matches.sort(key=lambda candidate: (candidate != preferred, candidate))
        if matches:
            compatible[locale] = matches
    return {
        "format": FORMAT_VERSION,
        "generator": {
            "icukit": actual_versions["icukit"],
            "backend": actual_versions["icukit-pyicu"],
            "icu": actual_versions["ICU"],
            "unicode": actual_versions["Unicode"],
            "source": {
                "icu": "https://icu.unicode.org/",
                "cldr": "https://cldr.unicode.org/",
            },
            "license": {
                "id": "Unicode-3.0",
                "notice": "LICENSE-UNICODE",
            },
        },
        "kinds": list(KINDS),
        "inventories": [unique[value] for value in encoded_inventories],
        "profiles": [list(value) for value in profiles],
        "locales": {
            locale: profile_ids[value] for locale, value in locale_profiles.items()
        },
        "locale_resolution": {
            "language_aliases": language_aliases,
            "likely": likely,
            "orthographic_compatible": compatible,
        },
    }


def render(data: dict[str, Any]) -> str:
    return (
        json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the packaged ICU standard and auxiliary exemplars."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail unless OUTPUT is byte-for-byte current",
    )
    args = parser.parse_args(argv)
    try:
        content = render(generate())
        if args.check:
            if (
                not args.output.exists()
                or args.output.read_text(encoding="utf-8") != content
            ):
                print(
                    f"{args.output} is stale; regenerate with this command",
                    file=sys.stderr,
                )
                return 1
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(content, encoding="utf-8")
    except (OSError, RuntimeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
