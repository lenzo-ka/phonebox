"""Packaged ICU exemplar inventory API."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from phonebox.exemplars import get_exemplars, supported_locales

ROOT = Path(__file__).parents[1]


def test_all_generated_locales_and_alias_lookup():
    locales = supported_locales()
    assert len(locales) == 906
    assert locales == tuple(sorted(locales))
    assert get_exemplars("EN-us") == get_exemplars("en_US")
    assert "a" in get_exemplars("en_US")
    assert "A" not in get_exemplars("en_US")


def test_unknown_locale_and_kind_do_not_fallback():
    with pytest.raises(KeyError, match="unknown exemplar locale"):
        get_exemplars("zh_Zzzz")
    with pytest.raises(ValueError, match="unknown exemplar kind"):
        get_exemplars("en_US", "made-up")


def test_ranges_strings_and_boundaries_are_exact():
    korean = get_exemplars("ko")
    assert len(korean.ranges) == 1
    start, end = korean.ranges[0]
    assert chr(start) in korean and chr(end) in korean
    assert chr(start - 1) not in korean and chr(end + 1) not in korean
    aghem = get_exemplars("agq")
    assert "ɔ̀" in aghem
    assert "ɔ̀" in tuple(aghem)
    assert len(aghem) == sum(1 for _ in aghem)


def test_exact_inventories_and_profiles_are_deduplicated():
    data = json.loads((ROOT / "phonebox/config/exemplars.json").read_text())
    assert data["generator"]["source"] == {
        "icu": "https://icu.unicode.org/",
        "cldr": "https://cldr.unicode.org/",
    }
    assert data["generator"]["license"] == {
        "id": "Unicode-3.0",
        "notice": "LICENSE-UNICODE",
    }
    inventories = [
        json.dumps(v, ensure_ascii=False, sort_keys=True) for v in data["inventories"]
    ]
    profiles = [tuple(v) for v in data["profiles"]]
    assert len(inventories) == len(set(inventories)) == 415
    assert len(profiles) == len(set(profiles)) == 242


def test_reader_import_has_no_icu_dependency():
    code = "import sys; sys.modules['icu']=None; import phonebox.exemplars as e; assert 'en_US' in e.supported_locales()"
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_generator_import_and_cli_help_do_not_require_icu():
    code = (
        "import sys; sys.modules['icu']=None; sys.modules['icukit']=None; "
        "from phonebox.dev.exemplars import generate_exemplars; "
        "from phonebox.cli.main import main; "
        "main(['exemplars', 'generate', '--help'])"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "--check" in result.stdout


def test_generator_mechanics_preserve_native_ranges_and_strings():
    from phonebox.dev import exemplars as module

    class NativeSet:
        def getRangeCount(self):
            return 2

        def getRangeStart(self, index):
            return ("a", "x")[index]

        def getRangeEnd(self, index):
            return ("c", "x")[index]

        def strings(self):
            return iter(["ch"])

    assert module._inventory(NativeSet()) == {"c": "x", "r": [[97, 99]], "s": ["ch"]}
    with pytest.raises(RuntimeError, match=r"expected .*ICU=78\.3.*found .*ICU=77\.1"):
        module.validate_versions(
            {
                "icukit": "0.4.0",
                "icukit-pyicu": "78.3.0",
                "ICU": "77.1",
                "Unicode": "16.0",
            }
        )


def test_format_mismatch_has_regeneration_guidance(monkeypatch):
    import phonebox.exemplars as exemplars

    class Resource:
        def joinpath(self, name):
            return self

        def read_text(self, encoding):
            return '{"format":999}'

    exemplars._data.cache_clear()
    monkeypatch.setattr(exemplars, "files", lambda package: Resource())
    with pytest.raises(RuntimeError, match="regenerate.*pinned dev tools"):
        exemplars.supported_locales()
    exemplars._data.cache_clear()
