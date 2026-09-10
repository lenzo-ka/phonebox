"""Shared library and standalone text-tokenization behavior."""

from phonebox import G2P
from phonebox.normalize import normalize_text, tokenize_raw


def _recording_g2p():
    class RecordingG2P(G2P):
        def pronounce(self, word):
            return [word]

    return object.__new__(RecordingG2P)


def test_normalized_tokens_use_nfc_and_strip_edges_once():
    assert normalize_text("  ¡cafe\u0301!  rock-'n'-roll…  ") == [
        "café",
        "rock-'n'-roll",
    ]


def test_library_pronounce_text_uses_shared_normalized_and_raw_contracts():
    g2p = _recording_g2p()
    assert g2p.pronounce_text("¡cafe\u0301!") == [("café", ["café"])]
    assert g2p.pronounce_text("¡cafe\u0301!", raw=True) == [
        ("¡cafe\u0301!", ["¡cafe\u0301!"])
    ]
    assert tokenize_raw("  a\u0301  b ") == ["a\u0301", "b"]
