"""Known preprocessing rules for models exported before rule snapshots."""

from __future__ import annotations

_ES_MX_G2P_RULES = r"""
:: NFC ;

\u00F1 > \uE001 ;
\u00D1 > \uE002 ;
\u00FC > \uE003 ;
\u00DC > \uE004 ;

:: NFD ;
:: [:M:] Remove ;
:: NFC ;

\uE001 > \u00F1 ;
\uE002 > \u00D1 ;
\uE003 > \u00FC ;
\uE004 > \u00DC ;

:: Null ;
:: [^-.'[:L:]] Remove ;
:: Any-Lower ;
"""

_KNOWN_G2P_RULES = {"es_MX": _ES_MX_G2P_RULES}


def known_legacy_g2p_rules(locale: str | None) -> str | None:
    """Return the frozen pre-snapshot rules known for *locale*, if any."""
    return _KNOWN_G2P_RULES.get(locale)
