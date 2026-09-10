# Locale exemplar inventories

Phonebox packages the standard and auxiliary writing-system exemplars published
by [Unicode ICU](https://icu.unicode.org/) for every locale available to the
pinned generator. Most ICU locale data comes from the [Unicode Common Locale
Data Repository (CLDR)](https://cldr.unicode.org/), as described by the
[ICU data guide](https://unicode-org.github.io/icu/userguide/icu_data/). These
exemplars describe orthography: they are not phoneme inventories,
pronunciation models, text filters, or evidence that an accent may be folded.

```python
from phonebox.exemplars import get_exemplars, supported_locales

spanish = get_exemplars("es-MX")
assert "ñ" in spanish
print(len(supported_locales()))
for start, end in get_exemplars("ko").iter_ranges():
    print(hex(start), hex(end))
```

Locale lookup ignores case and accepts hyphens or underscores when the resulting
ID exactly matches a packaged ICU locale. It never falls back to a related
locale, which keeps distinctions such as `zh_Hans` and `zh_Hant` explicit.
Membership treats ICU multi-codepoint string members as single set members.
Iteration is lazy over stored ranges; `iter_ranges()` avoids expanding them.

The same generated artifact supplies Phonebox's ICU-free locale canonicalization
and likely-subtag lookup. Resource owners decide which fallback is appropriate:
bare `it` can select the `it_IT` spelling policy, while exemplar lookup keeps the
exact bare `it` inventory. Equivalent standard and auxiliary profiles may share
an existing spelling policy; they never substitute a pronunciation model or
phone post-processing defaults. Saved models continue to use their embedded
training-time preprocessing.

```python
from phonebox import canonical_locale, resolve_locale

assert canonical_locale("IT-it") == "it_IT"
selection = resolve_locale("it", {"en_US", "it_IT"})
assert selection.requested == "it"
assert selection.resolved == "it_IT"
assert selection.match == "likely"
```

`LocaleResolution` always returns the canonical `requested` identity, the
selected member of the caller's `available` collection as `resolved` (or
`None`), and `match` as `exact`, `likely`, `compatible`, or `None`. Compatibility
is opt-in through the mapping argument; the generic resolver does not infer it.

## Regeneration

Install the development requirements, then run:

```bash
python tools/generate_exemplars.py
python tools/generate_exemplars.py --check
```

The checked-in JSON records icukit 0.4.0, the icukit-pyicu 78.3.0 backend,
ICU 78.3, and Unicode 17.0. Generation uses ICU's native UnicodeSet range and
string APIs to read standard and auxiliary exemplars, fails if any locale/kind
cannot be read, then deduplicates identical inventories and paired profiles.
It derives canonical aliases and likely locales from ICU and derives
orthographic compatibility only when the standard and auxiliary profiles are
identical. The runtime standard-library resolver consumes that pinned metadata
and selects an existing Phonebox locale policy; the policy's checked-in
transliteration rules and joins remain authoritative. icukit contributes only
orthographic locale information: it defines no Phonebox phonemes, pronunciation
model, or inferred accent-folding rule.

The artifact records its ICU/CLDR source and Unicode-3.0 license reference.
The complete Unicode License v3 notice is shipped as `LICENSE-UNICODE`, with
additional provenance in `THIRD_PARTY_NOTICES.md`.
The short keys inside an inventory are `c` (packed singleton characters), `r`
(inclusive code-point ranges), and `s` (multi-codepoint string members).

Exact duplicate sets and profiles share one stored record. Partially overlapping sets
remain independent range lists: for example, the standard `zh` and `ja` inventories
share 1,093 members, while `zh_Hant` and `ja` share 1,341. Factoring those sparse
partial overlaps would make the format and reader more complex for little size gain.
