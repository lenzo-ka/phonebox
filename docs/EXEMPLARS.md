# Locale exemplar inventories

Phonebox packages the standard and auxiliary writing-system exemplars published
by ICU for every locale available to the pinned generator. They describe
characters conventionally used to write a language. They are not pronunciation
models, text filters, or evidence that accents may be discarded.

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

## Regeneration

Install the development requirements, then run:

```bash
python tools/generate_exemplars.py
python tools/generate_exemplars.py --check
```

The checked-in JSON records icukit 0.3.0, the icukit-pyicu 78.3.0 backend,
ICU 78.3, and Unicode 17.0. Generation uses ICU's native UnicodeSet range and
string APIs, fails if any locale/kind cannot be read, deduplicates identical
inventories and standard/auxiliary profiles, and emits deterministic JSON.
The short keys inside an inventory are `c` (packed singleton characters), `r`
(inclusive code-point ranges), and `s` (multi-codepoint string members).

Exact duplicate sets and profiles share one stored record. Partially overlapping sets
remain independent range lists: for example, the standard `zh` and `ja` inventories
share 1,093 members, while `zh_Hant` and `ja` share 1,341. Factoring those sparse
partial overlaps would make the format and reader more complex for little size gain.
