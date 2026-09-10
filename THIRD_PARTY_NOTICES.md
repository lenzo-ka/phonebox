# Third-party notices

## Unicode ICU and CLDR locale data

`phonebox/config/exemplars.json` is a modified, generated representation of
locale data exposed by [Unicode ICU](https://icu.unicode.org/) through
icukit 0.4.0 and the icukit-pyicu 78.3.0 backend. Most ICU locale data is
sourced from the [Unicode Common Locale Data Repository
(CLDR)](https://cldr.unicode.org/); ICU documents that relationship in its
[ICU data guide](https://unicode-org.github.io/icu/userguide/icu_data/).

Phonebox converts ICU standard and auxiliary exemplar sets into compact native
range, singleton, and string records; deduplicates identical inventories and
profiles; and adds derived canonical-locale, likely-subtag, and orthographic
profile-compatibility metadata. The resulting artifact is therefore modified
from the upstream representation.

The generated artifact is distributed under the Unicode License v3. The
complete applicable notice is reproduced in `LICENSE-UNICODE`. Phonebox source
code is distributed under the BSD 2-Clause License in `LICENSE`.
