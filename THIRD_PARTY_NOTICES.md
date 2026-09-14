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

## DeepPhonemizer training patch

`phonebox/eval/benchmark_neural_sources.py` embeds a unified diff against
[DeepPhonemizer](https://github.com/axelspringer/DeepPhonemizer) 0.0.19
(source revision `5dce7e27556aef4426f5623baf6351d266a30a73`), which the optional
neural benchmark applies to a separately installed copy. The diff reproduces
modified fragments of DeepPhonemizer's `dp/training/trainer.py`,
`dp/training/dataset.py` and `dp/model/predictor.py`, so it is a modification
of MIT-licensed code redistributed with this package. The same diff is tracked
as `docs/patches/deepphonemizer-0.0.19.patch`.

DeepPhonemizer is Copyright (c) 2021 Axel Springer News Media & Tech GmbH & Co.
KG - Ideas Engineering, under the MIT License. The complete license text is
reproduced in `LICENSE-DEEPPHONEMIZER` (also `docs/patches/DeepPhonemizer-LICENSE`).
DeepPhonemizer itself, PyTorch and the other neural toolchain dependencies are
not bundled; the pinned installation recipe is in
`docs/BENCHMARK_TOOLCHAINS.md` and `docs/locks/deepphonemizer-python312.txt`.

## Developer benchmark sources

The optional [benchmark workflow](docs/REPRODUCIBLE_BENCHMARKS.md) fetches
CMUdict and WikiPron pronunciation data into caller-owned caches and
can invoke separately installed Sequitur and Phonetisaurus training tools.
These datasets and executables are not bundled with Phonebox. Their licenses
are distinct from this package: CMUdict BSD-2-Clause; WikiPron/Wiktionary data
under its separate CC-BY-SA-4.0 terms; Sequitur GPL-2.0-only; Phonetisaurus and MITLM BSD-3-Clause;
OpenFst Apache-2.0. The benchmark manifests and protocol provide immutable
source references, attribution, and license links. Retain upstream notices
when sharing copies or prepared data.
