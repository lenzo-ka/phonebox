# Developer benchmark toolchains

The benchmark driver is installed with `python -m pip install -e '.[dev]'`.
Sequitur and Phonetisaurus are separate developer executables. Building them is
an explicit step: ordinary Phonebox installation does not download or compile
external systems. See [the comparison protocol](REPRODUCIBLE_BENCHMARKS.md) for
data attribution and the libraries' separate licenses.

The following Sequitur recipe was exercised on Linux x86-64 with Python 3.12.3
and GCC 13.3.0. Pins describe that verified environment, not a promise that the
same native build works on every supported Phonebox platform. Keep the source
checkout and its license with the developer environment.

## Sequitur

Start in a fresh `.cache/toolchains/sequitur` directory. Run builds at low
priority and bound build and numerical-library threads when sharing a machine.
The commands below build the actual authors' source, without patches.

```sh
git clone https://github.com/sequitur-g2p/sequitur-g2p.git source
git -C source checkout 7bd56d5d502325e0be3f14b7d898720a39db3338
python3.12 -m venv venv
export PATH="$PWD/venv/bin:$PATH"
export MAKEFLAGS=-j2 CMAKE_BUILD_PARALLEL_LEVEL=2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
python -m pip install pip==25.3 setuptools==80.9.0 wheel==0.45.1 \
  numpy==2.5.3 six==1.17.0 swig==4.3.1
nice -n 10 python -m pip wheel --no-build-isolation --no-deps ./source -w wheels
python -m pip install --no-deps wheels/*.whl
python -c 'import importlib.metadata; print(importlib.metadata.version("sequitur-g2p"))'
venv/bin/g2p.py --help
```

The resulting package version is 1.0.1668.30. Its `--version` text embeds an older
SVN revision, so retain the Git commit and distribution version as provenance.
Build against the same NumPy major version used at runtime: a NumPy 1.x build
cannot be reused with the pinned NumPy 2.x environment.

The runner invokes `g2p.py` with UTF-8 input and an explicit development file.
It trains orders 1–3 with minimum one and maximum ten iterations per order,
then selects using development predictions. Explicit minimum iterations matter:
the upstream default minimum of twenty conflicts with a maximum of ten.
A small accented-grapheme and multi-codepoint-IPA smoke test verified token
transport before corpus evaluation.

## Phonetisaurus

The pinned training pipeline uses MITLM, avoiding a dependency on SRILM.
Required sources are:

| Component | Source revision | License |
| --- | --- | --- |
| Phonetisaurus | `f08d3dfb10b8d619e665a9581d2a327bcc2504f7` | BSD-3-Clause |
| MITLM | `553edca763a8e142edd8ef6d51404bbf43b79c95` | BSD-3-Clause |
| OpenFst 1.7.2 developer mirror | `e04f9c15fab41a3544355857a8b0c71ee9229825` | Apache-2.0 |

The C++ pipeline is alignment, MITLM language-model estimation, and ARPA-to-WFST
compilation, followed by model-only decoding. Python bindings are not needed for
this route. The driver never supplies a gold lexicon to the decoder; doing so
could substitute reference pronunciations for model predictions.
