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

On the verified Ubuntu build host, GCC/G++/GFortran 13.3.0, Autoconf 2.71,
Automake 1.16.5, Libtool 2.4.7, and Make 4.3 were available. Start in a fresh
`.cache/toolchains/phonetisaurus` directory. The following commands install
only into its private prefix; they do not install system packages.

```sh
export PHONEBOX_TOOLCHAIN_ROOT="$PWD"
export PATH="$PHONEBOX_TOOLCHAIN_ROOT/prefix/bin:$PATH"
export MAKEFLAGS=-j2 CMAKE_BUILD_PARALLEL_LEVEL=2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1

git clone https://github.com/mjansche/openfst.git openfst
git -C openfst checkout e04f9c15fab41a3544355857a8b0c71ee9229825
git clone https://github.com/mitlm/mitlm.git mitlm
git -C mitlm checkout 553edca763a8e142edd8ef6d51404bbf43b79c95
git clone https://github.com/AdolfVonKleist/Phonetisaurus.git phonetisaurus
git -C phonetisaurus checkout f08d3dfb10b8d619e665a9581d2a327bcc2504f7

mkdir -p build-prerequisites
cd build-prerequisites
apt-get download autoconf-archive=20220903-3
printf '%s  %s\n' \
  36b178ef82b4a7fe8c4b7c4992b569b6f61454f2d1ec7b17f28d2570174a95b7 \
  autoconf-archive_20220903-3_all.deb | sha256sum -c -
dpkg-deb -x autoconf-archive_20220903-3_all.deb extracted
export ACLOCAL_PATH="$PHONEBOX_TOOLCHAIN_ROOT/build-prerequisites/extracted/usr/share/aclocal${ACLOCAL_PATH:+:$ACLOCAL_PATH}"

cd "$PHONEBOX_TOOLCHAIN_ROOT/openfst"
autoreconf -fi
./configure --prefix="$PHONEBOX_TOOLCHAIN_ROOT/prefix" \
  --enable-static --enable-shared --enable-far --enable-ngram-fsts \
  CXXFLAGS='-O2 -std=c++11'
nice -n 10 make -j2
nice -n 10 make -j2 install

cd "$PHONEBOX_TOOLCHAIN_ROOT/mitlm"
autoreconf -i
./configure --prefix="$PHONEBOX_TOOLCHAIN_ROOT/prefix" \
  CXXFLAGS='-O2 -std=c++11'
nice -n 10 make -j2
nice -n 10 make -j2 install

cd "$PHONEBOX_TOOLCHAIN_ROOT/phonetisaurus"
./configure --prefix="$PHONEBOX_TOOLCHAIN_ROOT/prefix" \
  --with-openfst-includes="$PHONEBOX_TOOLCHAIN_ROOT/prefix/include" \
  --with-openfst-libs="$PHONEBOX_TOOLCHAIN_ROOT/prefix/lib" \
  CXXFLAGS='-O2 -std=c++11 -fno-fast-math'
nice -n 10 make -j2
nice -n 10 make -j2 install
```

Regenerating OpenFst's build files avoids a dependency on an obsolete Automake
executable named by the checked-in generated files. The archive download above
supplies MITLM's required Autoconf macros without a global package installation.
The OpenFst source is a developer mirror, not an assertion that these bytes
came from an official release tarball.

The trailing `-fno-fast-math` matters with the verified compiler. The upstream
decoder build adds `-ffast-math`; the initial GCC 13 build returned empty paths
even for known words. Rebuilding with the final override restored correct
predictions on the same existing FST and word list. No training data, model,
order, or algorithm source changed. This is a build correction, not model tuning.

MITLM order eight also failed on a degenerate nine-example fixture with no
higher-order contexts. The representative toolchain smoke used longer synthetic
sequences; it verifies execution and token transport, not scientific accuracy.
A real experiment failure must be recorded rather than silently retried at a
different order. Preserve complete build logs, upstream licenses, compiler
settings, and the per-executable receipts with the experiment.
