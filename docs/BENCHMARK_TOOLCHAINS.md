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
Set `PHONEBOX_CHECKOUT` to the Phonebox checkout before entering this build
directory. The commands build the authors' pinned source with the small NumPy
compatibility patch described below.

```sh
git clone https://github.com/sequitur-g2p/sequitur-g2p.git source
git -C source checkout 7bd56d5d502325e0be3f14b7d898720a39db3338
git -C source apply "$PHONEBOX_CHECKOUT/docs/patches/sequitur-numpy2.patch"
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

The [compatibility patch](patches/sequitur-numpy2.patch) replaces the removed
NumPy `sometrue` alias with `any`, preserving the reduction's semantics.
The pinned upstream package requires NumPy 2 but still calls the removed name
in higher-order discount adjustment. This path failed during the full-budget
synthetic smoke; the patched build passed the same fixture. The patch SHA-256
is `72c6d06d170d37c80f33187ea9da7bf51003fcca9784e7f3bcea62285f47b985`.
It includes upstream code context and is distributed under the accompanying
[Sequitur GPL-2.0 license](patches/SEQUITUR_LICENSE.txt); it does not change
Phonebox's runtime dependencies or license. Preserve the applied source diff
and rebuilt wheel hash with the binary receipt.

The resulting package version is 1.0.1668.30. Its `--version` text embeds an older
SVN revision, so retain the Git commit and distribution version as provenance.
Build against the same NumPy major version used at runtime: a NumPy 1.x build
cannot be reused with the pinned NumPy 2.x environment.

The runner invokes `g2p.py` with UTF-8 input and an explicit development file.
It trains orders 1–3 with the upstream minimum twenty and initial maximum one
hundred iterations per order, then selects using development predictions.
The upstream convergence test uses the recent development likelihood trend.
If an order reaches its initial cap, the runner restarts that order from the
same preceding-order initialization with a maximum of two hundred iterations.
The initial model and logs are preserved; exported models are not treated as
training checkpoints. An order still capped at two hundred is explicitly
reported in `iteration_limited_orders`, without a claim of convergence.
Test decoding begins only after all orders complete this declared policy.

The Python API exposes `sequitur_min_iterations`, `sequitur_max_iterations`
and `sequitur_extension_iterations`; the CLI exposes matching hyphenated
options, with defaults 20/100/200. Setting the extension equal to the initial
maximum disables the restart for diagnostic runs. All budgets, attempts,
stop reasons and final development likelihoods appear in result JSON.
Malformed stopping evidence or upstream iteration failure aborts evaluation.
The earlier maximum-ten runs were private budget pilots: development
likelihood was still improving, so they are not the final comparison.
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


## Source and binary receipts

Use Phonebox's public `phonebox.eval.benchmark_provenance.write_tool_receipt`
API, also exposed as `phonebox compare benchmark-receipt`, to bind metadata to
the actual executable and observed source tree. This API was exercised for
Sequitur's launcher and all four native pipeline executables before the final
smoke. It records the pinned source revision, tracked diff hash, dirty status
and executable hash. The Sequitur compatibility patch is an intentional tracked
diff; do not conceal it by ignoring or reverting the source file.

Generated build products may need explicit entries in the source checkout's
`.git/info/exclude`; inspect them first. For example, Sequitur produces
`/sequitur_g2p.egg-info/`. Ignore only generated products, not source edits.
Run receipt generation from an environment containing Phonebox's `[dev]` tools:

```python
from phonebox.eval.benchmark_provenance import write_tool_receipt

write_tool_receipt(
    "venv/bin/g2p.py", "source",
    declared_version="1.0.1668.30",
    expected_revision="7bd56d5d502325e0be3f14b7d898720a39db3338",
    build={
        "numpy": "2.5.3",
        "compatibility_patch_sha256":
            "72c6d06d170d37c80f33187ea9da7bf51003fcca9784e7f3bcea62285f47b985",
    },
)
```

Supply additional observed compiler, wheel, installed-module and shared-library
hashes through `build`; the generated binding fields cannot be overridden.
When invoking a launcher, pass the launcher as `executable` and preserve the
underlying program hash separately as `underlying_executable_sha256`.
Create one receipt beside each resolved executable. `estimate-ngram` must use
MITLM's own source revision/version; the other three native executables use
Phonetisaurus's. Source and binary identities must not be conflated with the
whole toolchain's identity. The runner verifies each receipt's binary binding.

## Optional neural toolchain: DeepPhonemizer

DeepPhonemizer is Axel Springer's MIT-licensed author implementation, pinned to
[0.0.19 source revision 5dce7e2](https://github.com/axelspringer/DeepPhonemizer/tree/5dce7e27556aef4426f5623baf6351d266a30a73).
Its [license notice](patches/DeepPhonemizer-LICENSE) accompanies the
[reproducible training patch](patches/deepphonemizer-0.0.19.patch). Model,
Adam optimizer and greedy decoder remain upstream implementations. The patch
adds an epoch callback, retains partial batches, rejects nonfinite loss or
gradients, fixes best-checkpoint bookkeeping, supports a smaller-than-bin
population, and moves Predictor inputs to the actual model device.

Torch is not an ordinary Phonebox runtime, `[dev]` or CI dependency. Use a
separate environment. The [exact dependency lock](locks/deepphonemizer-python312.txt)
was exercised with Python 3.12.12 on macOS arm64, Torch 2.5.1 and NumPy 1.26.4,
with CPU and MPS synthetic train/save/reload tests. This is compatibility
evidence, not measured neural accuracy or an assertion of other platform support.
The custom upstream checkpoint uses unrestricted `torch.load`; load only your
own trusted checkpoints. Newer Torch checkpoint defaults are not verified.

From the Phonebox checkout, create the toolchain and export its patch using the
installed public command. Keep download/build logs and the pip artifact report:

```bash
PHONEBOX_NEURAL_ROOT="$PWD/.cache/benchmarks/neural-toolchain"
python3.12 -m venv "$PHONEBOX_NEURAL_ROOT/venv"
"$PHONEBOX_NEURAL_ROOT/venv/bin/python" -m pip install \
  --report "$PHONEBOX_NEURAL_ROOT/pip-artifacts.json" \
  -r docs/locks/deepphonemizer-python312.txt
"$PHONEBOX_NEURAL_ROOT/venv/bin/python" -m pip install --no-deps --no-build-isolation .
"$PHONEBOX_NEURAL_ROOT/venv/bin/phonebox" compare benchmark-neural-patch \
  --output "$PHONEBOX_NEURAL_ROOT/training.patch"
git clone https://github.com/axelspringer/DeepPhonemizer.git "$PHONEBOX_NEURAL_ROOT/source"
git -C "$PHONEBOX_NEURAL_ROOT/source" checkout --detach \
  5dce7e27556aef4426f5623baf6351d266a30a73
git -C "$PHONEBOX_NEURAL_ROOT/source" apply "$PHONEBOX_NEURAL_ROOT/training.patch"
"$PHONEBOX_NEURAL_ROOT/venv/bin/python" -m pip install --no-deps --no-build-isolation \
  "$PHONEBOX_NEURAL_ROOT/source"
```

Inspect and ignore only generated source build products (for example `build/`
and `deep_phonemizer.egg-info/`) before creating the source receipt. The Python
interpreter is the executable; installed DeepPhonemizer module hashes are checked
separately by the worker. Exporting the patch is also available through
`phonebox.eval.benchmark_neural.write_neural_patch(output)`. Create the receipt
with the public API, explicitly beside the environment's Python symlink rather
than beside its potentially shared resolved interpreter:

```python
from phonebox.eval.benchmark_neural import PATCH_SHA256, SOURCE_REVISION
from phonebox.eval.benchmark_provenance import write_tool_receipt

write_tool_receipt(
    ".cache/benchmarks/neural-toolchain/venv/bin/python",
    ".cache/benchmarks/neural-toolchain/source",
    declared_version="0.0.19", expected_revision=SOURCE_REVISION,
    build={"patch_sha256": PATCH_SHA256},
    output=".cache/benchmarks/neural-toolchain/venv/bin/python.provenance.json",
)
```

The runner verifies the interpreter's observed binary hash, source revision,
patch hash, every installed upstream Python module and critical dependency
versions. Retain the complete pip report, source licenses and dependency lock;
no pretrained model, raw data or external author model code is vendored.

Before any full run, profile one complete training epoch and full dev evaluation:

```bash
phonebox compare benchmark --dataset italian --system deepphonemizer \
  --neural-python .cache/benchmarks/neural-toolchain/venv/bin/python \
  --neural-device mps --neural-profile-only \
  --work-dir .cache/benchmarks/italian-neural-profile \
  --output italian-neural-profile.json
```

A profile returns accounting and resource observations, never a test score. It
gives the training subprocess no test references and performs no test decoding.
Parent dataset validation still checks every split for consistency and leakage. Remove `--neural-profile-only` and use
a new empty work directory for the full experiment. CPU is the default device;
MPS uses the author's lower-level `Trainer(device=...)` API, is seeded but not
bit deterministic, and is available only when Torch reports it usable. The
ordinary public upstream `train` entrypoint selects CPU/CUDA only. Record actual
device/platform, not a presumed speed advantage. Peak process RSS includes the
entire worker; MPS allocation fields are epoch-boundary observations, not GPU
peak measurements. Training time includes full dev selection; artifact size is
the weights/config/tokenizer checkpoint without optimizer or dictionary payload.

The accepted substantial configuration is the author's autoregressive model:
four encoder and four decoder layers, width512, feed-forward1024, four heads,
dropout0.1, batch32, Adam learning rate0.0001, warmup10000 updates and max500
epochs. Every epoch is selected with the same whole-population best-variant PER
and WER evaluator as the other systems; exact ties keep the earliest checkpoint.
Plateau learning rate halves after patience10; training stops after30 consecutive
nonimproving dev evaluations after warmup, or reports that the epoch cap did not
establish convergence. Greedy generation uses the upstream maximum100 steps,
with unterminated predictions counted. This bounded protocol is not exhaustive
neural tuning or a claim of reproducing published scores.

Training-only symbol vocabularies retain atomic phone tokens, case and NFC
spelling. Unknown inputs fail as whole words and remain in the evaluation
population; unknown dev target tokens fail preflight. Dev words with unknown
input symbols are omitted only from the unused upstream loss-loader subset,
and remain in full shared dev selection with explicit counts. The sampler's
upstream seed42 is recorded separately from the model seed1729. All supplied
training variants and partial batches are retained; nonfinite training aborts
with failed accounting rather than silently dropping a target. Model-only
held-out evaluation has dictionary lookup disabled.


Custom settings are a separate experiment, not the default neural row. Supply
one JSON object with `--neural-settings settings.json`; fields omitted from the
object retain `NeuralSettings` defaults. Unknown fields, invalid types and ranges
are rejected by `NeuralSettings.from_dict(...)`, shared with the Python API.
The same file controls profile and full training. For example, a CPU experiment
may set `{"threads": 1}`; altering architecture or stopping settings must be
disclosed with its own result. The public API accepts the validated settings
object through `settings=` or `run_benchmark(..., neural_settings=...)`.
