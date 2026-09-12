# Upgrading and preparing a release

## Upgrading from 0.1.0

0.2.0 is a new alpha minor line. Review integrations rather than assuming API,
CLI, or saved-model compatibility. Pin `phonebox==0.2.0` when adopting this line.
The [changelog](../CHANGELOG.md) records the changes since `v0.1.0`.

- Use `phonebox.train_g2p` or `train_g2p_from_config` for dictionary-to-CART
  training and consume their `TrainingResult` (`model`, `metrics`, and artifact
  paths). `G2P.train` wraps the same workflow and returns a `G2P` predictor.
  `phonebox train` and config training share this workflow. Replace
  `phonebox model build --config ...` with `phonebox train --config ...`.
  Prepared alignments and vectors remain separate low-level operations under
  `phonebox model`.
- `G2P.train` changes its default phoneset from CMU to IPA and its default
  pruning from disabled to enabled with a 5% validation split. Select the
  phoneset explicitly for your data. The primary training default is
  `ipa`, with stress preserved, native serial training, stored leaf distributions,
  pruning enabled, a 5% validation split, and no held-out test split. Pass
  `phoneset="cmu"` / `--phoneset cmu` for CMU stress/join syntax. Stress removal
  is optional; request it with `remove_stress=True` / `--remove-stress`.
- Base workflows use the native CART trainer. Install `phonebox[sklearn]` to
  opt into scikit-learn, or `phonebox[config,sklearn]` for YAML presets selecting
  it. The old `icu` extra is removed: icukit already supplies the `icu` namespace;
  installing PyICU alongside it is not a supported alternative.
- Keep the multigram `<stem>.units.json` and `<stem>.lm.json` files together.
  Export/load uses a model stem such as `model.g2p.gz`; that stem need not exist
  as a separate file. New models embed their preprocessing; current
  train/save/load round trips retain it.
  Retrain and re-export old artifacts when adopting the new line; compatibility
  with snapshotless 0.1.0 artifacts is not a release requirement.
- Standalone bundling supports decision trees. The full package uses cartlet
  and ICU; generated bundles provide the standard-library-only deployment path.
  Unsupported portable preprocessing is rejected at export rather than silently
  changed. See [bundling](BUNDLING.md).
- Use shared locale APIs (`canonical_locale`, `resolve_locale`) instead of
  manually changing case or regions. Bare language codes can resolve a spelling
  resource; exact exemplar and trained-model identities remain distinct.
  ICU exemplars supply orthography, not phonemes. See [exemplars](EXEMPLARS.md).
- Replace repository script invocations with installed CLI commands and package
  APIs. Run `phonebox --help` and command-specific `--help` for the current
  surface; use `phonebox.cli.main.main(argv)` for an in-process CLI status.
  See [library and CLI workflows](WORKFLOWS.md) for `train_multigram`,
  `discover_joins`, and structured lexicon validation entry points.
  Comparison paths use `PHONEBOX_LEXICON_DIR` and `PHONEBOX_MODEL_DIR`.
- Use `Dictionary.process` / `phonebox dict process` for mapped dictionaries.
  Mapping precedes optional stress removal, then final deduplication emits bare
  first spellings and dense `(2)`, `(3)`, … variants. Input and output must be
  different files. See [dictionary processing](DICTIONARY_PROCESSING.md).

The [CMUdict report](CMUDICT_COMPARISON.md) is measured at its recorded source
revision and dependency versions. Its CART benchmark helper is unpruned, unlike
primary training's default. Updating documentation or releasing 0.2.0 does not
make those historical metrics a measurement of the release commit. Rerunning
on newer code creates a new snapshot.

## Installation scope

The full runtime needs compatible cartlet and ICU dependencies. The pinned
[icukit-pyicu 78.3.0 files](https://pypi.org/pypi/icukit-pyicu/78.3.0/json)
cover macOS ARM64 and Linux x86_64/aarch64, with no Windows/Intel macOS wheels
or source distribution. Full installation support follows dependency wheel
availability for the selected Python/platform. Standard-library-only bundles
remain a separate inference deployment path, not evidence of full-package
installation support on every platform.

## Release preparation

The 0.2.0 changelog is unreleased until a release is explicitly authorized.
Preparation does not create a tag, publish a GitHub release, or upload to PyPI.

1. Work on a release branch and update `pyproject.toml` and
   `phonebox.__version__` together. Keep the changelog date unset until the cut.
2. Install `.[dev]` and run the checks defined in `.github/workflows/ci.yml`:
   pytest (including packaging tests), Ruff, Ruff format, mypy, the pinned
   exemplar generation check, and the committed CMUdict JSON/Markdown check.
   Build with `python -m build` and check distributions with `twine check dist/*`.
3. Confirm the wheel and sdist agree on version, package data, type marker, and
   complete license notices. The packaging regression builds a wheel from the
   extracted sdist. Independently install each distribution outside the source
   tree and check `phonebox --version`, imports, CLI help, small CART/multigram
   training/save/load, and tree bundle execution under `python -S`.
4. Record exact commit and artifact checksums, test counts and skips, and CI
   outcomes. Python 3.11–3.13 are the required CI matrix; 3.14 is currently
   experimental (`continue-on-error`). Optional public datasets/assets may be
   absent locally; name skipped coverage rather than treating it as passed.
5. Obtain independent review and green checks on the immutable final PR head
   before merging. Preserve benchmark source references and JSON provenance.

## Authorized release cut

Only after the owner explicitly authorizes publication, set the changelog date,
review the final version commit, rerun release checks, and create `v0.2.0` at the
verified commit. Publishing the GitHub release triggers
[`.github/workflows/publish.yml`](https://github.com/lenzo-ka/phonebox/blob/main/.github/workflows/publish.yml): it checks
release-tag/package-version agreement, rebuilds the wheel and sdist from that
release, runs `twine check`, and publishes the resulting artifacts to PyPI via
trusted publishing in the `pypi` GitHub environment (`id-token: write`). Confirm
the environment and PyPI trusted-publisher configuration before publishing the
GitHub release; publication is the trigger, not a draft release. Local readiness
artifacts are checks, not the files uploaded by this workflow. Record the workflow
result, published artifact hashes, and URLs. These steps remain pending during
release preparation.

Phonebox code is BSD 2-Clause. Packaged ICU/CLDR-derived exemplar data carries
Unicode-3.0; ship `LICENSE`, `LICENSE-UNICODE`, and `THIRD_PARTY_NOTICES.md`
with distributions. See [third-party notices](../THIRD_PARTY_NOTICES.md).
