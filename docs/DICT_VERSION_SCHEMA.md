# Dictionary content identifiers

Decision-tree models store `dict_hash`, an eight-character content identifier
computed while loading the training dictionary. It is not a Git revision or a
user-assigned version name.

`DecisionTree.load_prondict(infile)` hashes each input line after stripping its
leading and trailing whitespace, encoding it as UTF-8, and concatenating it
without a separator. It stores the first eight hexadecimal characters of that
SHA-256 digest. Line order, internal whitespace, comments, and duplicate input
rows can affect this identifier even when preprocessing produces equivalent
training examples. It is not a digest of the original file bytes or a unique
identity for the complete trained model.

## Loading and exporting

```python
from io import StringIO
from phonebox import DecisionTree

model = DecisionTree(locale="en_US", phoneset_name="cmu", parallel_align=False)
model.load_prondict(StringIO("cat K AE T\ndog D AO G\n"))
print(model.dict_hash)
model.align()
model.train(prune=False)
model.export("model.g2p.gz")

loaded = DecisionTree(model="model.g2p.gz")
assert loaded.pronounce("cat") == model.pronounce("cat")
```

The loader accepts only the dictionary stream. It does not accept
`dict_source_path` or `dict_version_suffix`, inspect Git history, or infer
version suffixes from filenames. Earlier versions of this document described
those unimplemented behaviors as available; the example above reflects the
current API.

## Reproducible builds

Keep the source repository and revision, a full file digest, training options,
Phonebox revision, and dependency versions in a separate build record.
`dict_hash` is useful for comparing loaded input identifiers, but its short
digest does not replace that record. See the pinned source and environment
record in [the CMUdict comparison](CMUDICT_COMPARISON.md).

Dictionary download manifests are separate from model metadata. Creating a
manifest does not make the model loader copy its source revision into
`dict_hash`.
