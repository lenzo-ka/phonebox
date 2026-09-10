# Dictionary processing

`Dictionary.process()` reads pronunciation dictionaries through Phonebox's shared
line parser. A line contains a word, optional numeric variant suffix, and phones
separated by arbitrary whitespace; `#` starts an inline comment.

Processing applies lowercase and an optional literal phone mapping or Python
`phone_transform`, then optional phoneset-specific stress removal. Deduplication
happens after those operations.
For each word, the first distinct pronunciation is written under the bare spelling
and later distinct pronunciations use dense `(2)`, `(3)`, … suffixes in first
occurrence order.

The CLI accepts a JSON mapping whose values are one phone or a nonempty list:

```json
{"IY1": "IY", "IY2": "IY", "AXR": ["AH", "R"]}
```

```console
phonebox dict process input.dict -o output.dict --phone-map phones.json
```

If two variants become identical after mapping, only the first remains and later
distinct variants are renumbered without gaps.

The library accepts the same mapping directly:

```python
from phonebox import Dictionary

processed = Dictionary("input.dict").process(
    phone_mapping={"IY1": "IY", "IY2": "IY", "AXR": ["AH", "R"]},
    output="output.dict",
)
```

For a context-dependent mapping, pass `phone_transform` instead: a callable that
receives a list of phones and returns a nonempty list of phones. Stress stripping
is optional (`remove_stress=True` or `--remove-stress`) and uses the selected
phoneset's syntax (`phoneset="cmu"` or `--phoneset cmu`). Unknown phoneset tags
preserve phone tokens.
Input and output must be different files.
