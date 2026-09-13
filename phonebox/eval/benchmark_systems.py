"""One identity and presentation registry for supported benchmark adapters."""

SYSTEM_LABELS = {
    "cart": "Phonebox CART",
    "multigram": "Phonebox n:m",
    "sequitur": "Sequitur",
    "phonetisaurus": "Phonetisaurus",
    "deepphonemizer": "DeepPhonemizer autoregressive",
}
SYSTEMS = tuple(SYSTEM_LABELS)
