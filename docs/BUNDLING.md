# Bundling G2P for Deployment

Create a self-contained Python G2P executable with the model embedded using
`phonebox bundle`.

## Quick Start

```bash
# Train a model
phonebox model build en_US dictionary.txt -o model.g2p.gz --remove-stress

# Bundle for deployment
phonebox bundle model.g2p.gz -o g2p.py
```

## Python Bundle

```bash
phonebox bundle model.g2p.gz -o g2p.py
```

### CLI Usage

```bash
python g2p.py hello world
# hello   HH EH L OW
# world   W ER L D
```

### Library Usage

```python
from g2p import G2PPredictor

g2p = G2PPredictor.from_embedded()
phones = g2p.pronounce('hello')  # ['HH', 'EH', 'L', 'OW']

# Batch processing
for word in ['hello', 'world', 'test']:
    print(f"{word}: {' '.join(g2p.pronounce(word))}")
```

### FastAPI Service

```python
from fastapi import FastAPI
from g2p import G2PPredictor

app = FastAPI()
g2p = G2PPredictor.from_embedded()

@app.get("/pronounce/{word}")
def pronounce(word: str):
    phones = g2p.pronounce(word)
    return {"word": word, "phones": phones}
```

## Deployment Scenarios

### Docker Container

```dockerfile
FROM python:3.12-slim

COPY g2p.py /app/
WORKDIR /app

# No pip install needed - g2p.py is self-contained
CMD ["python3", "g2p.py"]
```

### AWS Lambda

```python
# lambda_function.py
from g2p import G2PPredictor

# Load once at cold start
g2p = G2PPredictor.from_embedded()

def handler(event, context):
    word = event['word']
    phones = g2p.pronounce(word)
    return {'word': word, 'phones': phones}
```

Package just `g2p.py` - no dependencies needed.

## Bundle Contents

Each bundled file contains:

1. **CART predictor code** (from cartlet) - tree traversal, model loading
2. **G2P wrapper code** - word vectorization, phoneme handling
3. **Embedded model** - compressed and encoded

Typical size is ~100 KB with no runtime dependencies beyond Python stdlib.

## Best Practices

1. **Bundle once, deploy anywhere** - The bundled file is self-contained
2. **Load once per process** - Use `G2PPredictor.from_embedded()` at startup
3. **Reuse the instance** - Don't recreate for each word
4. **Version your bundles** - Track which model version is embedded

## Comparison: Library vs Bundle

### Using phonebox library (development)

```python
from phonebox import G2P
g2p = G2P(model="model.g2p.gz")
phones = g2p.pronounce("hello")
```

- Requires phonebox installed
- Full API available
- Good for development and training

### Using bundled file (production)

```python
from g2p import G2PPredictor
g2p = G2PPredictor.from_embedded()
phones = g2p.pronounce("hello")
```

- No dependencies beyond Python stdlib
- Single file deployment
- Good for production, serverless, embedded Python runtimes

## Performance

| Operation | Time |
|-----------|------|
| Load model | ~50ms |
| Pronounce word | < 1ms |
| Memory usage | ~5-10 MB |

Bundle files are optimized for fast loading and low memory usage.
