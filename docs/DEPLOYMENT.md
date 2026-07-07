# G2P Deployment Guide

## Overview

Phonebox separates **training** (development) from **inference** (production):

| Mode | Use Case | Approach |
|------|----------|----------|
| Training | Build models from dictionaries | Full phonebox library |
| Inference | Pronounce words in production | Bundled standalone Python file |

## Creating Deployable Bundles

### Step 1: Train a Model

```bash
# Fetch dictionary and train
phonebox dict fetch cmudict
phonebox model build en_US data/cmudict/cmudict.dict \
  -o model.g2p.gz \
  --remove-stress
```

### Step 2: Bundle for Deployment

```bash
phonebox bundle model.g2p.gz -o g2p.py
```

### Step 3: Deploy

Copy the single bundled file to your deployment target. No installation needed
beyond the Python standard library.

## Deployment Scenarios

### Web Service (Python/FastAPI)

```python
# app.py
from fastapi import FastAPI
from g2p import G2PPredictor

app = FastAPI()
g2p = G2PPredictor.from_embedded()

@app.get("/pronounce/{word}")
def pronounce(word: str):
    return {"word": word, "phones": g2p.pronounce(word)}
```

```dockerfile
FROM python:3.12-slim
RUN pip install fastapi uvicorn
COPY g2p.py app.py /app/
WORKDIR /app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0"]
```

### AWS Lambda

```python
# lambda_function.py
from g2p import G2PPredictor

g2p = G2PPredictor.from_embedded()

def handler(event, context):
    word = event.get('word', '')
    return {
        'word': word,
        'phones': g2p.pronounce(word),
        'pronunciation': ' '.join(g2p.pronounce(word))
    }
```

Package: Just zip `g2p.py` and `lambda_function.py`. No dependencies.

### Docker Container

```dockerfile
FROM python:3.12-slim

COPY g2p.py /app/
WORKDIR /app

# Self-contained - no pip install needed
ENTRYPOINT ["python3", "g2p.py"]
```

```bash
docker build -t g2p .
docker run g2p hello world
```

### Command-Line Tool

```bash
python g2p.py hello world testing
# hello   HH EH L OW
# world   W ER L D
# testing T EH S T IH NG
```

## Library Usage

```python
from g2p import G2PPredictor

g2p = G2PPredictor.from_embedded()

# Single word
phones = g2p.pronounce('hello')  # ['HH', 'EH', 'L', 'OW']

# Batch
words = ['hello', 'world', 'python']
for word in words:
    print(f"{word}: {' '.join(g2p.pronounce(word))}")

# Callable interface
phones = g2p('hello')  # Same as g2p.pronounce('hello')
```

## Performance

| Metric | Value |
|--------|-------|
| Model load time | ~50ms |
| Inference time | < 1ms per word |
| Memory usage | ~5-10 MB |
| Bundle size | ~100 KB |

## Best Practices

1. **Load once, reuse** - Create `G2PPredictor` once at startup
2. **Bundle per environment** - Different models for different locales
3. **Version bundles** - Track which model is embedded
4. **Test before deploy** - Verify pronunciation accuracy

## Development vs Production

### Development (with phonebox)

```python
from phonebox import G2P

# Full API with model path
g2p = G2P(model="model.g2p.gz")
phones = g2p.pronounce("hello")
confidence = g2p.pronounce_with_confidence("hello")
nbest = g2p.pronounce_nbest("hello", n=3)
```

### Production (bundled)

```python
from g2p import G2PPredictor

# Embedded model - no file path needed
g2p = G2PPredictor.from_embedded()
phones = g2p.pronounce("hello")
```

## Deployment Checklist

- [ ] Train model with appropriate options (locale, remove-stress)
- [ ] Test model accuracy before bundling
- [ ] Bundle (`phonebox bundle model.g2p.gz -o g2p.py`)
- [ ] Test bundled file works standalone
- [ ] Deploy single bundled file
- [ ] Verify in production environment
