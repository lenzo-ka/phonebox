# Dictionary Version Schema

This document describes how dictionary files are versioned and tracked in the g2p system.

## Version Identifier Format

All dictionary sources are assigned a `version_id` in one of these formats:

### 1. Git Repository Source
```
git:<commit-hash>[-<suffix>]
```
- **When**: Dictionary is in a git repository (e.g., CMUdict)
- **Examples**:
  - `git:a1b2c3d4` - Original from git
  - `git:a1b2c3d4-v1` - First edited version
  - `git:a1b2c3d4-custom` - Custom modifications
  - `git:a1b2c3d4-patched` - Patched version
- **Properties**:
  - 8-character short commit hash
  - Optional suffix for tracking local modifications
  - Reproducible from git history (base version)
  - Includes git remote URL in metadata
  - Content hash stored to detect modifications

### 2. URL Download Source
```
hash:<content-hash>[-<suffix>]
```
- **When**: Dictionary downloaded from a URL
- **Examples**:
  - `hash:5f6e7d8c` - Original download
  - `hash:5f6e7d8c-fixed` - With corrections
  - `hash:5f6e7d8c-v2` - Version 2 of edits
- **Properties**:
  - 8-character SHA256 hash of file contents
  - Optional suffix for tracking modifications
  - Source URL stored in `.source_url` marker file
  - Detects file changes

### 3. Local File Source
```
hash:<content-hash>[-<suffix>]
```
- **When**: Dictionary is a local file not in git
- **Examples**:
  - `hash:9a0b1c2d` - Original
  - `hash:9a0b1c2d-custom` - Custom version
- **Properties**:
  - 8-character SHA256 hash of file contents
  - Optional suffix for tracking versions
  - No git or URL information available

## Metadata Schema

### Dictionary Metadata
```json
{
  "path": "data/cmudict/cmudict.dict",
  "size_bytes": 3621079,
  "modified_time": 1700000000.0,
  "source_type": "git|url|local",
  "version_id": "git:a1b2c3d4-v1",

  // Version tracking (for modified versions)
  "base_version": "git:a1b2c3d4",
  "version_suffix": "v1",
  "content_hash": "7e8f9d0a",

  // Git-specific fields
  "git_commit": "a1b2c3d4",
  "git_remote": "https://github.com/cmusphinx/cmudict.git",

  // URL-specific fields
  "source_url": "https://example.com/dict.txt",

  // Always present
  "content_hash": "7e8f9d0a"
}
```

### Model Metadata
Trained models embed the dictionary version:

```json
{
  "locale": "en_US",
  "dict_hash": "git:a1b2c3d4",
  "export_time": "2024-01-01 12:00:00 UTC",
  "exceptions": {...},
  ...
}
```

## Use Cases

### 1. Model Versioning
Models include `dict_hash` so you know which dictionary version was used:
```python
dt = DecisionTree(model="model.g2p.gz")
print(dt.dict_hash)  # "git:a1b2c3d4"
```

### 2. Rebuild Detection
Check if dictionary changed and model needs retraining:
```python
from phonebox import DecisionTree

dt = DecisionTree(model="model.g2p.gz")
print(f"Model dict_hash: {dt.dict_hash}")
```

### 3. Reproducibility
Given a model with `dict_hash = "git:a1b2c3d4"`:
- For git sources: `git checkout a1b2c3d4` to get exact version
- For hash sources: Compare hash to verify file unchanged

## Implementation

### Training
When training a model:

**Original dictionary:**
```python
with open('cmudict.dict') as f:
    dt.load_prondict(f, dict_source_path='cmudict.dict')
    # Result: dict_hash = "git:a1b2c3d4"

dt.export('model.g2p.gz')
```

**Modified dictionary with version suffix:**
```python
with open('cmudict_edited.dict') as f:
    dt.load_prondict(
        f,
        dict_source_path='cmudict_edited.dict',
        dict_version_suffix='v1'  # Track this as version 1
    )
    # Result: dict_hash = "git:a1b2c3d4-v1"
    # Includes base_version and content_hash metadata

dt.export('model_v1.g2p.gz')
```

**Custom local dictionary:**
```python
with open('my_custom.dict') as f:
    dt.load_prondict(
        f,
        dict_source_path='my_custom.dict',
        dict_version_suffix='production'
    )
    # Result: dict_hash = "hash:5f6e7d8c-production"
```

### Manifest
The manifest file tracks all dictionaries:
```json
{
  "generated": "2024-01-01T12:00:00Z",
  "dictionaries": [
    {
      "language": "en_US",
      "source": "cmudict",
      "version_id": "git:a1b2c3d4",
      "git_commit": "a1b2c3d4",
      "git_remote": "https://github.com/cmusphinx/cmudict.git"
    }
  ]
}
```

## Benefits

1. **Reproducibility**: Know exactly which dictionary version was used
2. **Change Detection**: Automatically detect when retraining needed
3. **Flexibility**: Works with git repos, URLs, and local files
4. **Traceability**: Track dictionary provenance
5. **CI/CD**: Automate model rebuilds when source changes
