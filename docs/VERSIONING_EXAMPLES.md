# Dictionary Versioning Examples

## Use Case 1: Original Dictionary

You download CMUdict and train a model:

```bash
# Train with original dictionary
phonebox model build en_US data/cmudict/cmudict.dict -o model.g2p.gz

# Model embeds: dict_hash = "git:a1b2c3d4"
```

Later, you can verify which version was used:
```python
dt = DecisionTree(model="model.g2p.gz")
print(dt.dict_hash)  # "git:a1b2c3d4"
```

## Use Case 2: Custom Edits - Version 1

You fix some entries in CMUdict:

```bash
# Edit the dictionary
cp data/cmudict/cmudict.dict data/cmudict/cmudict_edited.dict
vim data/cmudict/cmudict_edited.dict  # Make your fixes

# Train with version suffix
phonebox model build en_US data/cmudict/cmudict_edited.dict \
  -o model_v1.g2p.gz

# Model embeds: dict_hash = "git:a1b2c3d4-v1"
```

The model now tracks:
- Base version: `git:a1b2c3d4` (original CMUdict commit)
- Your version: `v1`
- Content hash: `7e8f9d0a` (your edited version)

## Use Case 3: Production Dictionary

You maintain a curated dictionary:

```bash
# Initial version
phonebox model build en_US my_dict.txt -o production.g2p.gz

# First update
vim my_dict.txt  # Add 100 new words
phonebox model build en_US my_dict.txt -o production_v2.g2p.gz

# Second update
vim my_dict.txt  # Fix pronunciations
phonebox model build en_US my_dict.txt -o production_v3.g2p.gz
```

Now you can track exactly which version each model uses:
- `production.g2p.gz`: `hash:abc123-prod`
- `production_v2.g2p.gz`: `hash:def456-prod-v2`
- `production_v3.g2p.gz`: `hash:789xyz-prod-v3`

## Use Case 4: Patching Upstream

You're using CMUdict but need to patch some entries:

```bash
# Get original CMUdict
cd data/cmudict
git log -1 --format="%h"  # Shows: a1b2c3d4

# Create patched version
cp cmudict.dict cmudict_patched.dict
vim cmudict_patched.dict  # Fix specific words

# Train patched model
phonebox model build en_US cmudict_patched.dict \
  -o model_patched.g2p.gz

# Model embeds: dict_hash = "git:a1b2c3d4"
```

Later, when CMUdict updates:
```bash
cd data/cmudict
git pull  # Updates to commit b2c3d4e5

# Re-apply your patches
cp cmudict.dict cmudict_patched.dict
# ... apply same patches ...

# Retrain
phonebox model build en_US cmudict_patched.dict \
  -o model_patched_new.g2p.gz

# Model embeds: dict_hash = "git:b2c3d4e5-patched"
```

Now you can compare:
- Old: `git:a1b2c3d4-patched`
- New: `git:b2c3d4e5-patched`

Both use `patched` suffix, but base version changed!

## Use Case 5: Multiple Variants

You maintain several variants for different use cases:

```bash
# Formal speech variant
phonebox model build en_US dict.txt -o model_formal.g2p.gz

# Casual speech variant
phonebox model build en_US dict_casual.txt -o model_casual.g2p.gz

# Regional dialect
phonebox model build en_US dict_regional.txt -o model_regional.g2p.gz
```

## Checking Version Compatibility

```python
from phonebox import DecisionTree

# Load model and check its dictionary hash
dt = DecisionTree(model="model.g2p.gz")
print(f"Model trained on dict_hash: {dt.dict_hash}")
```

## Best Practices

1. **Original dictionaries**: No suffix
   - `git:a1b2c3d4` or `hash:5f6e7d8c`

2. **Sequential edits**: Use v1, v2, v3...
   - `git:a1b2c3d4-v1`
   - `git:a1b2c3d4-v2`

3. **Purpose-based**: Use descriptive names
   - `git:a1b2c3d4-patched`
   - `git:a1b2c3d4-custom`
   - `hash:abc123-production`

4. **Environment-based**: Track by deployment
   - `hash:abc123-dev`
   - `hash:abc123-staging`
   - `hash:abc123-prod`

5. **Date-based**: For periodic updates
   - `hash:abc123-2024-01`
   - `hash:abc123-2024-02`
