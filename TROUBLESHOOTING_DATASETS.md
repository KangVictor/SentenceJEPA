# Troubleshooting Dataset Loading Issues

Common issues when loading datasets from disk and how to fix them.

## Quick Diagnosis

First, inspect your dataset to understand its structure:

```bash
python scripts/inspect_dataset.py --path /path/to/your/dataset
```

This will show you:
- Dataset structure
- Available columns
- Sample data
- Recommended commands

## Common Issues

### Issue 1: "TypeError: string indices must be integers, not 'str'"

**Symptom:**
```
TypeError: string indices must be integers, not 'str'
    text = item[text_column]
```

**Cause:** Dataset structure doesn't match expected format.

**Diagnosis:**
```bash
python scripts/inspect_dataset.py --path /content/drive/MyDrive/SentenceJEPA
```

**Possible causes and fixes:**

#### Cause A: Dataset is a DatasetDict (has splits)

**Diagnosis output:**
```
DatasetDict splits: ['train', 'validation', 'test']
```

**Fix:** The code now automatically uses 'train' split, but you can manually extract:

```python
from datasets import load_from_disk

# Load
dataset = load_from_disk('/content/drive/MyDrive/SentenceJEPA')

# Check if it's a DatasetDict
if hasattr(dataset, 'keys'):
    print("Splits:", list(dataset.keys()))
    # Extract the split you want
    dataset = dataset['train']  # or 'validation', 'test'

# Save the specific split
dataset.save_to_disk('/content/drive/MyDrive/SentenceJEPA_train')
```

Then train:
```bash
python scripts/train_hf.py \
    --dataset from-disk \
    --dataset-path /content/drive/MyDrive/SentenceJEPA_train
```

#### Cause B: Wrong column name

**Diagnosis output:**
```
Column names: ['content', 'title', 'id']
WARNING: Column 'text' not found!
```

**Fix:** Use the correct column name:

```bash
python scripts/train_hf.py \
    --dataset from-disk \
    --dataset-path /content/drive/MyDrive/SentenceJEPA \
    --text-column content  # Use actual column name
```

#### Cause C: Dataset contains strings instead of dicts

**Diagnosis output:**
```
Sample type: <class 'str'>
ERROR: Sample is a string, not a dictionary!
```

**Fix:** Dataset needs to be reformatted. Create a proper dataset:

```python
from datasets import Dataset

# Load your data (whatever format it's in)
# If it's currently strings in a list:
raw_data = load_from_disk('/content/drive/MyDrive/SentenceJEPA')

# Reformat as list of dicts
data_dicts = []
for item in raw_data:
    if isinstance(item, str):
        data_dicts.append({'text': item})
    elif isinstance(item, dict):
        data_dicts.append(item)

# Create new dataset
new_dataset = Dataset.from_list(data_dicts)

# Save properly formatted dataset
new_dataset.save_to_disk('/content/drive/MyDrive/SentenceJEPA_fixed')
```

Then train:
```bash
python scripts/train_hf.py \
    --dataset from-disk \
    --dataset-path /content/drive/MyDrive/SentenceJEPA_fixed
```

#### Cause D: Dataset has only 1 sample

**Diagnosis output:**
```
Total samples: 1
```

**This is unusual.** Check what's in that one sample:

```python
from datasets import load_from_disk

dataset = load_from_disk('/content/drive/MyDrive/SentenceJEPA')
print("Length:", len(dataset))
print("First item:", dataset[0])
print("Type:", type(dataset[0]))
```

If the single sample is actually a large text, you might need to split it:

```python
# Split into multiple samples
large_text = dataset[0]['text']
paragraphs = large_text.split('\n\n')

# Create multi-sample dataset
new_dataset = Dataset.from_dict({'text': paragraphs})
new_dataset.save_to_disk('/content/drive/MyDrive/SentenceJEPA_split')
```

### Issue 2: "Column not found in dataset"

**Symptom:**
```
ValueError: Column 'text' not found in dataset
Available: ['content', 'title']
```

**Fix:** Use the correct column name:

```bash
python scripts/train_hf.py \
    --dataset from-disk \
    --dataset-path /path/to/dataset \
    --text-column content
```

### Issue 3: Empty dataset after filtering

**Symptom:**
```
Processed 0 paragraphs from HuggingFace dataset
```

**Cause:** All paragraphs filtered out (too few sentences, too short, etc.)

**Fix:** Lower the filtering thresholds:

```python
from data import load_from_disk_dataset

dataset = load_from_disk_dataset(
    dataset_path='/path/to/dataset',
    min_sentences=2,  # Lower from 3
    max_sentences=20,  # Increase from 10
)
```

Or check your data:

```python
from datasets import load_from_disk

dataset = load_from_disk('/path/to/dataset')
print("Sample texts:")
for i in range(min(3, len(dataset))):
    text = dataset[i]['text']  # or your column name
    print(f"\nSample {i}:")
    print(text[:200])
```

## Debugging Workflow

### Step 1: Inspect the dataset

```bash
python scripts/inspect_dataset.py --path /path/to/your/dataset
```

Look for:
- ✅ "Total samples" > 0
- ✅ "Column names" includes 'text' or similar
- ✅ "Sample type: <class 'dict'>"
- ✅ Sample has 'text' field

### Step 2: If issues found, fix them

**DatasetDict → Extract split:**
```python
from datasets import load_from_disk
ds = load_from_disk('/path/to/dataset')
ds['train'].save_to_disk('/path/to/dataset_train')
```

**Wrong column → Use correct one:**
```bash
--text-column your_actual_column_name
```

**Wrong structure → Reformat:**
```python
from datasets import Dataset

# Load however you can
data = load_from_disk('/path/to/dataset')

# Ensure it's list of dicts with 'text' field
formatted = [{'text': item['your_field']} for item in data]

# Save
Dataset.from_list(formatted).save_to_disk('/path/to/fixed')
```

### Step 3: Test with small sample

```python
from datasets import Dataset

# Create tiny test dataset
test_data = {
    'text': [
        'First paragraph. Has multiple sentences. Testing it works.',
        'Second paragraph here. Also multiple sentences. Good for testing.',
        'Third one. More sentences. Final test.',
    ]
}

dataset = Dataset.from_dict(test_data)
dataset.save_to_disk('./test_dataset')
```

Train on test:
```bash
python scripts/train_hf.py \
    --dataset from-disk \
    --dataset-path ./test_dataset
```

If test works, issue is with your original dataset structure.

## Correct Dataset Format

Your dataset should look like this:

```python
from datasets import Dataset

# Correct format
data = {
    'text': [
        'First paragraph text here...',
        'Second paragraph text...',
        'Third paragraph...',
        # ... more paragraphs
    ]
}

dataset = Dataset.from_dict(data)
dataset.save_to_disk('/path/to/save')
```

**Structure:**
- ✅ Dataset (not DatasetDict, unless you extract a split)
- ✅ Dict-like items (not strings)
- ✅ Has 'text' field (or specify with --text-column)
- ✅ Multiple samples (paragraphs)

## Example: Fixing Your Dataset

### Your Current Situation

```bash
python scripts/inspect_dataset.py --path /content/drive/MyDrive/SentenceJEPA
```

Based on output, choose fix:

### If it's a DatasetDict:

```python
from datasets import load_from_disk

# Load
dataset = load_from_disk('/content/drive/MyDrive/SentenceJEPA')

# Extract train split
train_dataset = dataset['train']

# Save
train_dataset.save_to_disk('/content/drive/MyDrive/SentenceJEPA_train')
```

```bash
# Train
python scripts/train_hf.py \
    --dataset from-disk \
    --dataset-path /content/drive/MyDrive/SentenceJEPA_train
```

### If it's wrong column name:

Check output of inspect_dataset.py, then:

```bash
python scripts/train_hf.py \
    --dataset from-disk \
    --dataset-path /content/drive/MyDrive/SentenceJEPA \
    --text-column actual_column_name
```

### If it's wrong structure:

```python
from datasets import load_from_disk, Dataset

# Load whatever format it's in
data = load_from_disk('/content/drive/MyDrive/SentenceJEPA')

# Reformat
paragraphs = []
for item in data:
    # Adjust based on your actual structure
    if isinstance(item, str):
        paragraphs.append({'text': item})
    elif isinstance(item, dict) and 'your_field' in item:
        paragraphs.append({'text': item['your_field']})

# Create proper dataset
new_dataset = Dataset.from_list(paragraphs)

# Save
new_dataset.save_to_disk('/content/drive/MyDrive/SentenceJEPA_fixed')
```

## Quick Reference

| Problem | Command |
|---------|---------|
| **Inspect dataset** | `python scripts/inspect_dataset.py --path /path` |
| **Wrong column** | Add `--text-column your_column` |
| **DatasetDict** | Extract split first (see above) |
| **Wrong structure** | Reformat with Dataset.from_list (see above) |
| **Test with sample** | Create test_dataset (see above) |

## Still Having Issues?

1. Run inspect_dataset.py and share the output
2. Share the first few lines of how you created/saved the dataset
3. Check if dataset loads in Python:
   ```python
   from datasets import load_from_disk
   ds = load_from_disk('/path')
   print(type(ds))
   print(ds[0] if len(ds) > 0 else "empty")
   ```

## Prevention

When creating datasets, always use this format:

```python
from datasets import Dataset

# Collect your paragraphs
paragraphs = [
    "paragraph 1 text...",
    "paragraph 2 text...",
    # ...
]

# Create dataset with 'text' field
dataset = Dataset.from_dict({'text': paragraphs})

# Verify structure
print("Length:", len(dataset))
print("Columns:", dataset.column_names)
print("Sample:", dataset[0])

# Save
dataset.save_to_disk('/path/to/save')
```

This ensures compatibility with the training script!
