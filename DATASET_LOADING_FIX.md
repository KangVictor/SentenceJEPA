# Fix: Dataset Loading Error

## Your Error

```
TypeError: string indices must be integers, not 'str'
    text = item[text_column]
```

## Quick Fix (3 Steps)

### Step 1: Inspect your dataset

```bash
python scripts/inspect_dataset.py --path /content/drive/MyDrive/SentenceJEPA
```

This will show you exactly what's wrong.

### Step 2: Apply the fix based on what you see

#### If you see "DatasetDict splits: ['train', ...]"

Your dataset has multiple splits. Extract the one you need:

```python
from datasets import load_from_disk

# Load
dataset = load_from_disk('/content/drive/MyDrive/SentenceJEPA')

# Extract train split
train_data = dataset['train']

# Save it
train_data.save_to_disk('/content/drive/MyDrive/SentenceJEPA_train')
```

Then train:
```bash
python scripts/train_hf.py \
    --dataset from-disk \
    --dataset-path /content/drive/MyDrive/SentenceJEPA_train
```

#### If you see "Column 'text' not found!"

Your dataset uses a different column name. Use the correct one shown in the inspect output:

```bash
python scripts/train_hf.py \
    --dataset from-disk \
    --dataset-path /content/drive/MyDrive/SentenceJEPA \
    --text-column YOUR_ACTUAL_COLUMN_NAME
```

#### If you see "Sample type: <class 'str'>" (strings not dicts)

Your dataset has wrong structure. Fix it:

```python
from datasets import load_from_disk, Dataset

# Load
data = load_from_disk('/content/drive/MyDrive/SentenceJEPA')

# Check what you have
print("First item:", data[0])
print("Type:", type(data[0]))

# If it's strings, wrap in dicts:
if isinstance(data[0], str):
    # Your data is list of strings, convert to list of dicts
    fixed_data = [{'text': item} for item in data]
else:
    # Might be dicts with different field name
    # Adjust 'your_field_name' to match your actual field
    fixed_data = [{'text': item['your_field_name']} for item in data]

# Create proper dataset
new_dataset = Dataset.from_list(fixed_data)

# Save
new_dataset.save_to_disk('/content/drive/MyDrive/SentenceJEPA_fixed')
```

Then train:
```bash
python scripts/train_hf.py \
    --dataset from-disk \
    --dataset-path /content/drive/MyDrive/SentenceJEPA_fixed
```

### Step 3: Train

Once fixed, train normally:

```bash
python scripts/train_hf.py \
    --dataset from-disk \
    --dataset-path /content/drive/MyDrive/SentenceJEPA_fixed \
    --device cuda
```

## Most Likely: DatasetDict Issue

The most common cause is having a DatasetDict (with 'train', 'test', 'validation' splits).

**Quick fix in Colab:**

```python
from datasets import load_from_disk

# Load your dataset
dataset = load_from_disk('/content/drive/MyDrive/SentenceJEPA')

# Check if it's a DatasetDict
if hasattr(dataset, 'keys'):
    print("This is a DatasetDict with splits:", list(dataset.keys()))

    # Get the train split
    if 'train' in dataset:
        train = dataset['train']
    else:
        # Use first split
        split_name = list(dataset.keys())[0]
        train = dataset[split_name]

    # Save the train split
    train.save_to_disk('/content/drive/MyDrive/SentenceJEPA_train')
    print("✓ Saved train split")
else:
    print("This is already a regular Dataset")
```

Then train on the extracted split:

```bash
!python scripts/train_hf.py \
    --dataset from-disk \
    --dataset-path /content/drive/MyDrive/SentenceJEPA_train \
    --device cuda
```

## Test with Sample Data

To verify everything works, test with a simple dataset first:

```python
from datasets import Dataset

# Create test data
test_data = {
    'text': [
        'First paragraph with multiple sentences. This is sentence two. And three.',
        'Second paragraph here. Also has multiple sentences. Testing continues.',
        'Third paragraph. More sentences here. Final test paragraph.',
    ]
}

# Create dataset
test_dataset = Dataset.from_dict(test_data)

# Save
test_dataset.save_to_disk('/content/drive/MyDrive/test_dataset')

print("✓ Test dataset created")
```

Train on test:

```bash
!python scripts/train_hf.py \
    --dataset from-disk \
    --dataset-path /content/drive/MyDrive/test_dataset \
    --device cuda
```

If this works, the issue is with your original dataset structure.

## Still Stuck?

Run the inspector and share the output:

```bash
python scripts/inspect_dataset.py --path /content/drive/MyDrive/SentenceJEPA
```

The output will show exactly what's wrong and how to fix it!

---

**See also:**
- [TROUBLESHOOTING_DATASETS.md](TROUBLESHOOTING_DATASETS.md) - Complete troubleshooting guide
- [OFFLINE_TRAINING.md](OFFLINE_TRAINING.md) - Offline training documentation
