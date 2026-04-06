# Using Your Trained Model

Complete guide on how to use your trained Sentence JEPA model.

## What Can You Do With It?

Your trained model has learned to create **contextualized sentence embeddings** - 512-dimensional vectors that capture:
- Sentence meaning
- Context from surrounding sentences
- Position in discourse flow

### Use Cases

1. **Sentence Similarity** - Find similar sentences
2. **Semantic Search** - Search for sentences by meaning
3. **Sentence Clustering** - Group related sentences
4. **Document Understanding** - Analyze document structure
5. **Retrieval** - Find relevant sentences in large corpora

## Quick Start

### Load Your Trained Model

```python
import torch
from models import HierarchicalSentenceJEPA

# Create model (match your training config)
model = HierarchicalSentenceJEPA(
    sentence_encoder_name='roberta-base',
    d_model=768,
    num_layers=4,
    projection_output_dim=512,
)

# Load trained weights
checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### Example Script

```bash
python scripts/use_model.py \
    --checkpoint checkpoints/best_model.pt \
    --config configs/base.yaml
```

This will:
- Load your trained model
- Encode example paragraphs
- Find similar sentences
- Show how to use the embeddings

## Detailed Usage

### 1. Generate Sentence Embeddings

```python
from scripts.use_model import load_model, encode_paragraph
from transformers import AutoTokenizer
import yaml

# Load config and model
with open('configs/base.yaml') as f:
    config = yaml.safe_load(f)

model = load_model('checkpoints/best_model.pt', config)
tokenizer = AutoTokenizer.from_pretrained('roberta-base')

# Encode a paragraph
paragraph = """
Machine learning is amazing. It can solve complex problems.
Neural networks are powerful tools.
"""

embeddings, sentences = encode_paragraph(model, tokenizer, paragraph)

print(f"Encoded {len(sentences)} sentences")
print(f"Embedding shape: {embeddings.shape}")  # [num_sentences, 512]

# Use embeddings
for i, (sent, emb) in enumerate(zip(sentences, embeddings)):
    print(f"Sentence {i+1}: {sent}")
    print(f"  Embedding (first 5 dims): {emb[:5]}")
```

**Output:**
```
Encoded 3 sentences
Embedding shape: (3, 512)
Sentence 1: Machine learning is amazing.
  Embedding (first 5 dims): [0.23, -0.15, 0.87, -0.42, 0.56]
...
```

### 2. Find Similar Sentences

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Encode two paragraphs
emb1, sent1 = encode_paragraph(model, tokenizer, paragraph1)
emb2, sent2 = encode_paragraph(model, tokenizer, paragraph2)

# Compute similarity
similarity = cosine_similarity(emb1, emb2)  # [len(sent1), len(sent2)]

# Find most similar pair
i, j = np.unravel_index(similarity.argmax(), similarity.shape)
print(f"Most similar:")
print(f"  '{sent1[i]}'")
print(f"  '{sent2[j]}'")
print(f"  Similarity: {similarity[i, j]:.4f}")
```

### 3. Semantic Search

```python
def search_sentences(query_paragraph, corpus_paragraphs, top_k=5):
    """
    Search for sentences similar to query in a corpus.

    Args:
        query_paragraph: Query text
        corpus_paragraphs: List of corpus paragraphs
        top_k: Number of results

    Returns:
        List of (paragraph_idx, sentence_idx, sentence, score) tuples
    """
    # Encode query
    query_emb, query_sents = encode_paragraph(model, tokenizer, query_paragraph)

    # Use first sentence of query as query embedding
    query_vec = query_emb[0]  # [512]

    # Search corpus
    results = []
    for para_idx, para in enumerate(corpus_paragraphs):
        corpus_emb, corpus_sents = encode_paragraph(model, tokenizer, para)

        # Compute similarity to each sentence
        for sent_idx, sent_emb in enumerate(corpus_emb):
            sim = cosine_similarity([query_vec], [sent_emb])[0, 0]
            results.append((para_idx, sent_idx, corpus_sents[sent_idx], sim))

    # Sort by similarity
    results.sort(key=lambda x: x[3], reverse=True)

    return results[:top_k]


# Use it
query = "What is machine learning?"
corpus = [
    "Machine learning is a method of data analysis. It automates analytical model building.",
    "Deep learning is a subset of machine learning. It uses neural networks with many layers.",
    "Computer vision enables machines to see. It's used in self-driving cars.",
]

results = search_sentences(query, corpus, top_k=3)

print("Search results:")
for i, (para_idx, sent_idx, sentence, score) in enumerate(results, 1):
    print(f"{i}. Score: {score:.4f}")
    print(f"   {sentence}")
```

### 4. Cluster Sentences

```python
from sklearn.cluster import KMeans
import numpy as np

# Collect sentences from multiple paragraphs
all_embeddings = []
all_sentences = []

for paragraph in paragraphs:
    emb, sents = encode_paragraph(model, tokenizer, paragraph)
    all_embeddings.append(emb)
    all_sentences.extend(sents)

# Stack embeddings
embeddings_matrix = np.vstack(all_embeddings)  # [total_sentences, 512]

# Cluster
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(embeddings_matrix)

# Show clusters
for cluster_id in range(n_clusters):
    print(f"\nCluster {cluster_id}:")
    cluster_sentences = [s for s, l in zip(all_sentences, labels) if l == cluster_id]
    for sent in cluster_sentences[:3]:  # Show first 3
        print(f"  - {sent}")
```

### 5. Build a Sentence Index

For large-scale retrieval:

```python
import faiss  # pip install faiss-cpu or faiss-gpu

# Build index
dimension = 512
index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)

# Add embeddings (normalize first for cosine similarity)
normalized_emb = embeddings_matrix / np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
index.add(normalized_emb.astype('float32'))

# Search
query_vec = query_emb[0]  # Your query embedding
query_norm = query_vec / np.linalg.norm(query_vec)
k = 10

distances, indices = index.search(query_norm.reshape(1, -1).astype('float32'), k)

print("Top 10 most similar sentences:")
for i, (idx, dist) in enumerate(zip(indices[0], distances[0]), 1):
    print(f"{i}. Similarity: {dist:.4f}")
    print(f"   {all_sentences[idx]}")
```

## What the Model Outputs

### Embedding Properties

The 512-dimensional embeddings have these properties:

1. **Contextualized** - Same sentence in different contexts has different embeddings
2. **Semantic** - Similar meaning = similar embeddings
3. **Normalized** - Unit vectors (after normalization)
4. **Dense** - All 512 dimensions are meaningful

### Comparison to Sentence-BERT

| Property | Sentence-BERT | Sentence JEPA (yours) |
|----------|---------------|----------------------|
| Context-aware | ❌ No | ✅ Yes |
| Same sentence always same embedding | ✅ Yes | ❌ No (context-dependent) |
| Good for static embeddings | ✅ Yes | ❌ No |
| Good for context-aware tasks | ❌ No | ✅ Yes |
| Paragraph-level understanding | ❌ Limited | ✅ Yes |

**When to use yours:**
- When context matters (same sentence, different meanings)
- Document-level understanding
- Discourse analysis

**When to use Sentence-BERT:**
- Static sentence embeddings
- Caching embeddings
- When context doesn't matter

## Performance Tips

### Batch Processing

Process multiple paragraphs efficiently:

```python
def encode_paragraphs_batch(paragraphs, batch_size=16):
    """Encode multiple paragraphs in batches."""
    all_embeddings = []
    all_sentences = []

    for i in range(0, len(paragraphs), batch_size):
        batch = paragraphs[i:i+batch_size]

        for para in batch:
            emb, sents = encode_paragraph(model, tokenizer, para)
            all_embeddings.append(emb)
            all_sentences.extend(sents)

    return all_embeddings, all_sentences
```

### GPU Acceleration

```python
# Move model to GPU
device = torch.device('cuda')
model = model.to(device)

# Embeddings will be computed on GPU
embeddings, sentences = encode_paragraph(model, tokenizer, paragraph, device='cuda')
```

### Caching Embeddings

For repeated use:

```python
import pickle

# Compute once
embeddings_dict = {}
for doc_id, paragraph in documents.items():
    emb, sents = encode_paragraph(model, tokenizer, paragraph)
    embeddings_dict[doc_id] = (emb, sents)

# Save
with open('embeddings_cache.pkl', 'wb') as f:
    pickle.dump(embeddings_dict, f)

# Load later
with open('embeddings_cache.pkl', 'rb') as f:
    embeddings_dict = pickle.load(f)
```

## Common Patterns

### Pattern 1: Document Similarity

```python
def document_similarity(doc1, doc2):
    """Compute similarity between two documents."""
    emb1, _ = encode_paragraph(model, tokenizer, doc1)
    emb2, _ = encode_paragraph(model, tokenizer, doc2)

    # Average sentence embeddings
    doc1_emb = emb1.mean(axis=0)
    doc2_emb = emb2.mean(axis=0)

    # Cosine similarity
    sim = cosine_similarity([doc1_emb], [doc2_emb])[0, 0]
    return sim
```

### Pattern 2: Sentence Deduplication

```python
def deduplicate_sentences(sentences, threshold=0.95):
    """Remove duplicate/near-duplicate sentences."""
    # Encode all sentences in one paragraph
    paragraph = ' '.join(sentences)
    embeddings, _ = encode_paragraph(model, tokenizer, paragraph)

    # Compute pairwise similarity
    similarity = cosine_similarity(embeddings)

    # Find duplicates
    keep = []
    for i in range(len(sentences)):
        is_duplicate = False
        for j in keep:
            if similarity[i, j] > threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            keep.append(i)

    return [sentences[i] for i in keep]
```

### Pattern 3: Sentence Ordering

```python
def order_sentences(sentences):
    """Order sentences to maximize coherence."""
    paragraph = ' '.join(sentences)
    embeddings, _ = encode_paragraph(model, tokenizer, paragraph)

    # Simple greedy ordering
    ordered = [0]  # Start with first sentence
    remaining = list(range(1, len(sentences)))

    while remaining:
        last_emb = embeddings[ordered[-1]]

        # Find most similar to last sentence
        best_idx = None
        best_sim = -1
        for idx in remaining:
            sim = cosine_similarity([last_emb], [embeddings[idx]])[0, 0]
            if sim > best_sim:
                best_sim = sim
                best_idx = idx

        ordered.append(best_idx)
        remaining.remove(best_idx)

    return [sentences[i] for i in ordered]
```

## Limitations

1. **Context-dependent** - Same sentence in different contexts = different embeddings
2. **Requires paragraph** - Works best with 3+ sentences
3. **Computational cost** - More expensive than static embeddings
4. **Domain-specific** - Trained on your data, may not generalize to very different domains

## Next Steps

1. **Evaluate on your task** - Test on your specific use case
2. **Fine-tune if needed** - Continue training on domain-specific data
3. **Build applications** - Create search engines, clustering tools, etc.
4. **Experiment with embeddings** - Try different aggregation methods

---

**See also:**
- [MODEL_CHECKPOINTS.md](MODEL_CHECKPOINTS.md) - What's in the checkpoint
- [README.md](README.md) - Full documentation
- `scripts/use_model.py` - Example usage script
