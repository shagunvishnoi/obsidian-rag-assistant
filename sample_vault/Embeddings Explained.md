# Embeddings Explained

Vector embeddings convert textual tokens, sentences, or documents into dense numerical vectors (arrays of floating-point numbers) in a high-dimensional continuous space.

## Semantic Representation

Unlike traditional keyword search (such as TF-IDF or BM25), embeddings capture semantic meaning. Words or sentences with similar contextual meanings are positioned closely together in the embedding space.

For example:
- `"How do I configure vector storage?"`
- `"Setting up ChromaDB index"`

Even though these two sentences share few identical keywords, their embeddings map close together in vector space.

## Model Selection: all-MiniLM-L6-v2

`all-MiniLM-L6-v2` is a lightweight, highly efficient sentence transformer model:
- Produces 384-dimensional dense vectors.
- Optimized for fast CPU execution and local deployment without GPU hardware.
- Ideal for personal note-taking systems, local obsidian vault querying, and [[RAG Fundamentals|RAG search]].

In a RAG pipeline, the same embedding model must be used for both indexing note chunks and embedding incoming user queries to ensure space compatibility in [[Vector Databases]].

#embeddings #nlp #sentencetransformers #vectors
