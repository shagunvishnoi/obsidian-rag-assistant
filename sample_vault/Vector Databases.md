# Vector Databases

Vector databases are specialized data stores designed to store, index, and query high-dimensional vector representations ([[Embeddings Explained|embeddings]]) efficiently.

## Core Concepts

In traditional relational or key-value databases, queries search for exact matches or numerical ranges. In vector databases, queries search for mathematical proximity in high-dimensional vector space using distance metrics:

- **Cosine Similarity**: Measures the cosine of the angle between two vectors (popular for normalized text embeddings).
- **Euclidean Distance (L2)**: Measures the straight-line distance between vector endpoints.
- **Dot Product**: Measures magnitude and direction match.

## ChromaDB Overview

ChromaDB is an open-source, developer-friendly vector store ideal for local and lightweight RAG applications:
- Operates locally with persistent file storage (`chromadb.PersistentClient`).
- Seamlessly integrates with embedding models like `sentence-transformers` (`all-MiniLM-L6-v2`).
- Stores raw text, vector embeddings, and arbitrary metadata (such as note title, heading, and line numbers) together.

Vector databases form the retrieval backbone of [[RAG Fundamentals|RAG architecture]], enabling sub-second context search over hundreds of Markdown notes.

#vectordb #chromadb #retrieval #database
