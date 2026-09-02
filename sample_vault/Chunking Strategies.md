# Chunking Strategies

Chunking is the process of partitioning large Markdown notes or documents into smaller continuous text segments before generating [[Embeddings Explained|embeddings]] and storing them in [[Vector Databases|ChromaDB]].

## Why Chunking Matters

Large Language Models have finite context window limits, and embedding models process fixed token lengths efficiently (e.g., 256 to 512 tokens). Passing full 5,000-word notes degrades retrieval precision and dilutes embedding vectors.

## Common Chunking Strategies

### Fixed-Size Chunking
Splits text into uniform character or word counts regardless of sentence structures. Quick to compute but risks splitting sentences or code blocks mid-thought.

### Recursive Character Chunking
Splits text hierarchically using natural separator boundaries in order:
1. Double newlines (`\n\n` - paragraphs)
2. Single newlines (`\n` - lines)
3. Sentence punctuation (`. `, `? `, `! `)
4. Spaces

This approach preserves paragraph cohesion. Recommended parameters for RAG over personal notes:
- **Chunk Size**: ~300–500 tokens (approx. 1,200–2,000 characters).
- **Chunk Overlap**: ~50 tokens (approx. 150–200 characters) to ensure semantic continuity across split boundaries.

### Markdown Header Chunking
Uses Markdown heading markers (`#`, `##`, `###`) to preserve logical note sections and attach header contextual metadata to each chunk.

Optimal chunking directly enhances retrieval accuracy in [[RAG Fundamentals|RAG systems]] and reduces [[Hallucinations|LLM hallucinations]].

#chunking #nlp #rag #dataprep
