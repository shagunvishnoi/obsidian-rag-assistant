# RAG Fundamentals

Retrieval-Augmented Generation (RAG) is an architectural pattern that enhances Large Language Model (LLM) capabilities by attaching an external knowledge base. Instead of relying solely on parametric knowledge learned during training, a RAG system fetches relevant documents from a database when a query is submitted.

## How RAG Works

The RAG workflow consists of three primary steps:

1. **Ingestion and Indexing**: Markdown notes or documents are split into manageable text chunks. These chunks are transformed into dense vector representations using [[Embeddings Explained|embedding models]] and stored in a [[Vector Databases|vector database]].
2. **Retrieval**: When a user submits a prompt, the system converts the user's query into an embedding and computes vector similarity against stored document chunks to retrieve top matching contexts.
3. **Generation**: The retrieved context chunks are inserted into a structured prompt alongside the original question and sent to an LLM like Groq's Llama 3. The LLM synthesizes an accurate answer based strictly on the retrieved facts.

## Benefits of RAG

- **Reduces Hallucination**: Grounding response generation in retrieved facts prevents model fabricated claims (see [[Hallucinations]]).
- **Up-to-date Knowledge**: Domain knowledge can be updated by re-indexing files without expensive model re-training (compare with [[Fine-Tuning vs RAG]]).
- **Verifiable Citations**: Enables explicit citations referencing source note filenames for auditability.

#rag #architecture #llm #knowledge_base
