"""
rag_engine.py - Chunking, Embedding, and Retrieval Engine for Obsidian Notes.

Handles Obsidian Markdown parsing, recursive text chunking, local vector embedding
via sentence-transformers (all-MiniLM-L6-v2), and ChromaDB persistence.
"""

import os
import re
import shutil
from typing import List, Dict, Any, Tuple, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


# Configuration Constants
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "obsidian_vault"
DEFAULT_CHUNK_SIZE = 1400      # ~350 tokens (characters)
DEFAULT_CHUNK_OVERLAP = 200    # ~50 tokens (characters)


class RAGEngine:
    def __init__(self, persist_directory: str = CHROMA_PERSIST_DIR):
        self.persist_directory = persist_directory
        self._embedding_model: Optional[SentenceTransformer] = None
        self._chroma_client = None
        self._collection = None

    @property
    def embedding_model(self) -> SentenceTransformer:
        """Lazy loader for SentenceTransformer model."""
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return self._embedding_model

    def _get_client(self):
        """Lazy loader for ChromaDB persistent client."""
        if self._chroma_client is None:
            self._chroma_client = chromadb.PersistentClient(path=self.persist_directory)
        return self._chroma_client

    def get_collection(self):
        """Retrieve or create the ChromaDB collection."""
        client = self._get_client()
        return client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    def parse_obsidian_markdown(self, raw_text: str) -> Tuple[str, List[str]]:
        """
        Parses Obsidian Markdown content:
        - Cleans Wikilinks: [[Page Name|Alias]] -> Alias, [[Page Name]] -> Page Name
        - Preserves tags into a metadata list without removing context.
        Returns: (cleaned_text, tags_list)
        """
        # Extract tags (#tag_name, ignoring hex colors or header symbols)
        tag_pattern = r'(?<!\S)#([a-zA-Z0-9_\-]+)'
        tags = list(set(re.findall(tag_pattern, raw_text)))

        # Clean wikilinks: [[Link Target|Displayed Alias]] -> Displayed Alias
        cleaned = re.sub(r'\[\[([^\]\|]+)\|([^\]]+)\]\]', r'\2', raw_text)
        # Clean wikilinks: [[Link Target]] -> Link Target
        cleaned = re.sub(r'\[\[([^\]]+)\]\]', r'\1', cleaned)

        return cleaned, tags

    def split_text_by_headings(self, text: str) -> List[Tuple[str, str]]:
        """
        Splits markdown text by section headers (#, ##, ###).
        Returns a list of tuples: (heading_name, section_content).
        """
        heading_pattern = r'^(#{1,6})\s+(.+)$'
        lines = text.split('\n')
        
        sections: List[Tuple[str, str]] = []
        current_heading = "Introduction"
        current_lines: List[str] = []

        for line in lines:
            match = re.match(heading_pattern, line.strip())
            if match:
                if current_lines:
                    sections.append((current_heading, '\n'.join(current_lines)))
                    current_lines = []
                current_heading = match.group(2).strip()
            else:
                current_lines.append(line)

        if current_lines:
            sections.append((current_heading, '\n'.join(current_lines)))

        return sections

    def recursive_chunk_text(
        self,
        text: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP
    ) -> List[str]:
        """
        Recursive character-based text chunker.
        Splits by paragraph (\n\n), then line (\n), then sentence (. ), then space.
        """
        if len(text) <= chunk_size:
            return [text.strip()] if text.strip() else []

        separators = ["\n\n", "\n", ". ", " "]
        
        def split_with_separators(content: str, sep_idx: int) -> List[str]:
            if len(content) <= chunk_size or sep_idx >= len(separators):
                return [content]
            
            sep = separators[sep_idx]
            splits = content.split(sep)
            chunks: List[str] = []
            current_chunk: List[str] = []
            current_len = 0

            for part in splits:
                part_len = len(part) + len(sep)
                if current_len + part_len > chunk_size and current_chunk:
                    chunk_str = sep.join(current_chunk).strip()
                    if chunk_str:
                        chunks.append(chunk_str)
                    
                    # Compute overlap
                    overlap_parts = []
                    overlap_len = 0
                    for prev in reversed(current_chunk):
                        if overlap_len + len(prev) <= overlap:
                            overlap_parts.insert(0, prev)
                            overlap_len += len(prev)
                        else:
                            break
                    current_chunk = overlap_parts + [part]
                    current_len = sum(len(p) + len(sep) for p in current_chunk)
                else:
                    current_chunk.append(part)
                    current_len += part_len

            if current_chunk:
                chunk_str = sep.join(current_chunk).strip()
                if chunk_str:
                    chunks.append(chunk_str)

            return chunks

        final_chunks = split_with_separators(text, 0)
        return [c for c in final_chunks if c.strip()]

    def process_note(self, filename: str, content: str) -> List[Dict[str, Any]]:
        """
        Parses and chunks a single note into dictionary records ready for vector store.
        """
        cleaned_text, tags = self.parse_obsidian_markdown(content)
        sections = self.split_text_by_headings(cleaned_text)
        
        chunk_records: List[Dict[str, Any]] = []
        global_chunk_idx = 0

        for heading, section_text in sections:
            if not section_text.strip():
                continue
            
            chunks = self.recursive_chunk_text(section_text)
            for chunk_str in chunks:
                if not chunk_str.strip():
                    continue
                chunk_records.append({
                    "id": f"{filename}_chunk_{global_chunk_idx}",
                    "text": chunk_str,
                    "metadata": {
                        "source": filename,
                        "chunk_index": global_chunk_idx,
                        "heading": heading,
                        "tags": ", ".join(tags) if tags else "none"
                    }
                })
                global_chunk_idx += 1

        return chunk_records

    def clear_vault(self):
        """Clears and resets the local vector storage collection."""
        client = self._get_client()
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass  # Collection might not exist yet
        self._collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    def index_vault(
        self,
        files_dict: Dict[str, str],
        progress_callback=None
    ) -> Tuple[int, int]:
        """
        Indexes a dictionary of {filename: content} into ChromaDB.
        Returns: (total_notes, total_chunks)
        """
        self.clear_vault()
        collection = self.get_collection()

        all_records: List[Dict[str, Any]] = []
        total_files = len(files_dict)

        for idx, (filename, content) in enumerate(files_dict.items()):
            records = self.process_note(filename, content)
            all_records.extend(records)
            if progress_callback:
                progress_callback(int((idx + 1) / total_files * 50))  # 0 to 50% for parsing

        if not all_records:
            return total_files, 0

        # Batch embed and store in ChromaDB
        texts = [r["text"] for r in all_records]
        ids = [r["id"] for r in all_records]
        metadatas = [r["metadata"] for r in all_records]

        # Generate embeddings
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False).tolist()
        
        if progress_callback:
            progress_callback(80)

        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            collection.upsert(
                ids=ids[i:i+batch_size],
                embeddings=embeddings[i:i+batch_size],
                documents=texts[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size]
            )

        if progress_callback:
            progress_callback(100)

        return total_files, len(all_records)

    def query_vault(self, query_text: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Queries ChromaDB collection for top_k relevant chunk snippets.
        """
        collection = self.get_collection()
        if collection.count() == 0:
            return []

        query_embedding = self.embedding_model.encode([query_text], show_progress_bar=False).tolist()
        
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        retrieved_chunks = []
        if results and results.get("documents"):
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]

            for doc, meta, dist in zip(documents, metadatas, distances):
                retrieved_chunks.append({
                    "text": doc,
                    "source": meta.get("source", "Unknown"),
                    "chunk_index": meta.get("chunk_index", 0),
                    "heading": meta.get("heading", ""),
                    "tags": meta.get("tags", ""),
                    "score": round(1 - float(dist), 4)  # Convert cosine distance to similarity
                })

        return retrieved_chunks

    def get_vault_stats(self) -> Dict[str, Any]:
        """Returns statistics for currently loaded vault notes."""
        collection = self.get_collection()
        total_chunks = collection.count()
        
        if total_chunks == 0:
            return {"total_notes": 0, "total_chunks": 0, "filenames": []}

        # Fetch metadata to extract distinct filenames
        data = collection.get(include=["metadatas"])
        filenames = sorted(list(set(m.get("source", "") for m in data.get("metadatas", []) if m)))
        
        return {
            "total_notes": len(filenames),
            "total_chunks": total_chunks,
            "filenames": filenames
        }
