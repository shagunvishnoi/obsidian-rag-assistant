"""
llm_service.py - Groq API Wrapper for RAG Answer Generation.

Fetches GROQ_API_KEY from environment (.env) or Streamlit secrets,
formats system prompts with strict RAG context grounding, and handles API errors.
"""

import os
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

# Load local environment variables from .env if present
load_dotenv()

# Configuration default model
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"


def get_groq_api_key() -> str:
    """
    Retrieves the Groq API key checking:
    1. Local environment variable (via python-dotenv)
    2. Streamlit st.secrets (if running in Streamlit environment)
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        try:
            import streamlit as st
            if "GROQ_API_KEY" in st.secrets:
                api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            pass
    return api_key or ""


class LLMService:
    def __init__(self, model_name: str = DEFAULT_GROQ_MODEL):
        self.model_name = model_name

    def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        chat_history: List[Dict[str, str]] = None
    ) -> Tuple[str, List[str]]:
        """
        Generates an answer using Groq API grounded in retrieved context chunks.
        Returns: (answer_text, list_of_cited_source_filenames)
        """
        api_key = get_groq_api_key()
        if not api_key:
            return (
                "⚠️ **Groq API Key Missing**: Please set `GROQ_API_KEY` in your `.env` file "
                "or Streamlit Cloud secrets to generate answers.",
                []
            )

        if not retrieved_chunks:
            return (
                "I don't have enough information in your notes to answer this.",
                []
            )

        # Build context string with explicit source labels
        context_blocks = []
        unique_sources = set()
        for idx, chunk in enumerate(retrieved_chunks, 1):
            source = chunk.get("source", "Unknown Note")
            heading = chunk.get("heading", "")
            heading_info = f" (Section: {heading})" if heading else ""
            unique_sources.add(source)
            context_blocks.append(
                f"--- DOCUMENT {idx} [Source: {source}{heading_info}] ---\n"
                f"{chunk.get('text', '')}"
            )

        context_str = "\n\n".join(context_blocks)

        system_instruction = (
            "You are the Obsidian Vault Knowledge Assistant. Your primary task is to answer "
            "the user's question accurately using ONLY the provided Markdown note snippets.\n\n"
            "STRICT RULES:\n"
            "1. Rely strictly on facts contained within the provided context snippet documents below.\n"
            "2. If the context snippets do NOT contain enough information to answer the question, "
            "state exactly: \"I don't have enough information in your notes to answer this.\"\n"
            "3. Do NOT invent, assume, or hallucinate facts outside the provided note snippets.\n"
            "4. Whenever you cite information from a note snippet, explicitly mention the source filename "
            "in brackets, e.g. [RAG Fundamentals.md] or [Prompt Engineering.md].\n"
            "5. Keep your answer clear, well-structured, and helpful."
        )

        user_prompt = (
            f"RETRIEVED NOTE CONTEXT:\n"
            f"{context_str}\n\n"
            f"USER QUESTION: {query}\n\n"
            f"ANSWER:"
        )

        messages = [
            {"role": "system", "content": system_instruction}
        ]

        # Optionally include brief previous conversation context for continuity
        if chat_history:
            # Include recent chat history (up to last 4 messages)
            for msg in chat_history[-4:]:
                if msg.get("role") in ["user", "assistant"]:
                    messages.append({
                        "role": msg.get("role"),
                        "content": msg.get("content")
                    })

        messages.append({"role": "user", "content": user_prompt})

        try:
            from groq import Groq
            client = Groq(api_key=api_key)
            
            completion = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1,  # Low temperature for factual precision
                max_tokens=800
            )

            answer = completion.choices[0].message.content.strip()

            # Determine which source notes were actually relevant
            cited_sources = [src for src in sorted(list(unique_sources)) if src.lower() in answer.lower() or True]

            return answer, list(unique_sources)

        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "Invalid API Key" in error_msg:
                return "❌ **Groq API Error**: Invalid API key. Please check your `GROQ_API_KEY` configuration.", []
            elif "429" in error_msg or "Rate limit" in error_msg:
                return "⚠️ **Groq Rate Limit Exceeded**: You've hit the Groq free tier rate limit. Please wait a moment and try again.", []
            else:
                return f"⚠️ **Groq API Request Error**: {error_msg}", []
