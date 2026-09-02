# Prompt Engineering

Prompt engineering is the craft of structuring instructions provided to Large Language Models to achieve optimal accuracy, reasoning, and formatting in generated outputs.

## Key Techniques

### System Prompts and Roles
Defining explicit roles (e.g., "You are an expert Obsidian Knowledge Assistant") establishes behavioral boundaries. System prompts instruct the LLM on output format, tone, and constraints such as restricting answers to provided contexts.

### Few-Shot Prompting
Providing concrete input-output examples inside the prompt helps guide complex formatting, classification tasks, and citation structures.

### Context Grounding for RAG
In [[RAG Fundamentals|RAG systems]], the context prompt must explicitly constrain the LLM:
- Direct the model to rely exclusively on the provided context snippets.
- Instruct the model to clearly state "I don't have enough information in your notes to answer this" if the required details are absent.
- Enforce strict source document citation tags.

## Temperature and Decoding
Setting lower temperature values (e.g., 0.0 to 0.2) promotes deterministic and factual responses, minimizing non-deterministic output drift when answering factual queries.

#prompting #groq #llm #best_practices
