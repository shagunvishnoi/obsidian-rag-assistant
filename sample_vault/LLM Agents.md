# LLM Agents

LLM Agents are autonomous AI systems that leverage large language models as central decision-making engines to achieve complex goals by planning, executing tools, and evaluating feedback loop outcomes.

## Core Agent Components

1. **Planning**: Breaking down high-level objectives into sequential steps and sub-goals.
2. **Memory**:
   - *Short-term memory*: Context window history maintained during a conversation session.
   - *Long-term memory*: External knowledge retrieved via [[RAG Fundamentals|RAG]] or [[Vector Databases|vector databases]].
3. **Tool Use**: Executing external APIs, web searches, calculators, code interpreters, or database search handlers.

## Agent Workflows vs Simple RAG

While basic RAG retrieves fixed document chunks to answer single-turn questions, an agent can dynamically decide when to search, refine query terms, inspect retrieved source notes, and perform iterative multi-step reasoning.

#agents #autonomous #ai #workflows
