# Hallucinations

In large language models, a hallucination refers to a generated response that sounds confident and plausible but is factually incorrect, ungrounded, or unsupported by external reality.

## Causes of Hallucination

1. **Parametric Compression**: LLMs compress world knowledge into billions of weights, leading to fuzziness or memory degradation on niche details.
2. **Probability Maximization**: Decoding algorithms select tokens based on statistical likelihood rather than verified truth lookup.
3. **Over-generalization**: Models may combine disjoint facts from different training contexts incorrectly.

## Mitigating Hallucinations with RAG

Implementing [[RAG Fundamentals|Retrieval-Augmented Generation]] significantly reduces hallucinations:

- **Context Grounding**: System prompts explicitly command the LLM to restrict its answers strictly to retrieved source chunks.
- **Source Verification**: Explicit citations reference note titles and excerpt previews, allowing human verification.
- **Fallback Instructions**: Instructing the model to output a clear refusal message when information is missing prevents plausible guessing.

#hallucinations #safety #rag #factuality
