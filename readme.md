# Chunking as a Retrieval Bottleneck in Commercial used RAG Systems  
An Empirical Systems Level Evaluation Using Cohere Embed & Rerank

---

## Overview

Modern Retrieval-Augmented Generation (RAG) systems rely heavily on commercial embedding and reranking APIs (e.g., Cohere Embed, Cohere Rerank 4). These components deliver strong performance but operate as black boxes, encouraging practitioners to treat upstream design decisions such as document chunking as secondary.

This project challenges that assumption.

We present an empirical systems evaluation demonstrating that static, query-independent chunking introduces a recall ceiling that even state-of-the-art rerankers cannot overcome. By holding embeddings and reranking constant and varying only the chunking strategy, we show that chunking directly constrains what information is retrievable, shaping end-to-end system performance.

---

## Core Question

> Can strong commercial used rerankers compensate for poor document segmentation or does chunking fundamentally bound retrieval performance?

---

## Key Findings

- Static token-based chunking (512 tokens) creates an early recall ceiling:
  - Baseline Recall@1 = 0.5
  - After reranking: Recall@1 = 0.75
-  Semantic, query aligned chunking removes this ceiling entirely:
  - Recall@1 = 1.0 before reranking
  - Reranking provides no additional gain because retrieval is already optimal
- Reranking improves ordering, but cannot recover information that was never retrievable
- Chunking is therefore a first-order systems decision, not a preprocessing detail

Chunking defines what is retrievable and reranking only reorders what survives.

---

## Methodology

This work adopts a black-box systems evaluation approach appropriate for commercial APIs.

### Controlled Components (held constant)

- **Embedding model:** Cohere Embed v4  
- **Reranking model:** Cohere Rerank 4 (32K context window)  
- **Retrieval strategy:** Dense retrieval (Top-150 to rerank Top-20)  
- **Corpus:** MIT OpenCourseWare *Quantum Physics III* lectures  
- **Queries:** Conceptual, long form technical questions  

### Independent Variable

- **Chunking strategy**
  - **Baseline:** Static token based chunking (512 tokens)
  - **Experimental:** Semantic, query fitted chunking (LumberChunker-style)

### Metrics

- Recall@k (k ∈ {1, 5, 10, 20})
- NDCG@k
- MRR (reported where applicable)

---

## Results Summary

| Pipeline | Recall@1 | NDCG@1 | Notes |
|--------|---------|--------|------|
| Static chunking (pre-rerank) | 0.50 | 0.50 | Relevant chunks often retrieved but poorly ranked |
| Static chunking (post-rerank) | 0.75 | 0.75 | Reranker helps but does not fully correct segmentation errors |
| Semantic chunking (pre-rerank) | 1.00 | 1.00 | Relevant chunk retrieved at rank 1 for all evaluated queries |
| Semantic chunking (post-rerank) | 1.00 | 1.00 | No further improvement needed |

Full tables, plots, and recall ceiling analyses are provided in the report.

---

## Why This Matters

- Confirms recall bounded reranking in real world, commercial RAG stacks
- Demonstrates that better scoring cannot compensate for poor segmentation
- Provides practical guidance for engineers building RAG systems with proprietary APIs
- Aligns with emerging literature on retrieval failure modes, but tests them in production-like conditions

This work complements prior research on reranking and embeddings by showing that segmentation is an equally critical bottleneck.

---

## Limitations

- Small, focused query set (technical physics domain)
- Single document corpus
- Semantic chunking implemented via LumberChunker-style segmentation  
  (full Relevant Segment Extraction not yet implemented)
- Ceiling effects limit fine-grained comparison at higher k

---

## Future Work

- Implement full query adaptive Relevant Segment Extraction
- Expand evaluation to multiple domains (legal, scientific, financial)
- Scale to larger corpora and more diverse query distributions
- Perform explicit cost benefit analysis under realistic query loads
- Explore hybrid strategies that selectively apply semantic chunking

---

## Full Report

**Empirical Research Study (PDF):**  
`Empirical_research_Study_.pdf`

The report includes:
- Literature grounding
- Experimental design
- Full metric tables and plots
- Recall ceiling analysis
- Systems-level interpretation


