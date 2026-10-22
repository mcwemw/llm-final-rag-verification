# Evaluating Published Faithfulness-Verification Techniques for a RAG QA System

**535106 LLM Team Project · 第三組**
Authors: Mei-Chen (Emma) Wu (414551035), Chun-Yao Tseng (315551125),
Bo-Yu Xu (315511057), Noa Andre (315554801) — NYCU.

This repository contains the **project proposal only**, for grading. It is an **evaluation
study of existing, published techniques** and contains **no unpublished method of the
authors** (see *Scope* below).

## Summary
RAG grounds LLM answers in retrieved documents, but a grounded *citation* does not
guarantee a grounded *claim*. Many **published** techniques can check claim-level
faithfulness; this project **evaluates adopting them** as a verification layer on a
retrieval-augmented QA system, and benchmarks their agreement with human labels, cost, and
reproducibility to produce an **adoption recommendation**. Full details in
[`proposal.pdf`](proposal.pdf).

## Published techniques evaluated (all already available)
- **LLM-as-judge**: RAGAS.
- **Reference-free / sampling**: SelfCheckGPT; purpose-built detectors benchmarked by Sardana (2025).
- **Claim-level / atomic**: FActScore, VeriScore (claim extraction).
- **NLI-based consistency**: TRUE.
- **Embedding-similarity**: normalized cosine (e.g., bge-m3); plus a from-scratch
  self-attention scorer built in course HW1.
- **Benchmarks / ground truth**: RAGTruth (span-level human labels), FaithBench
  (judge-disagreement cases), Face4RAG (Chinese RAG factual consistency).

Full references with DOIs: [`reference.bib`](reference.bib).

## Build
- **Overleaf**: upload this folder, set `proposal.tex` as the main file, Recompile.
- **Local**: `pdflatex proposal` → `bibtex proposal` → `pdflatex` ×2.

## Scope
This repo is a **course proposal that evaluates published methods**. Any novel method,
tuning, or integration the authors may pursue as separate (unpublished) research is **not**
part of this repository. Please cite the original authors of each technique evaluated here.

© 2026 M.-C. Wu & C.-Y. Tseng. Provided for coursework review.
