# AI Developer Guide: GTD RetClean Project

## 1. Project Goal
[cite_start]We are building a Python pipeline to fill in missing terrorist group names ("Gname") in the Global Terrorism Database (GTD)[cite: 329]. We use the RetClean method. [cite_start]This means using Retrieval-Augmented Generation (RAG) and local AI models[cite: 328, 329]. Reference the project proposal for overall plan.

## 2. Tech Stack
* [cite_start]**Language:** Python [cite: 329]
* [cite_start]**Text Search:** Elasticsearch [cite: 331]
* [cite_start]**Vector Search:** Faiss [cite: 331]
* [cite_start]**AI Models:** Sentence Transformers (BERT), RoBERTa, and local LLMs like LLaMA or Dolly-v2-3B [cite: 335, 336, 354]

## 3. Setup Steps
1.  [cite_start]**Data:** Put the GTD database file in the `data/` folder[cite: 335].
2.  [cite_start]**Search Server:** Start a local Elasticsearch instance[cite: 343].
3.  **Packages:** Install required libraries (e.g., `pandas`, `elasticsearch`, `faiss-cpu`, `sentence-transformers`).

## 4. Current AI Coding Tasks
[cite_start]We are currently working on the Tuple-based Indexer(Week 3-4 in Project Proposal). When writing code for us, focus on these steps:

1.  **Data Prep:** Load the GTD data. Split it into known attacks (our data lake) and unknown attacks (dirty data).
2.  [cite_start]**Elasticsearch Setup:** Write a script to index the text summaries of known attacks into Elasticsearch[cite: 331].
3.  [cite_start]**Faiss Setup:** Write a script to turn text summaries into dense vector embeddings using BERT, then add them to a Faiss index[cite: 331, 335].
4.  [cite_start]**Testing:** Create a script to run basic test queries to find matching historical attacks[cite: 343].

## 5. Coding Rules
* Keep functions small and easy to read.
* Always include comments explaining the RAG steps.