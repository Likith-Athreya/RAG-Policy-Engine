Prompt Engineering & RAG Mini Project

AI Engineer Intern – Take-Home Assignment

Overview

This project implements a small Retrieval-Augmented Generation (RAG) system for answering questions based on internal company policy documents such as Refund Policy, Shipping Policy, and Terms of Service.

The focus of this assignment is not UI or scale, but reasoning: how documents are retrieved, how prompts are designed, and how the system avoids hallucinations when information is missing.

The project also compares two prompt versions to demonstrate how prompt iteration improves answer quality and reliability.

Objective

Build a policy-aware question-answering assistant that:

Retrieves relevant information from provided documents

Generates accurate and grounded answers

Avoids hallucinating beyond the document content

Produces clear and structured responses

Architecture Overview

High-level flow:

Policy documents are loaded from disk

Text is chunked into overlapping segments

Embeddings are generated for each chunk

Chunks are stored in a vector database

Relevant chunks are retrieved per query

Retrieved context is passed to the LLM via a prompt

Documents → Chunking → Embeddings → Vector Store → Retrieval → LLM → Answer

Setup Instructions
Requirements

Python 3.10+

Ollama installed and running

Groq API key

Install Dependencies
pip install -r requirements.txt

Start Ollama
ollama serve
ollama pull nomic-embed-text

Environment Variables

Create a .env file:

GROQ_API_KEY=your_api_key_here

Run the Script
python lang.py

Data Preparation
Document Loading

All policy documents are stored in the docs/ directory

PDFs are loaded using DirectoryLoader with PyPDFLoader

Chunking Strategy
chunk_size = 600
chunk_overlap = 100


Reasoning:

Policy clauses often span multiple sentences

A 600-token chunk keeps related information together

Overlap helps preserve context across chunk boundaries

This setup works well for factual retrieval rather than summarization

RAG Pipeline
Embeddings

Model: nomic-embed-text (via Ollama)

Chosen for local execution, consistency, and zero API cost

Vector Store

Chroma

Lightweight and sufficient for the size of the document set

Retrieval

Top-k semantic retrieval (k = 3)

Balances relevance and noise

Prompt Engineering
Prompt Version 1 (Baseline)
Answer based on context: {context}
Question: {input}


This prompt provides minimal instruction and allows fluent answers, but does not strongly restrict the model from going beyond the retrieved context.

Prompt Version 2 (Improved)
You are a professional Policy Assistant.
Use ONLY the provided context to answer.
If the answer is not in the context, say "I do not have sufficient information."
Format your answer using bullet points for clarity.
Include section names if available.


What changed and why:

Explicitly restricts the model to retrieved context

Adds a clear fallback when information is missing

Enforces structured output

Significantly reduces hallucinations

Better suited for policy and legal content

Evaluation
Evaluation Set

Five test questions were created covering:

Clearly answerable questions

Questions requiring reasoning across sections

Edge cases such as legal clauses

Questions that could trigger hallucinations if not handled carefully

Evaluation Criteria

Accuracy of answers

Hallucination avoidance

Clarity and structure

Faithfulness to source documents

Results Summary

Prompt Version 2 consistently produced:

More grounded answers

Clearer structure

Safer handling of missing or partial information

Edge Case Handling

When no relevant information is found, the system responds with
"I do not have sufficient information."

Out-of-scope questions are not answered speculatively

Answers remain strictly tied to retrieved document content

Trade-offs and Design Decisions

Local embeddings were chosen over hosted APIs for simplicity and reliability

Retrieval was kept simple (no reranking) to maintain transparency

No UI was built to keep focus on prompt quality and reasoning

What I’m Most Proud Of

I’m most proud of how prompt iteration improved answer quality and hallucination control, and how the system remains grounded even for legal and edge-case queries.

What I’d Improve Next

With more time, I would:

Add a reranking step to improve retrieval precision

Persist the vector database to disk

Add automated evaluation metrics

Enforce structured output with a schema

Repository Structure
├── docs/
│   ├── Refund_policy.pdf
│   ├── logistics_policy.pdf
│   └── terms_service.pdf
├── lang.py
├── requirements.txt
└── README.md
