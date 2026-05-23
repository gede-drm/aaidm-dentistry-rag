**Outlining with Chat-GPT**

**Prompted: 2026-05-19 23:30**



Adapt RAG Foundry’s methodology/design ideas into your own pediatric dentistry RAG system.



That is much more doable and honestly the correct interpretation of the paper.



After reading it, the key contribution of the paper is that RAG should be treated as an experimental pipeline with 4 modular stages:



**Data Creation / Processing**

* load docs
* chunk
* retrieve
* build prompts
* cache datasets



**Training**

* LoRA / PEFT fine-tuning



**Inference**

* generate grounded answers



**Evaluation (Use RAGAS)**

* faithfulness
* relevancy
* exact match / semantic metrics





**PIPELINE Suggested by Chat-GPT**

PDF docs

&#x20;  ↓

Chunk + Embed

&#x20;  ↓

Neo4j



Store:

\- chunk text

\- embeddings

\- source metadata

\- concept relationships



Telegram

&#x20;  ↓

n8n

&#x20;  ↓

FastAPI

&#x20;  ↓

Neo4j vector search

&#x20;  ↓

(optional graph traversal)

&#x20;  ↓

Qwen/Llama via Ollama

&#x20;  ↓

Response





**Notes:**

1. Structured prompt templates (must implement)

&#x09;Paper explicitly uses prompt templates + system instructions

&#x09;e.g. 	Question: {query}

&#x09;	Context: {docs}



&#x09;	Explain step-by-step.

&#x09;	Only use context.

&#x09;	If insufficient evidence, say so.

&#x09;	Provide final concise guidance.

2\. Top-k retrieval tuning (easy win)

&#x09;Paper uses retrieval experimentation, you can test k=3,5,8.

3\. Optional LoRA fine-tuning (only if time)

&#x09;Paper uses LoRA fine-tuning heavily.

4\. Safety constraints (VERY relevant for healthcare)

&#x09;Paper explicitly mentions responsible usage and hallucination caution.

&#x09;For dentistry chatbot, implement:

&#x09;*If query contains: severe bleeding, fever after extraction, swelling + breathing issue, emergency symptoms*

&#x09;*Bot responds: Please seek immediate professional dental care*

5\. Convert messy textbook PDFs into clean structured knowledge

&#x09;Extract PDF to text, GPT suggests to use pymupdf

&#x09;It says that pymupdf can breakdown blocks/columns in pdf file.

&#x09;Then we can delete the obvious junk

