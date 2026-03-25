# RAG Chatbot with MultiQuery Retriever

A retrieval-augmented generation chatbot that answers questions over internal company documents using LangChain's MultiQuery Retriever, Elasticsearch as the vector store, and Groq as the LLM.

→ [lab_chatbot_with_multi_query_retriever.ipynb](lab_chatbot_with_multi_query_retriever.ipynb)

## What it demonstrates

Standard RAG fails when a user's phrasing doesn't closely match how relevant documents are worded. MultiQuery Retriever solves this by generating multiple rephrased versions of each question before searching — broadening recall without changing the underlying vector store or embedding model. This lab makes that tradeoff concrete: you can observe the generated query variants in the logs and see how retrieval coverage changes.

## Key decisions worth noting

**Embedding model matters more than the retriever.** The original scaffold used `all-MiniLM-L6-v2`, which returned NASA space agency content when queried about the "NASA sales region" acronym. Switching to `BAAI/bge-large-en-v1.5` (1024-dim, stronger semantic disambiguation) resolved it. The retriever pattern only helps if the embeddings understand the domain.

**Remote inference everywhere — no local GPU required.** Embeddings run via HuggingFace Inference API, the LLM runs on Groq (free tier, fast), and vectors are stored in Elasticsearch Serverless. The entire pipeline runs on any machine with API keys.

**MultiQuery doubles token usage.** Each question triggers one Groq call to generate query variants, then a final call to answer — budget accordingly on free tiers.

**Changing embedding models requires re-indexing.** Switching from MiniLM to BGE produces vectors of different dimensions; storing them in the same Elasticsearch index causes errors. The notebook uses a separate `INDEX_NAME` variable so re-indexing creates a new index cleanly.

**Self-healing retriever is redundant here.** The retry logic (rewrite the query and search again on empty results) adds latency but the multi-query loop already covers empty-result cases through query variation. Left in as an illustration, not a recommendation.

## Stack

| Component | Choice |
|---|---|
| LLM | Groq `llama-3.3-70b-versatile` |
| Embeddings | `BAAI/bge-large-en-v1.5` via HuggingFace Hub |
| Vector store | Elasticsearch Serverless |
| Retriever | LangChain `MultiQueryRetriever` |
| Framework | LangChain / LCEL |

## How to run

1. Copy `.env.example` to `.env` and fill in your keys:
   ```
   GROQ_API_KEY=
   HF_TOKEN=
   ELASTIC_API_KEY=
   ELASTIC_CLOUD_ID=
   ```

2. Install dependencies:
   ```bash
   pip install "langchain>=1.0" "langchain-core>=0.3" "langchain-community>=0.4" \
     "langchain-classic>=0.3" langchain-groq langchain-huggingface \
     langchain-elasticsearch jq lark elasticsearch python-dotenv
   ```

3. Open and run `lab_chatbot_with_multi_query_retriever.ipynb` top to bottom. The first run indexes the documents into Elasticsearch; subsequent runs can skip the indexing cells.
