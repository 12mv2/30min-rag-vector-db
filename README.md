# 30min-rag-vector-db

**Repository Name:** `12mv2-30min-rag-vector-db`

**Description:**
This concise, hands-on workshop shows how to convert structured data (like gait metrics) into vector embeddings, store them in a vector database, perform similarity searches using cosine similarity, and use a Large Language Model (LLM) to generate responses grounded in the retrieved data. Participants will use familiar runner and animal gait metrics to build an end-to-end RAG pipeline.

---

## 🏃 Workshop Overview

Participants will learn:

* How to transform structured numerical data into vector embeddings.
* How to store and manage embeddings in Pinecone.
* How to perform similarity searches using cosine similarity.
* How to use an LLM (ChatGPT or Gemini) to generate fact-grounded responses using retrieved vector data.

---

## 🚀 Quick Start Guide

### 1. Clone the repository

```bash
git clone https://github.com/your-username/12mv2-30min-rag-vector-db.git
cd 12mv2-30min-rag-vector-db
```

### 2. Set up your environment

```bash
python -m venv venv
# On macOS/Linux
source venv/bin/activate
# On Windows
venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Configure API keys

Create a `.env` file in the project root:

```env
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX=runners-index

# Choose an LLM provider:
OPENAI_API_KEY=your-openai-api-key
# OR
GEMINI_API_KEY=your-gemini-api-key
```

**Reminder**: Do not commit `.env` to GitHub.

### 4. Set up Pinecone (if needed)

If you haven't configured Pinecone yet, please follow [the setup instructions from our previous workshop](https://github.com/12mv2/30min-vector-db). Experienced participants are encouraged to assist others.

---

## 🔍 How to Use

Run `rag_query.py` and input queries:

```bash
python rag_query.py
```

Example queries:

* "Which animal has a running style closest to Eliud Kipchoge?"
* "Find runners similar to Usain Bolt based on their gait metrics."
* "Compare human and animal vertical oscillation metrics."

The script will:

1. Embed your query into a vector.
2. Retrieve similar entries from Pinecone.
3. Use an LLM to generate a response based only on that retrieved data.

---

## 📁 Repository Structure

```
12mv2-30min-rag-vector-db/
├── data/
│   ├── runners.json            # Raw data for embeddings
│   └── definitions.json        # Metric definitions
├── db/
│   └── vector_db.py            # Local data handling
├── rag_query.py                # RAG + Vector DB integration script
├── pinecone_upload.py          # Script for Pinecone setup
├── requirements.txt            # Python dependencies
├── .env                        # API keys
└── README.md
```

---

## 🤖 Technologies Used

* Python 3.11
* OpenAI API (GPT-3.5-turbo)
* Google Gemini API
* Pinecone Vector Database
* Dependencies: `python-dotenv`, `pinecone`, `openai`, `google-generativeai`

---

## 📚 Core Concepts

* **Vector Embeddings:** Converting structured data into numerical vectors.
* **Vector Database:** Efficient storage and retrieval of vectors.
* **Cosine Similarity:** A way to find the most similar entities.
* **RAG (Retrieval-Augmented Generation):** Enhance LLM responses by injecting real-world vector-based search results.

---

## 🧠 Learning Objectives

* Implement the full RAG workflow using structured, numeric input.
* Upload and query vector embeddings with Pinecone.
* Use an LLM to produce accurate, grounded responses from retrieved results.

---

## 🗣️ Licensing

Licensed under MIT – freely use and adapt for learning and personal development.

