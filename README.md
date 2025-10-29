# chatbot_RAG

🚀 Retrieval-Augmented Generation (RAG) with LangChain, OpenAI, and Chroma — AI Knowledge Retrieval from Custom Documents

**Title**

🚀 Retrieval-Augmented Generation (RAG) with LangChain, OpenAI, and Chroma — AI Knowledge Retrieval from Custom Documents

**Description**

This project demonstrates a complete Retrieval-Augmented Generation (RAG) pipeline built using LangChain, OpenAI GPT-4, and Chroma vector database.

It showcases how to:

Load and process domain-specific documents (Markdown or PDF)

Split content into manageable chunks with structured metadata

Generate embeddings using text-embedding-ada-002

Store and persist them locally in Chroma for efficient vector search

Retrieve the most relevant text chunks using Maximal Marginal Relevance (MMR)

Generate factually grounded GPT-4 responses using only retrieved data

**Example Use Case**

This example uses a file titled "Introduction to Artificial Intelligence and Machine Learning", allowing GPT-4 to answer conceptual questions like:

“Which programming languages are commonly used in AI and Machine Learning?”

The model retrieves the relevant lecture sections and produces a precise, context-aware answer — citing its sources from the document.

**Tech Stack**

🧠 LangChain (document loaders, retrievers, and chains)

🪶 OpenAI GPT-4 (LLM for response generation)

🔍 Chroma (local vector database for embedding storage)

🧩 text-embedding-ada-002 (embedding model)

🧱 MarkdownHeaderTextSplitter (for structured content splitting)

## Installation

git clone https://github.com/selimbenhajbraiek/chatbot_RAG.git
cd CHATBOT_RAG

pip install -r requirements.txt

please make sure the .env file in the same folder if the python file and contain the OpenAI key: OPENAI_API_KEY = ""

and if you're using the jupyter make sure to add:
%load_ext dotenv
%dotenv

