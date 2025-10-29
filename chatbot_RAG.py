"""
Author: Selim Ben Haj Braiek
Project: Chatbot with Retrieval-Augmented Generation (RAG)
Description: Simple RAG chatbot using LangChain to load PDF lectures, create embeddings,
             store them in Chroma, and build a retrieval-based QA system with GPT-4.
  
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

file_path = "intro-to-ai-lectures.pdf"   
persist_dir = "./intro-to-ai-lectures-chroma"

# 1️ Load PDF
loader = PyPDFLoader(file_path)
raw_docs = loader.load()  
print(f"Loaded {len(raw_docs)} pages from {file_path}")

# 2️ Split documents into chunks
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=800,
    chunk_overlap=150
)

docs = text_splitter.split_documents(raw_docs)
print(f"Split into {len(docs)} chunks")

for d in docs:
    if "Course Title" not in d.metadata:
        d.metadata["Course Title"] = "Introduction to Artificial Intelligence and Machine Learning"


# 3️ Create embeddings and persist to Chroma
embedding = OpenAIEmbeddings(model="text-embedding-ada-002")

# Create Chroma from documents (this will embed and store docs)
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding,
    persist_directory=persist_dir
)

# Persist to disk
vectorstore.persist()
print(f"✅ Embeddings stored in {persist_dir}")

# 4️ Load persisted Chroma and build retriever (MMR)
vectorstore = Chroma(
    persist_directory=persist_dir,
    embedding_function=embedding
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 20, "lambda_mult": 0.7}
)

# 5️ Build the RAG chain (LCEL style)
TEMPLATE = """
Answer the following question:
{question}

Use only the following context:
{context}

At the end, mention the lecture title(s) where this information was found in the format:
Resources: *Lecture Title*
"""

prompt_template = PromptTemplate.from_template(TEMPLATE)

chat = ChatOpenAI(
    model_name="gpt-4",
    model_kwargs={"seed": 365},
    max_tokens=250
)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt_template
    | chat
    | StrOutputParser()
)

# 6️ Query & generate answer

question = "Which programming languages are commonly used in AI and Machine Learning?"
response = chain.invoke(question)

print(response)

