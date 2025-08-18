from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
book_title = "Revelation"

pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index(PINECONE_INDEX)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)

vector_store = PineconeVectorStore(index=index, embedding=embeddings)

with open(f"bible/{book_title}.txt", encoding="euc-kr") as f:
    book_text = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_text(book_text)

documents = [Document(page_content=chunk, metadata={"book": book_title}) for _, chunk in enumerate(chunks)]
ids = [f"{book_title}-{i}" for i in range(len(chunks))]

vector_store = PineconeVectorStore(index=index, embedding=embeddings, namespace=book_title)

vector_store.add_documents(
    documents=documents,
    ids=ids
)