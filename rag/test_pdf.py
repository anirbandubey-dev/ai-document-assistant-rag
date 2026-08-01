from langchain_community.document_loaders import PyPDFLoader
import sentence_transformers
from streamlit import form

import faiss
import numpy as np

#load the pdf
loader = PyPDFLoader("database/data/Deepdive.pdf")
docs = loader.load()

#number of pages
print("Number of pages:", len(docs))

#print first 100 characters of first page
print(docs[0].page_content[:100])

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(docs)

print("Number of chunks:", len(chunks))
print(chunks[0].page_content[:100])

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [chunk.page_content for chunk in chunks]

embeddings = model.encode(texts)

print("Embedding vector size:", len(embeddings[0]))
print("First embedding sample:", embeddings[0][:5])

embeddings_array = np.array(embeddings).astype("float32")

dimension = embeddings_array.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings_array)

print("FAISS index size:", index.ntotal)

