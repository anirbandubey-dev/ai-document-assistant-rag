from rag.ingest import load_and_chunk
from rag.embed import create_embeddings, model
from rag.vector_store import create_faiss_index
from rag.retrieve import retrieve_chunks
from rag.generate import generate_answer

pdf_path = "database/data/Deepdive.pdf"

# Load and Chunk
chunks = load_and_chunk(pdf_path)
print("Chunks created:", len(chunks))

# Embeddings
embeddings = create_embeddings(chunks)
print("Embeddings created:", len(embeddings))

# FAISS Index
index = create_faiss_index(embeddings)
print("FAISS index size:", index.ntotal)

# Ask questions
query = input("\nAsk a question: ")

results = retrieve_chunks(query, model, index, chunks)
 
print("\nRelevant chunks:\n")
for r in results:
    print(r)
    print("------")

context = "\n".join(results)

answer = generate_answer(query, context)

print("\nAI Answer:\n")
print(answer)