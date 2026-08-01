import numpy as np

def retrieve_chunks(query, model, index, chunks, k=2):
    
    # cnovert questioon to embedding
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    # search FAISS
    distances, indices = index.search(query_embedding, k)

    #get  relevant chunks
    results = []
    for i in indices[0]:
        results.append(chunks[i].page_content)
    
    return results