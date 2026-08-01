import faiss
import numpy as np
def create_faiss_index(embeddings):

    embeddings_array = np.array(embeddings).astype("float32")

    dimension = embeddings_array.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings_array)

    return index