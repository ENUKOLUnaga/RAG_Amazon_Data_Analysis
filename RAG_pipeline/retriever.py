import faiss
import numpy as np


class Retriever:
    def __init__(self, embeddings, documents):
        self.docs = documents
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.array(embeddings))

    def search(self, query_vec, k=5):
        _, idx = self.index.search(query_vec, k)
        return [self.docs[i] for i in idx[0]]