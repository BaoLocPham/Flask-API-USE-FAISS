import faiss
import pandas as pd

class FAISS():
    def __init__(self, sentence_embeddings) -> None:
        self.index = faiss.IndexFlatL2(sentence_embeddings.shape[1])
    
    def add(self, questions_bank_embedded):
        self.index.add(questions_bank_embedded)

    def search(self, encoded_question, top_k=5):
        top_k = min(top_k, len(encoded_question))
        D, I = self.index.search(encoded_question, top_k)
        return D, I