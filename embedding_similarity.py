# from langchain.embeddings import OllamaEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from langchain_ollama import OllamaEmbeddings

class EmbeddingSimilarity:
    def __init__(self, model_name="llama3.1"):
        self.embeddings_model = OllamaEmbeddings(model=model_name)

    def generate_embeddings(self, text_chunks):
        return self.embeddings_model.embed_documents(text_chunks)

    def compute_similarity(self, query, document_embeddings):
        query_embedding = self.embeddings_model.embed_query(query)
        similarities = cosine_similarity([query_embedding], document_embeddings)
        return max(similarities[0])

