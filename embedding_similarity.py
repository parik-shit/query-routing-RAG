from langchain.embeddings import OllamaEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingSimilarity:
    def __init__(self, model_name="llama"):
        self.embeddings_model = OllamaEmbeddings(model_name=model_name)

    def generate_embeddings(self, text_chunks):
        return self.embeddings_model.embed_documents(text_chunks)

    def compute_similarity(self, query, document_embeddings):
        query_embedding = self.embeddings_model.embed_query(query)
        similarities = cosine_similarity([query_embedding], document_embeddings)
        return max(similarities[0])

