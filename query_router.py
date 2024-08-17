class QueryRouter:
    def __init__(self, user_manual_embeddings, release_notes_embeddings):
        self.user_manual_embeddings = user_manual_embeddings
        self.release_notes_embeddings = release_notes_embeddings

    def route_query(self, query, embedding_similarity):
        max_similarity_user_manual = embedding_similarity.compute_similarity(query, self.user_manual_embeddings)
        max_similarity_release_notes = embedding_similarity.compute_similarity(query, self.release_notes_embeddings)

        if max_similarity_user_manual > max_similarity_release_notes:
            return "Book One"
        else:
            return "Book Two"

