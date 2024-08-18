class QueryRouter:
    def __init__(self, book_one_embeddings, book_two_embeddings):
        self.book_one_embeddings = book_one_embeddings
        self.book_two_embeddings = book_two_embeddings

    def route_query(self, query, embedding_similarity):
        max_similarity_user_manual = embedding_similarity.compute_similarity(query, self.book_one_embeddings)
        max_similarity_release_notes = embedding_similarity.compute_similarity(query, self.book_two_embeddings)

        if max_similarity_user_manual > max_similarity_release_notes:
            return "Book One"
        else:
            return "Book Two"

