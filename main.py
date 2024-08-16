from pdf_text_extractor import PDFTextExtractor
from embedding_similarity import EmbeddingSimilarity
from query_router import QueryRouter

def main():
    # Paths to your PDF files
    user_manual_pdf = 'user_manual.pdf'  # Path to the User Manual and Troubleshooting Guide
    release_notes_pdf = 'release_notes.pdf'  # Path to the Release Notes and Updates

    # Extract text from the PDFs
    extractor = PDFTextExtractor()
    user_manual_text = extractor.extract_text_from_pdf(user_manual_pdf)
    release_notes_text = extractor.extract_text_from_pdf(release_notes_pdf)

    # Split the text into smaller chunks for more granular embeddings
    user_manual_chunks = user_manual_text.split("\n\n")  # Split text into paragraphs or other chunks
    release_notes_chunks = release_notes_text.split("\n\n")

    # Initialize the embedding similarity class
    embedding_similarity = EmbeddingSimilarity(model_name="llama")

    # Generate embeddings for each chunk
    user_manual_embeddings = embedding_similarity.generate_embeddings(user_manual_chunks)
    release_notes_embeddings = embedding_similarity.generate_embeddings(release_notes_chunks)

    # Initialize the query router
    router = QueryRouter(user_manual_embeddings, release_notes_embeddings)

    # Example queries
    query1 = "How do I reset my password?"
    routing_decision1 = router.route_query(query1, embedding_similarity)
    print(routing_decision1)

    query2 = "What changes were made in the latest update?"
    routing_decision2 = router.route_query(query2, embedding_similarity)
    print(routing_decision2)

if __name__ == "__main__":
    main()

