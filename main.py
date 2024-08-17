from pdf_text_extractor import PDFTextExtractor
from embedding_similarity import EmbeddingSimilarity
from query_router import QueryRouter

def main():
    # Paths to your PDF files
    user_manual_pdf = './documents/Book_One.pdf'  # Path to the User Manual and Troubleshooting Guide
    release_notes_pdf = './documents/Book_Two.pdf'  # Path to the Release Notes and Updates

    # Extract text from the PDFs
    extractor = PDFTextExtractor()
    user_manual_text = extractor.extract_text_from_pdf(user_manual_pdf)
    release_notes_text = extractor.extract_text_from_pdf(release_notes_pdf)

    # Split the text into smaller chunks for more granular embeddings
    user_manual_chunks = user_manual_text.split("\n\n")  # Split text into paragraphs or other chunks
    release_notes_chunks = release_notes_text.split("\n\n")

    # Initialize the embedding similarity class
    embedding_similarity = EmbeddingSimilarity(model_name="llama3.1")

    # Generate embeddings for each chunk
    user_manual_embeddings = embedding_similarity.generate_embeddings(user_manual_chunks)
    release_notes_embeddings = embedding_similarity.generate_embeddings(release_notes_chunks)

    # Initialize the query router


    ########################################################################

    query1 = "What is the central premise of 'I Sell My Dreams'"
    routing_decision1 = router.route_query(query1, embedding_similarity)
    print(routing_decision1)

    query2 = "What is the narrator’s initial reaction to the yellow wallpaper in her room?"
    routing_decision2 = router.route_query(query2, embedding_similarity)
    print(routing_decision2)

if __name__ == "__main__":
    main()
