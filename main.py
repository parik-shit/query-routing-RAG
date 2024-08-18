from pdf_text_extractor import PDFTextExtractor
from embedding_similarity import EmbeddingSimilarity
from query_router import QueryRouter

def main():
    # Paths to PDF files
    book_one_pdf = './documents/Book_One.pdf'  # Path to the User Manual and Troubleshooting Guide
    book_two_pdf = './documents/Book_Two.pdf'  # Path to the Release Notes and Updates

    # Extract text from the PDFs
    extractor = PDFTextExtractor()
    book_one_text = extractor.extract_text_from_pdf(book_one_pdf)
    book_two_text = extractor.extract_text_from_pdf(book_two_pdf)

    # Split the text into smaller chunks for more granular embeddings
    book_one_chunks = book_one_text.split("\n\n")  # Split text into paragraphs or other chunks
    book_two_chunks = book_two_text.split("\n\n")

    # Initialize the embedding similarity class
    embedding_similarity = EmbeddingSimilarity(model_name="llama3.1")

    # Generate embeddings for each chunk
    book_one_embeddings = embedding_similarity.generate_embeddings(book_one_chunks)
    book_two_embeddings = embedding_similarity.generate_embeddings(book_two_chunks)

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

