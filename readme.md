Here's a basic documentation file for your project. This file explains the setup process, the directory structure, and how to use the different components of your project.

### **Documentation for PDF Query Routing Project**

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Directory Structure](#directory-structure)
3. [Setup Instructions](#setup-instructions)
   - [1. Install Prerequisites](#1-install-prerequisites)
   - [2. Setup Virtual Environment](#2-setup-virtual-environment)
4. [How to Run the Project](#how-to-run-the-project)
5. [Code Overview](#code-overview)
   - [1. `pdf_text_extractor.py`](#1-pdf_text_extractorpy)
   - [2. `embedding_similarity.py`](#2-embedding_similaritypy)
   - [3. `query_router.py`](#3-query_routerpy)
   - [4. `main.py`](#4-mainpy)
6. [License](#license)

---

## Project Overview

This project is designed to route user queries to the appropriate document (either a User Manual or Release Notes) by using machine learning techniques such as embeddings and cosine similarity. The main components of this project include extracting text from PDFs, generating embeddings for the text, and routing queries based on similarity.

---

## Directory Structure

The project directory is organized as follows:

```
/project-directory
    ├── main.py
    ├── pdf_text_extractor.py
    ├── embedding_similarity.py
    ├── query_router.py
    ├── requirements.txt
    └── setup_env.sh
```

- **`main.py`**: The main script that ties all components together and handles the query routing.
- **`pdf_text_extractor.py`**: Contains the `PDFTextExtractor` class for extracting text from PDF files.
- **`embedding_similarity.py`**: Contains the `EmbeddingSimilarity` class for generating embeddings and computing similarity.
- **`query_router.py`**: Contains the `QueryRouter` class for routing queries based on similarity scores.
- **`requirements.txt`**: Lists all the dependencies required for the project.
- **`setup_env.sh`**: A shell script to automate the setup of a virtual environment and the installation of dependencies.

---

## Setup Instructions

### 1. Install Prerequisites
Ensure you have Python 3.7 or later installed on your system. You can check your Python version by running:

```bash
python3 --version
```

### 2. Setup Virtual Environment

1. **Clone the Repository**:
   If you haven't already, clone the repository to your local machine.

   ```bash
   git clone <repository-url>
   cd project-directory
   ```

2. **Run the Setup Script**:
`python3 -m venv venv`
`pip install -r requirements.txt`
   - Install the required Python packages listed in `requirements.txt`.

3. **Activate the Virtual Environment**:
   To start working with the project, activate the virtual environment:

   ```bash
   source venv/bin/activate
   ```

   To deactivate the environment, simply run:

   ```bash
   deactivate
   ```

---

## How to Run the Project

Once the setup is complete, you can run the main script to process queries:

```bash
python main.py
```

This will execute the script, which will:
- Extract text from the provided PDF files.
- Generate embeddings for the extracted text.
- Route example queries to the appropriate document based on similarity scores.

---

## Code Overview

### 1. `pdf_text_extractor.py`
This module contains the `PDFTextExtractor` class, which is responsible for extracting text from a given PDF file.

```python
import PyPDF2

class PDFTextExtractor:
    @staticmethod
    def extract_text_from_pdf(pdf_path):
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        return text
```

### 2. `embedding_similarity.py`
This module contains the `EmbeddingSimilarity` class, which handles the generation of text embeddings and the computation of cosine similarity.

```python
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
```

### 3. `query_router.py`
This module contains the `QueryRouter` class, which determines where to route a user query based on similarity scores.

```python
class QueryRouter:
    def __init__(self, user_manual_embeddings, release_notes_embeddings):
        self.user_manual_embeddings = user_manual_embeddings
        self.release_notes_embeddings = release_notes_embeddings

    def route_query(self, query, embedding_similarity):
        max_similarity_user_manual = embedding_similarity.compute_similarity(query, self.user_manual_embeddings)
        max_similarity_release_notes = embedding_similarity.compute_similarity(query, self.release_notes_embeddings)

        if max_similarity_user_manual > max_similarity_release_notes:
            return "Route query to User Manual and Troubleshooting Guide"
        else:
            return "Route query to Release Notes and Updates"
```

### 4. `main.py`
The main script that ties everything together. It handles the extraction of text, embedding generation, and routing of queries.

```python
from pdf_text_extractor import PDFTextExtractor
from embedding_similarity import EmbeddingSimilarity
from query_router import QueryRouter

def main():
    # Paths to your PDF files
    user_manual_pdf = 'user_manual.pdf'
    release_notes_pdf = 'release_notes.pdf'

    # Extract text from the PDFs
    extractor = PDFTextExtractor()
    user_manual_text = extractor.extract_text_from_pdf(user_manual_pdf)
    release_notes_text = extractor.extract_text_from_pdf(release_notes_pdf)

    # Split the text into smaller chunks for more granular embeddings
    user_manual_chunks = user_manual_text.split("\n\n")
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
```

---

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

---

This documentation provides an overview of the project structure, setup instructions, and a breakdown of each module. It should help both you and others understand and use the project efficiently.
