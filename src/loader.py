import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader


def load_pdfs_from_uploads(uploaded_files: list) -> list:
    """
    Accept a list of Streamlit UploadedFile objects.
    Write each to a temp file, load with PyPDFLoader, return all documents.
    """
    all_documents = []

    for uploaded_file in uploaded_files:
        suffix = os.path.splitext(uploaded_file.name)[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()

            # Tag each document with the original filename
            for doc in docs:
                doc.metadata["source_filename"] = uploaded_file.name

            all_documents.extend(docs)
        finally:
            os.unlink(tmp_path)

    return all_documents
