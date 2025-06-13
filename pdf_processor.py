import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    with fitz.open(pdf_path) as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text
