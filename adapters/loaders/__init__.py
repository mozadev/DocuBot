"""Document loaders, one per supported file type."""

from adapters.loaders.docx_loader import DOCXLoader
from adapters.loaders.pdf_loader import PDFLoader
from adapters.loaders.text_loader import TextLoader

__all__ = ["DOCXLoader", "PDFLoader", "TextLoader"]
