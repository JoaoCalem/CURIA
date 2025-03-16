"""
PDF Processor: Extracts and chunks text from single-column PDFs.
"""

# pylint: disable=R0801

import logging
from pathlib import Path
from typing import List

from pypdf import PdfReader


class PDFProcessor:
    """
    Processes PDFs into text chunks with configurable size and overlap.

    Args:
        chunk_size: Words per chunk (default: 512)
        overlap: Overlapping words between chunks (default: 64)
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        """Initializes processor with chunking parameters."""
        if overlap >= chunk_size:
            logging.warning("Overlap should be smaller than chunk size")

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.logger = logging.getLogger("PDFProcessor")
        logging.basicConfig(level=logging.INFO)

    def process_pdf(self, pdf_path: Path) -> List[str]:
        """
        Extracts and chunks text from a PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of text chunks

        Raises:
            FileNotFoundError, PDFReadError, ValueError
        """
        try:
            self.logger.info("Processing %s", pdf_path)
            text = self.extract_text(pdf_path)
            return self.chunk_text(text)
        except Exception as excpetion:
            self.logger.error("Processing failed: %s", str(excpetion))
            raise

    def extract_text(self, pdf_path: Path) -> str:
        """
        Extracts text from PDF, removing newlines.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Cleaned text content

        Note: Assumes single-column documents.
        """
        text = []
        with open(pdf_path, "rb") as file:
            reader = PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)

        if not text:
            raise ValueError("No extractable text found")

        return " ".join(text).replace("\n", " ")

    def chunk_text(self, text: str) -> List[str]:
        """
        Splits text into chunks with overlap.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        words = text.split()
        chunks = []
        start = 0

        while start < len(words) - self.overlap:
            end = start + self.chunk_size
            chunks.append(" ".join(words[start:end]))
            start = end - self.overlap

        return chunks
