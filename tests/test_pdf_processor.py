"""
Tests for PDFProcessor class using unittest

Requires:
- pypdf
- reportlab (for generating test PDFs)
"""

import io
import tempfile
import unittest
from pathlib import Path

from reportlab.pdfgen import canvas

from src.rag.pdf_processor import PDFProcessor


class TestPDFProcessor(unittest.TestCase):
    """Test PDF Processor"""

    def _create_sample_pdf(self, directory: Path) -> Path:
        """Helper to generate a test PDF with known content"""
        pdf_path = directory / "test.pdf"

        packet = io.BytesIO()
        canv = canvas.Canvas(packet)

        # Page 1
        canv.drawString(50, 700, "This is a test PDF document.")
        canv.drawString(50, 680, "It contains sample text for unit testing.")
        canv.showPage()

        # Page 2
        canv.drawString(50, 700, "Second page of test content.")
        canv.drawString(50, 680, "More sample text for verification.")
        canv.save()

        packet.seek(0)
        with open(pdf_path, "wb") as file:
            file.write(packet.getvalue())

        return pdf_path

    def test_text_extraction(self):
        """Test PDF text extraction"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = self._create_sample_pdf(Path(tmp_dir))
            processor = PDFProcessor()
            text = processor.extract_text(pdf_path)

            self.assertIn("test PDF document", text)
            self.assertIn("Second page", text)
            self.assertNotIn("\n", text)

    def test_chunking_basic(self):
        """Test basic chunking functionality"""
        processor = PDFProcessor(chunk_size=5, overlap=2)
        text = "a b c d e f g h i j k"
        chunks = processor.chunk_text(text)

        self.assertEqual(chunks, ["a b c d e", "d e f g h", "g h i j k"])

    def test_chunk_edge_cases(self):
        """Test chunking with edge cases"""
        processor = PDFProcessor(chunk_size=3, overlap=1)

        # Test text shorter than chunk size
        self.assertEqual(processor.chunk_text("a b"), ["a b"])

        # Test exact chunk size match
        self.assertEqual(processor.chunk_text("a b c"), ["a b c"])

        # Test one word overlap
        self.assertEqual(processor.chunk_text("a b c d"), ["a b c", "c d"])
