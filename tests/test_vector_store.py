"""Test module for VectorStore functionality."""

# pylint: disable=consider-using-with

import io
import logging
import os
import tempfile
import unittest
from unittest.mock import patch

import yaml
from llama_index.core.base.llms.types import ChatResponse
from reportlab.pdfgen import canvas

from curia.rag import VectorStore


class TestVectorStore(unittest.TestCase):
    """Test case for VectorStore class."""

    def setUp(self):
        """Set up test environment."""
        logging.basicConfig(level=logging.INFO)
        logging.getLogger("chromadb").setLevel(logging.CRITICAL)
        logging.getLogger("ollama").setLevel(logging.CRITICAL)
        logging.getLogger("httpx").setLevel(logging.CRITICAL)

        # Create a temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()

        # Define the path for the temporary config.yaml file
        config_path = os.path.join(self.temp_dir.name, "config.yaml")

        # Create data directories
        raw_dir = os.path.join(self.temp_dir.name, "raw")
        os.makedirs(raw_dir, exist_ok=True)
        databases_dir = os.path.join(self.temp_dir.name, "databases")
        os.makedirs(databases_dir, exist_ok=True)

        # Define the content for the config.yaml file
        config_content = {
            "data": {
                "data_path": raw_dir,
                "db_path": databases_dir,
                "collection_name": "temp_docs",
            },
            "models": {"embed_model_name": "dummy", "llm_name": "dummy"},
        }

        # Write the content to the temporary config.yaml file
        with open(config_path, "w", encoding="utf-8") as config_file:
            yaml.dump(config_content, config_file)

        self._create_sample_pdf("0", "test.pdf")
        self._create_sample_pdf("1", "test2.pdf")
        self._create_sample_pdf("2", "test3.pdf")

        self._dummy_models()

        self.vector_store_new = VectorStore(config_path, restart_database=True)
        self.vector_store_repeat = VectorStore(
            config_path, restart_database=False
        )

    def tearDown(self):
        """Clean up the temporary directory."""
        self.temp_dir.cleanup()

    def _dummy_models(self):
        """Mock models for testing."""
        patch(
            ".".join(
                [
                    "curia.rag.vector_store.OllamaEmbedding",
                    "get_general_text_embedding",
                ]
            ),
            side_effect=lambda text: [int(text[-1])],
        ).start()
        patch(
            "curia.rag.vector_store.Ollama.chat",
            side_effect=lambda x: ChatResponse(message=x[1]),
        ).start()

    def _create_sample_pdf(self, text: str, file_name: str):
        """Generate a test PDF with known content.

        Args:
            text (str): Text to write in the PDF.
            file_name (str): Name of the PDF file.
        """
        pdf_path = os.path.join(self.temp_dir.name, "raw", file_name)

        packet = io.BytesIO()
        canv = canvas.Canvas(packet)

        # Page 1
        canv.drawString(50, 700, text)
        canv.save()

        packet.seek(0)
        with open(pdf_path, "wb") as file:
            file.write(packet.getvalue())

    def test_retrieve(self):
        """Test the retrieve method of VectorStore."""
        for text in ["0", "1", "2"]:
            self.assertEqual(
                self.vector_store_new.retrieve(text)[0].node.text, text
            )
            self.assertEqual(
                self.vector_store_repeat.retrieve(text)[0].node.text, text
            )

    def test_query(self):
        """Test the query method of VectorStore."""
        for text in ["0", "1", "2"]:
            self.assertIn(text, self.vector_store_new.query(text).response)
            self.assertIn(text, self.vector_store_repeat.query(text).response)
