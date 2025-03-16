"""
ChromaDB vector store manager with logging and error handling.
"""

import logging
from typing import Any, Dict, List

import chromadb
import numpy as np
import ollama


class VectorStoreManager:
    """
    Manages ChromaDB vector store operations with Ollama embeddings.

    Args:
        db_path (str): Path to ChromaDB persistence directory
        collection_name (str): Name of the Chroma collection
        embed_model_name (str): Ollama embedding model name
    """

    def __init__(
        self,
        db_path: str = "data/databases/",
        collection_name: str = "curia_docs",
        embed_model: str = "all-minilm:l6-v2",
    ):
        self.db_path = db_path
        self.collection_name = collection_name
        self.embed_model = embed_model

        self.logger = logging.getLogger("VectorStoreManager")

        self.init_components()

    def init_components(self) -> None:
        """Initialize ChromaDB client and embedding model."""
        try:
            self.logger.info("Initializing ChromaDB client: %s", self.db_path)
            self.chroma_client = chromadb.PersistentClient(path=self.db_path)

            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name
            )
            self.logger.info("Collection %s ready", self.collection_name)

        except Exception as error:
            self.logger.error("Initialization failed: %s", str(error))
            raise

    def store_document(
        self, chunks: List[str], metadata: Dict[str, Any], batch_size: int = 1
    ) -> None:
        """
        Store document chunks with embeddings and metadata.

        Args:
            chunks (List[str]): List of text chunks to store
            metadata (Dict[str, Any]): Document-level metadata
            batch_size (int): Number of chunks to process at once
        """
        try:
            total_chunks = len(chunks)
            self.logger.info(
                "Storing %d chunks for document %s",
                total_chunks,
                metadata["source"],
            )

            # Process in batches to manage memory
            for batch_idx in range(0, total_chunks, batch_size):
                batch = chunks[batch_idx : batch_idx + batch_size]
                # Corrected embeddings extraction
                batch_embeddings_response = ollama.embed(
                    model=self.embed_model, input=batch
                )

                batch_embeddings = np.array(
                    [
                        np.array(embedding)
                        for embedding in batch_embeddings_response.embeddings
                    ]
                )

                batch_ids = [
                    f"{metadata['source']}_chunk_{batch_idx + i}"
                    for i in range(len(batch))
                ]

                self.collection.add(
                    documents=batch,
                    embeddings=batch_embeddings,
                    ids=batch_ids,
                    metadatas=[metadata] * len(batch),
                )

                self.logger.debug(
                    "Stored batch %d-%d", batch_idx, batch_idx + len(batch)
                )

            self.logger.info("Successfully stored %d chunks", total_chunks)

        except Exception as error:
            self.logger.error("Failed to store document: %s", str(error))
            raise


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("chromadb").setLevel(
        logging.CRITICAL
    )  # Suppress logs from ChromaDB
    logging.getLogger("ollama").setLevel(logging.CRITICAL)

    vsm = VectorStoreManager()
    vsm.store_document(
        chunks=["Sample legal text chunk 1", "Sample chunk 2"],
        metadata={"source": "test_case.pdf"},
    )
    # Example query
    PROMPT = "What is the second sample?"

    # Generate embedding for the query and retrieve results
    response = ollama.embed(model="all-minilm:l6-v2", input=PROMPT)
    results = vsm.collection.query(
        query_embeddings=response["embeddings"],  # Fixed typo in key name
        n_results=1,
    )
    # Safe access to documents with direct key access
    documents = results["documents"]
    data = documents[0][0] if documents else None
    print(data)
