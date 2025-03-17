"""A module for managing a vector store using ChromaDB and Ollama embeddings.

This module provides a `VectorStore` class that uses a ChromaDB collection,
loads documents from a specified directory,
and processes them using Ollama embeddings.
It supports querying and retrieving documents from the vector store.
"""

import json
import logging
import os

import chromadb
import yaml
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import Document
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

from curia.params import CONFIG_PATH


class VectorStore:
    """A vector store implementation using ChromaDB and Ollama."""

    def __init__(
        self,
        config_path: str = CONFIG_PATH,
        restart_database: bool = False,
        **kwargs,
    ):
        """Initialize the VectorStore.

        Args:
            config_path (str): Path to the file containing project config.
            restart_database (bool): Whether to restart the database.
        """
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        data_path = kwargs.get(
            "data_path", config.get("data", {}).get("data_path", "data/raw/")
        )
        db_path = kwargs.get(
            "db_path", config.get("data", {}).get("db_path", "data/databases/")
        )
        collection_name = kwargs.get(
            "collection_name",
            config.get("data", {}).get("collection_name", "curia_docs"),
        )
        embed_model_name = kwargs.get(
            "embed_model_name",
            config.get("models", {}).get(
                "embed_model_name", "all-minilm:l6-v2"
            ),
        )
        llm_name = kwargs.get(
            "llm_name",
            config.get("models", {}).get("llm_name", "initium/law_model"),
        )

        self.paths = {"data": data_path, "db": db_path}
        self.collection_name = collection_name
        self.model_names = {"embed_model": embed_model_name, "llm": llm_name}
        self.restart_database = restart_database

        self.logger = logging.getLogger("VectorStore")

        collection = self.init_database()
        embed_model, llm = self.init_model()
        index = self.setup_vector_store(collection, embed_model)
        self.query_engine, self.retrieval_engine = self._setup_query_engines(
            index, llm
        )

    def init_database(self) -> chromadb.Collection:
        """Initialize the ChromaDB client and collection.

        Returns:
            chromadb.Collection: A chromaDB collection of documents.

        Raises:
            Exception: If initialization fails.
        """
        try:
            client = chromadb.PersistentClient(path=self.paths["db"])
            collection = client.get_or_create_collection(self.collection_name)
            self.logger.info("Collection %s ready", self.collection_name)
            return collection
        except Exception as error:
            self.logger.error("Database initialization failed: %s", str(error))
            raise

    def init_model(self) -> tuple[OllamaEmbedding, Ollama]:
        """Initialize the embedding model and LLM.

        Returns:
            OllamaEmbedding: Model for embedding text.
            llm: Model for user queries.

        Raises:
            Exception: If model initialization fails.
        """
        try:
            embed_model = OllamaEmbedding(
                model_name=self.model_names["embed_model"]
            )
            self.logger.info("Model %s ready", self.model_names["embed_model"])

            llm = Ollama(model=self.model_names["llm"], request_timeout=120.0)
            self.logger.info("Model %s ready", self.model_names["llm"])

            return embed_model, llm

        except Exception as error:  # pylint: disable=broad-except
            self.logger.error("Model initialization failed: %s", str(error))
            raise

    def setup_vector_store(
        self, collection: chromadb.Collection, embed_model: OllamaEmbedding
    ) -> VectorStoreIndex:
        """Set up the vector store, process new documents, and create indices.

        Args:
            chromadb.Collection: ChromaDB collection.
            OllamaEmbedding: Text embedding model.

        Returns:
            VectorStoreIndex: Created index.

        Raises:
            Exception: If vector storage setup fails.
        """
        try:
            processed_files_record = os.path.join(
                self.paths["db"], "processed.json"
            )
            processed_files = self._load_processed_files(
                processed_files_record
            )
            all_files = self._get_all_files()
            new_files = self._get_new_files(processed_files, all_files)
            documents = self._load_documents(new_files)
            index = self._create_index(
                new_files,
                documents,
                collection,
                embed_model,
            )
            self._update_and_save_processed_files(
                processed_files_record, processed_files, all_files
            )
            self.logger.info("Vectors ready")
            return index
        except Exception as error:  # pylint: disable=broad-except
            self.logger.error("Vector storage failed: %s", str(error))
            raise

    def _load_processed_files(self, processed_files_record: str) -> dict:
        """Load processed files metadata from JSON.

        Args:
            processed_files_record (str): Path to the processed files JSON.

        Returns:
            dict: Dictionary of processed files and their modification times.
        """
        if (
            os.path.exists(processed_files_record)
            and not self.restart_database
        ):
            with open(
                processed_files_record, "r", encoding="utf-8"
            ) as file_handle:
                return json.load(file_handle)
        return {}

    def _get_all_files(self) -> dict:
        """Retrieve all files in the data directory with modification times.

        Returns:
            dict: Dictionary mapping filenames to modification times.
        """
        return {
            file: os.path.getmtime(os.path.join(self.paths["data"], file))
            for file in os.listdir(self.paths["data"])
        }

    def _get_new_files(self, processed_files: dict, all_files: dict) -> list:
        """Identify new or modified files.

        Args:
            processed_files (dict): Previously processed files.
            all_files (dict): Current files with modification times.

        Returns:
            list: List of new or modified filenames.
        """
        return [
            file
            for file, mtime in all_files.items()
            if file not in processed_files
            or mtime > processed_files.get(file, 0)
        ]

    def _load_documents(self, new_files: list):
        """Load documents from new files using SimpleDirectoryReader.

        Args:
            new_files (list): List of new or modified filenames.

        Returns:
            list: Loaded documents, or None if no new files.
        """
        if not new_files:
            return None
        reader = SimpleDirectoryReader(
            input_files=[
                os.path.join(self.paths["data"], file) for file in new_files
            ]
        )
        return reader.load_data()

    def _create_index(
        self,
        new_files: list[str],
        documents: list[Document],
        collection: chromadb.Collection,
        embed_model: OllamaEmbedding,
    ) -> VectorStoreIndex:
        """Create VectorStoreIndex from documents or existing vector store.

        Args:
            new_files (list): List of new or modified filenames.
            list[Document]: Loaded documents.
            chromadb.Collection: ChromaDB collection.
            OllamaEmbedding: Text embedding model.

        Returns:
            VectorStoreIndex: Created index.
        """
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )

        if new_files:
            return VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                embed_model=embed_model,
            )
        return VectorStoreIndex.from_vector_store(
            vector_store, embed_model=embed_model
        )

    def _update_and_save_processed_files(
        self,
        processed_files_record: str,
        processed_files: dict,
        all_files: dict,
    ) -> None:
        """Update processed files metadata and save to JSON.

        Args:
            processed_files_record (str): Path to the JSON record.
            processed_files (dict): Previously processed files.
            all_files (dict): Current files with modification times.
        """
        processed_files.update(all_files)
        with open(
            processed_files_record, "w", encoding="utf-8"
        ) as file_handle:
            json.dump(processed_files, file_handle)

    def _setup_query_engines(
        self, index: VectorStoreIndex, llm: Ollama
    ) -> tuple[BaseQueryEngine, BaseRetriever]:
        """Set up query and retrieval engines from the index.

        Args:
            index (VectorStoreIndex): Index to create engines from.
            llm (Ollama): Language model for user queries.

        Returns:
            BaseQueryEngine: Query engine.
            BaseRetriever: Eetrieval engine.
        """
        query_engine = index.as_query_engine(llm=llm)
        retrieval_engine = index.as_retriever()

        return query_engine, retrieval_engine

    def query(self, query: str):
        """Query the vector store.

        Args:
            query (str): Query string.

        Returns:
            Query result.
        """
        return self.query_engine.query(query)

    def retrieve(self, query: str):
        """Retrieve documents from the vector store.

        Args:
            query (str): Query string.

        Returns:
            Retrieved documents.
        """
        return self.retrieval_engine.retrieve(query)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("chromadb").setLevel(logging.CRITICAL)
    logging.getLogger("ollama").setLevel(logging.CRITICAL)
    logging.getLogger("httpx").setLevel(logging.CRITICAL)
    vectors = VectorStore(restart_database=False)
    print(
        vectors.retrieve(
            """What is the folowing case about? Court under Article 177 of the
            EEC Treaty by the Oberlandes­gericht Karlsruhe"""
        )
    )
