# Local RAG Multi-Agent System for Legal Judgments

## Overview

This project aims to build a local Retrieval-Augmented Generation (RAG) system powered by a multi-agent architecture to query and understand European Court of Justice (ECJ) law judgments. The system will allow users to ask questions about the documents and receive synthesized answers based on relevant information retrieved from the judgments.

The design incorporates a multi-agent system to summarize documents at different levels of detail. When a user asks a question, the system will strategically retrieve information, starting with high-level summaries and progressively accessing more detailed summaries or full document chunks as needed to provide a comprehensive answer.

All components are designed to run locally, utilizing local models and storage solutions.

## Key Features (Planned)

* **PDF Document Ingestion:** Ability to load and process PDF files of ECJ judgments from a local directory.
* **Multi-Level Document Summarization:** Automatic generation of summaries at varying levels of detail (e.g., high-level, section-specific).
* **Local Storage:** Utilizes ChromaDB for storing document embeddings and potentially SQLite for other structured data.
* **Tiered Information Retrieval:** Intelligent retrieval of information starting from high-level summaries and drilling down to more detail if required.
* **Answer Synthesis:** Generation of coherent and informative answers to user queries based on retrieved information.
* **Simple Chat Interface:** A basic interface for users to interact with the system by asking questions.

## Technologies Used

* **RAG Framework:** [LlamaIndex](https://www.llamaindex.ai/)
* **Multi-Agent System:** [CrewAI](https://www.crewai.com/)
* **Local Large Language Model (LLM):** [Ollama](https://ollama.ai/) (running on local GPU with at least 10GB VRAM)
* **Vector Database:** [ChromaDB](https://www.trychroma.com/)
* **Optional Data Storage:** SQLite
* **Optional Pub/Sub Messaging:** [Redis](https://redis.io/)
* **Optional Analytics:** [DuckDB](https://duckdb.org/)
* **Programming Language:** Python

## Setup Instructions (Initial)

1.  **Install Dependencies:**
    ```bash
    pip install llama-index chromadb ollama crewai duckdb redis python-dotenv
    ```
2.  **Install and Run Ollama:**
    * Follow the instructions on the [Ollama website](https://ollama.ai/download) to install it for your operating system.
    * Run Ollama and ensure you have a suitable model downloaded (e.g., `llama2`). You can download a model using:
        ```bash
        ollama run llama2
        ```
3.  **Create Data Directory:**
    * Create a local directory to store your PDF files of ECJ judgments.
4.  **Project Structure (Initial):**
    ```
    your-project-directory/
    ├── data/
    │   └── # Your PDF files here
    ├── README.md
    └── # Your Python scripts will go here
    ```

## Current Status

This project is currently in the **initial setup and planning phase**. The immediate next steps involve:

* Setting up the basic environment and dependencies.
* Implementing the initial data loading and indexing using LlamaIndex (MVP - Milestone 1).
* Building a simple command-line chat interface to query the indexed data (MVP - Milestone 2).

## Future Milestones

* Implement multi-agent document summarization using CrewAI.
* Store summaries at different levels of detail.
* Develop the tiered retrieval logic within the multi-agent system.
* Integrate the summarization and retrieval agents for question answering.
* (Optional) Explore using SQLite, Redis, and DuckDB for specific functionalities.

## Learning Goals

This project is undertaken to learn and gain practical experience in:

* Retrieval-Augmented Generation (RAG) systems.
* Multi-agent system architectures.
* Specific Python packages: LlamaIndex, CrewAI, Ollama, ChromaDB, DuckDB, and Redis.
* Local deployment of LLMs and vector databases.