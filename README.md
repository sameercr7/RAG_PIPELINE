# UP Police Matrix - RAG Pipeline

This project implements a Retrieval-Augmented Generation (RAG) pipeline for intelligence monitoring. It synchronizes analyzed data from a MySQL database into a ChromaDB vector store and provides a chat interface for querying the intelligence.

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.9+ installed and a MySQL database running with the `analyzed_data` table.

### 2. Installation
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 3. Configuration
Edit the `.env` file or `config.py` with your credentials:
- **MySQL**: Host, Port, User, Password, DB.
- **HuggingFace**: `HF_API_TOKEN` (for the LLM).
- **ChromaDB**: Persistence directory path.

---

## 🛠️ How to Run

You need to run **two separate processes** for the system to work fully:

### A. The Ingestion Worker
This process moves data from MySQL to ChromaDB in the background.
```bash
python worker.py
```
- It checks `status_state.json` to see which sources are enabled.
- It processes data in batches of 100.
- It converts text to vectors using `SentenceTransformer`.

### B. The API Server
This process serves the Chat UI and the Admin Dashboard.
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```
- **Chat UI**: Open `http://localhost:8000` in your browser.
- **Admin Dashboard**: Open `http://localhost:8000/admin` to monitor ingestion.

---

## 📂 Project Structure & Flow

### 1. Data Ingestion Flow (`worker.py`)
- **Source**: MySQL `analyzed_data` table.
- **Filter**: Rows with `ingestion_status = 'PENDING'`.
- **Process**:
    1. Fetch rows.
    2. Convert to document text + metadata.
    3. Generate embeddings using `all-MiniLM-L6-v2`.
    4. Upsert into **ChromaDB**.
    5. Update MySQL status to `'INGESTED'`.

### 2. RAG Query Flow (`rag.py` & `api.py`)
- **User Query**: User sends a question via the frontend.
- **Retrieval**: `rag.py` searches ChromaDB for the top-K most similar chunks.
- **Generation**: The retrieved context + query is sent to a **Llama 3.1** model (via HuggingFace Inference API).
- **Response**: The answer and cited sources are returned to the user.

### 3. Core Files
- `api.py`: FastAPI application with endpoints.
- `worker.py`: Background synchronization worker.
- `rag.py`: Core logic for retrieval and LLM interaction.
- `db.py`: Database connection and helper functions.
- `config.py`: Environment and global settings.
- `admin.html`: Monitoring dashboard.
- `index.html`: Main chat interface.

---

## 🛡️ Admin Controls
You can enable/disable ingestion for specific sources (WhatsApp, YouTube, etc.) through the Admin Dashboard (`/admin`). These settings are persisted in `status_state.json`.
