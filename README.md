# Bijulee Knowledge Assistant

This repository contains a small example Streamlit web app and ingestion utilities that use LlamaIndex (llama-index) to parse PDFs, build both a vector index and a Knowledge Graph index (Graph RAG style), persist them to disk, and provide an interactive QA UI.

Files added
- requirements.txt — Python dependencies
- ingest.py — PDF ingestion, document creation, build & persist vector + knowledge-graph indices, and query helpers
- app.py — Streamlit UI to upload PDFs, run ingestion, load indices, and ask questions (Vector RAG + Graph)

Quickstart

1. Clone the repo

   git clone https://github.com/satya120781/BijuleeKnowledgeAssistant.git
   cd BijuleeKnowledgeAssistant

2. Create a virtual environment and install dependencies

   python -m venv .venv
   source .venv/bin/activate   # macOS / Linux
   .\.venv\Scripts\activate  # Windows
   pip install --upgrade pip
   pip install -r requirements.txt

3. Set your OpenAI API key (or configure another LLM/embedding provider in ingest.py)

   export OPENAI_API_KEY="sk-..."   # macOS / Linux
   setx OPENAI_API_KEY "sk-..."     # Windows (restart terminal)

4. Run the Streamlit app

   streamlit run app.py

   The web UI will open in your browser. Use the sidebar to upload PDFs and ingest them. After ingestion, load indices into memory and ask questions.

Notes and troubleshooting

- LlamaIndex (llama-index) evolves quickly; if you see import errors, check the version you installed and adapt the imports:
  - ServiceContext, LLMPredictor, index classes, and API signatures can change between releases.

- PDF parsing
  - This example uses PyPDF2 for basic text extraction. For more accurate layout/table extraction consider `unstructured`, `pdfminer.six`, or OCR (Tesseract) for scanned PDFs.

- Embeddings & LLM
  - The example uses OpenAI for both embeddings and LLMs. You can replace OpenAI with a local model or another provider by changing `build_service_context` in ingest.py.

- Streaming responses
  - To stream token-by-token responses in Streamlit you need to enable streaming on your LLM client and wire llama-index callbacks to Streamlit. I can add an example callback that updates a Streamlit placeholder as tokens arrive.

- Persistence
  - Indices are persisted under the `storage/` directory by default: `storage/vector` and `storage/graph`.

Security

- Do not commit your API keys to the repository. Use environment variables or a secrets manager.

Next steps I can help with

- Add token streaming callbacks for real-time UI updates in Streamlit.
- Swap OpenAI for a local LLM + local embeddings (Hugging Face + text-generation-inference / llama.cpp) for an offline setup.
- Create a feature branch, add CI (linting, type checks), and a GitHub Actions workflow to run tests or format checks.

If you want any of the above, tell me which and I’ll implement it next.
