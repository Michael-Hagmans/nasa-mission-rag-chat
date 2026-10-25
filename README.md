# NASA Mission RAG Chat

A retrieval-augmented generation (RAG) prototype for question answering over historical NASA mission records: Apollo 11, Apollo 13 and Challenger transcripts, flight plans and reports.

I built it to test how well a standard RAG stack copes with noisy, OCR-extracted source documents, and to measure answer quality with RAGAS instead of judging it by eye.

**Stack:** OpenAI (`text-embedding-3-small`, `gpt-4o`) · ChromaDB · RAGAS · Streamlit · Python 3.13

## Features

- **Embedding pipeline:** token-based chunking with overlap, batched OpenAI embeddings, stable chunk IDs and incremental updates (`skip` / `update` / `replace`)
- **Grounded answers:** the model may only use the retrieved documents and says so when the answer is not there
- **Conversation memory:** follow-up questions take earlier turns into account
- **Live evaluation:** every answer is scored for *faithfulness* and *answer relevancy* in the sidebar
- **Batch evaluation:** scores the full pipeline on a test question set
- **Retrieval inspection:** a CLI tool that shows the retrieved chunks with similarity scores and metadata

## How it works

```
data_text/*.txt
   │  embedding_pipeline.py: chunk (500 tokens, 100 overlap) → embed → store
   ▼
ChromaDB (chroma_db_openai/)
   │  rag_client.py: embed question → top-k similarity search → format context
   ▼
gpt-4o (llm_client.py): answer only from the retrieved context
   │
   ▼
RAGAS (ragas_evaluator.py): faithfulness + answer relevancy → Streamlit sidebar
```

## Project structure

```
.
├── chat.py                 # Streamlit chat app
├── embedding_pipeline.py   # Builds and updates the ChromaDB collection
├── rag_client.py           # Backend discovery, retrieval, context formatting
├── llm_client.py           # Grounded answer generation
├── ragas_evaluator.py      # RAGAS scoring (single answer and batch)
├── inspect_retrieval.py    # CLI: show retrieved chunks for a query
├── config.py               # Models, paths, endpoint; loads .env
├── test_questions.json     # Test questions for batch evaluation (JSON Lines)
├── requirements.txt
├── .env.example            # Template for your .env
├── data_text/              # Source documents (not in the repo, see Setup)
└── chroma_db_openai/       # Vector store (created by the pipeline, not in the repo)
```

## Setup

### Prerequisites

- Python 3.13 (other recent versions may work but are untested)
- An OpenAI API key with access to `gpt-4o` and `text-embedding-3-small`

### 1. Clone and create a virtual environment

**Windows (PowerShell)**
```powershell
git clone <repo-url>
cd <repo-folder>
python -m venv .venv
.venv\Scripts\Activate.ps1
```
If activation fails with *"running scripts is disabled"*, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and try again.

**macOS / Linux**
```bash
git clone <repo-url>
cd <repo-folder>
python3 -m venv .venv
source .venv/bin/activate
```

Your prompt should now start with `(.venv)`. All following commands assume the environment is active.

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Patch RAGAS (required)

RAGAS 0.4.3 imports Vertex AI classes from an outdated `langchain-community` path, so `import ragas` fails with `ImportError: cannot import name 'ChatVertexAI'`. This command rewrites the two imports in the installed package. It works on every OS and is safe to run more than once:

```bash
python -c "import importlib.util, pathlib; p = pathlib.Path(importlib.util.find_spec('ragas').origin).parent / 'llms' / 'base.py'; s = p.read_text(encoding='utf-8'); s = s.replace('from langchain_community.chat_models.vertexai import ChatVertexAI', 'from langchain_google_vertexai import ChatVertexAI').replace('from langchain_community.llms import VertexAI', 'from langchain_google_vertexai import VertexAI'); p.write_text(s, encoding='utf-8'); print('Patched', p)"
```

Verify:
```bash
python -W ignore -c "import ragas; print('RAGAS', ragas.__version__)"
```
Expected output: `RAGAS 0.4.3`. Run the patch again after reinstalling or upgrading RAGAS.

### 4. Configure your API key

Copy the template and fill in your key:

```powershell
Copy-Item .env.example .env     # Windows
```
```bash
cp .env.example .env            # macOS / Linux
```

```ini
OPENAI_API_KEY=sk-...
# Only needed for an OpenAI-compatible proxy; leave empty for api.openai.com
OPENAI_BASE_URL=
```

`.env` is listed in `.gitignore` and is never committed. Variables already set in your shell take precedence over `.env`.

### 5. Add the source documents

The documents are not part of this repository. Download the `apollo11`, `apollo13` and `challenger` folders from the [Udacity project starter](https://github.com/udacity/cd13318-exercises-project/tree/main/Project-NASA-Mission-Intelligence-Starter) and place them under `data_text/`:

```
data_text/
├── apollo11/     # 6 .txt files
├── apollo13/     # 3 .txt files
└── challenger/   # 3 .txt files
```

## Usage

### 1. Build the vector store

```bash
python embedding_pipeline.py
```

This chunks all 12 text files, embeds them and writes the collection to `chroma_db_openai/`. At the end, the log should show something like:

```
Files processed: 12
Total chunks created: <n>
Documents added to collection: <n>
Errors: 0
```

Re-running the command skips chunks that already exist. After changing chunk settings, rebuild with `--update-mode replace`.

| Option | Default | Description |
|---|---|---|
| `--data-path` | `data_text` | Folder containing the mission folders |
| `--chroma-dir` | `./chroma_db_openai` | Where the vector store is written |
| `--chunk-size` / `--chunk-overlap` | `500` / `100` | Chunk length and overlap in tokens |
| `--batch-size` | `50` | Chunks embedded per API call |
| `--update-mode` | `skip` | `skip`, `update` or `replace` existing chunks |
| `--stats-only` | | Print chunk counts per mission and category, then exit |
| `--test-query "<text>"` | | Run one similarity search after processing |
| `--delete-source <pattern>` | | Delete all chunks whose source name contains the pattern |

### 2. Start the chat app

```bash
streamlit run chat.py
```

Streamlit opens http://localhost:8501 in your browser. In the sidebar:

1. **Select Document Collection:** pick `chroma_db_openai / nasa_space_missions_text (<n> chunks)`.
2. **OpenAI API Key:** pre-filled from `.env`.
3. **Documents to retrieve:** number of chunks passed to the model (default 3).
4. **Enable RAGAS Evaluation:** scores each answer; this adds a few seconds and extra API calls per question.

Then ask a question, for example *"What happened to the oxygen tank on Apollo 13?"*

### 3. Inspect retrieval for a query

```bash
python inspect_retrieval.py "What happened to the oxygen tank on Apollo 13?" -n 5
python inspect_retrieval.py "Who was on the Apollo 11 crew?" --mission apollo_11 --show-context
```

This prints each retrieved chunk with its similarity score, mission, source file and chunk number. `--show-context` also prints the exact context sent to the model.

### 4. Run the batch evaluation

```bash
python ragas_evaluator.py                      # uses test_questions.json
python ragas_evaluator.py my_questions.jsonl   # or your own file
```

Each question is run through retrieval and answer generation and then scored. The script prints the average faithfulness and answer relevancy. The question file has one JSON object per line:

```json
{"question": "What was the overall goal of Apollo 11?"}
```

## Testing

There is no automated test suite yet. Run these checks in order after setup; each step depends on the previous one.

| # | Check | Command | Expected result |
|---|---|---|---|
| 1 | Modules import | `python -c "import config, rag_client, llm_client, ragas_evaluator; print('OK', ragas_evaluator.RAGAS_AVAILABLE)"` | `OK True` |
| 2 | Vector store is populated | `python embedding_pipeline.py --stats-only` | `total_documents` > 0 and chunks for `apollo_11`, `apollo_13` and `challenger` |
| 3 | Retrieval finds the right mission | `python inspect_retrieval.py "What happened to the oxygen tank on Apollo 13?"` | Top results have `mission=apollo_13` |
| 4 | Grounded answer in the app | Ask *"What happened to the oxygen tank on Apollo 13?"* | An answer based on the documents, with Faithfulness and Answer Relevancy scores in the sidebar |
| 5 | Out-of-scope question | Ask *"Who won the 2018 FIFA World Cup?"* | The model says the documents do not contain the answer; Faithfulness may show `n/a` |
| 6 | Batch evaluation | `python ragas_evaluator.py` | Two average scores between 0 and 1 |

## Configuration

| Setting | Where | Default |
|---|---|---|
| `OPENAI_API_KEY` | `.env` or environment | required |
| `OPENAI_BASE_URL` | `.env` or environment | empty = api.openai.com |
| `RAG_EVAL_MODEL` | `.env` or environment | `gpt-4o` |
| Chat models offered in the UI | `config.py` → `CHAT_MODELS` | `["gpt-4o"]` |
| Embedding model | `config.py` → `EMBEDDING_MODEL` | `text-embedding-3-small` |
| Vector store path and collection name | `config.py` → `CHROMA_DIR`, `COLLECTION_NAME` | `./chroma_db_openai`, `nasa_space_missions_text` |
| Characters per chunk sent to the model | `rag_client.py` → `MAX_CHUNK_CHARS` | `1000` |

If you change `EMBEDDING_MODEL`, rebuild the vector store with `--update-mode replace`; queries and stored chunks must use the same model.

## Findings

<!-- Fill in after running `python ragas_evaluator.py` and testing a few questions. -->

| Metric (test set) | Score |
|---|---|
| Faithfulness | x.xx |
| Answer relevancy | x.xx |

Observations:
- …

## Troubleshooting

| Problem | Cause and fix |
|---|---|
| `ImportError: cannot import name 'ChatVertexAI'` | The RAGAS patch is missing; run [Setup step 3](#3-patch-ragas-required). |
| Pipeline reports `Total text files to process: 0` | `data_text/` is missing or has the wrong layout; see [Setup step 5](#5-add-the-source-documents). |
| `No ChromaDB backends found!` in the app | The vector store hasn't been built yet, or the app was started from a different folder. Run the pipeline and start Streamlit from the project root. |
| `401` / invalid API key | Wrong key, or a proxy key without `OPENAI_BASE_URL` set. |
| `404` model not found or deprecated | The model isn't available for your key. Change `CHAT_MODELS` / `RAG_EVAL_MODEL`. |
| Evaluation shows `n/a` | RAGAS could not compute the metric, for example because the answer was "not in the documents" and contains no factual statements to check. |

## Limitations and next steps

- **Source quality:** much of the text is OCR output with timestamps and fragments, which lowers retrieval precision.
- **Chunking:** fixed-size token windows ignore sentence and speaker boundaries.
- **Context truncation:** only the first 1,000 characters of each 500-token chunk reach the model.
- **Mission filter:** supported by the retrieval layer but not yet exposed in the UI.
- **Tests:** no automated unit or integration tests yet.

Ideas to try next: hybrid search (BM25 + embeddings), re-ranking, sentence-aware chunking, and reference-based RAGAS metrics such as context precision and recall.

## Acknowledgements

Built on starter code and data from Udacity's *Generative AI* Nanodegree. The implementation, fixes, tooling and evaluation are my own work. The source documents are NASA mission records.
