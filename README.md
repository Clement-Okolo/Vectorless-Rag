# Reasoning-Based, Vectorless RAG with PageIndex and VLM

A Retrieval-Augmented Generation (RAG) pipeline that retrieves without vector embeddings. Instead of chunking documents into text passages and indexing them in a vector store, this approach uses [PageIndex](https://pageindex.ai) to build a **hierarchical tree structure** of PDF documents and a **Vision-Language Model (VLM)** to reason over that tree and answer questions directly from PDF page images.

---

## How It Works

1. **Document Indexing** — A PDF is submitted to the PageIndex API, which parses it into a hierarchical tree of nodes (sections, subsections, etc.), each annotated with a title and summary.

2. **Reasoning-Based Retrieval** — A VLM (Llama 4 Scout via Groq) receives the entire tree (without full text) alongside the user's query. It reasons over node titles and summaries to identify which nodes are most likely to contain the answer.

3. **Visual QA** — The PDF pages corresponding to the retrieved nodes are rendered as JPEG images using PyMuPDF. The VLM then answers the query by visually inspecting those page images, grounding its response in the original document layout, figures, and tables.

This eliminates the need for text chunking, embedding models, and vector databases entirely.

---

## Architecture

```text
Upload Document
 └─► PageIndex API ──► Hierarchical Tree (titles + summaries)
                              │
                    VLM reasons over tree
                              │
                    Retrieved node IDs
                              │
               Render matching PDF page images
                              │
                    VLM generates answer from image(s) context
```

---

## Project Structure

```text
vectorless-rag/
├── vectorless-rag.ipynb   # Jupyter notebook with end-to-end pipeline
├── streamlit_app.py       # Streamlit UI for interactive demo (ZeroVec AI)
├── credentials.py         # API keys (do not commit to version control)
├── requirements.txt       # Python dependencies
├── data/                  # PDFs to query (pre-load or upload via the Streamlit app)
└── pdf_images/            # Rendered PDF page images (auto-generated)
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys

Create `credentials.py` with your own keys:

```python
PAGEINDEX_API_KEY = "your_pageindex_api_key"
GROQ_API_KEY = "your_groq_api_key"
```

- Get a PageIndex API key at [pageindex.ai](https://pageindex.ai)
- Get a Groq API key at [console.groq.com](https://console.groq.com)

### 3. Run the notebook

Open `vectorless-rag.ipynb` in VS Code or JupyterLab and run all cells in order.

### 4. Run the Streamlit app

From the `vectorless-rag/` directory:

```bash
streamlit run streamlit_app.py
```

The app (titled **ZeroVec AI**) will:

- read PDFs from `data/` or accept uploads via the sidebar file uploader
- save uploaded files to `data/` automatically
- render per-page JPEGs into `pdf_images/<pdf_name>/`
- index the document with PageIndex
- retrieve relevant nodes with a VLM
- answer your question using the retrieved page images as visual context

---

## Demo

The `data/` folder ships with an arXiv paper:

| File | Paper |
|------|-------|
| `1706.03762.pdf` | **Attention Is All You Need** — Vaswani et al., 2017 |


Example query against the paper:

> *"What is the last operation in the Scaled Dot-Product Attention figure?"*

The VLM inspects the tree, selects the relevant section node, and answers directly from the rendered PDF page image containing that figure.

---

## Use Cases

This pipeline is well-suited for any scenario where document layout, visuals, or structure matter as much as the text itself:

| Use Case | Description |
|----------|-------------|
| **Academic paper Q&A** | Query research papers for specific figures, equations, tables, or methodology details without losing their visual structure. |
| **Technical documentation** | Navigate large manuals, datasheets, or API docs by section — ask about diagrams, configuration tables, or step-by-step procedures. |
| **Legal & compliance documents** | Retrieve and reason over specific clauses, terms, or exhibits in contracts where formatting and layout carry meaning. |
| **Financial reports** | Answer questions about charts, financial tables, and summaries in annual reports or earnings filings. |
| **Medical & scientific literature** | Query clinical studies, lab reports, or imaging summaries where figures and annotated images are central to the answer. |
| **Educational content** | Students and educators can ask questions about textbook chapters, slides, or course materials in their original visual form. |
| **Enterprise knowledge bases** | Index internal PDFs (policy documents, SOPs, product specs) and enable employees to query them conversationally. |

---

## Dependencies

| Package | Purpose |
| ------- | ------- |
| `pageindex` | Document tree generation and retrieval |
| `groq` | Async VLM inference (Llama 4 Scout) |
| `PyMuPDF` | PDF rendering to JPEG images |
| `requests` | PDF download |

---

## Key Design Decisions

- **No vector store** — retrieval is done by VLM reasoning over structured summaries, not cosine similarity over embeddings.
- **Visual context** — the VLM reads rendered page images rather than extracted text, preserving layout, figures, and tables.
- **Hierarchical tree** — PageIndex produces a document tree that allows targeted retrieval at the section/subsection level, reducing noise.
