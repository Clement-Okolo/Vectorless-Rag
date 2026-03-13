import asyncio
import base64
import copy
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import fitz
import pageindex.utils as utils
import streamlit as st
from groq import AsyncGroq
from pageindex import PageIndexClient

try:
    from credentials import GROQ_API_KEY, PAGEINDEX_API_KEY
except Exception:
    GROQ_API_KEY = ""
    PAGEINDEX_API_KEY = ""

MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
IMAGE_ROOT = BASE_DIR / "pdf_images"


def run_async(coro):
    return asyncio.run(coro)


@st.cache_resource
def get_clients() -> Tuple[PageIndexClient, AsyncGroq]:
    if not PAGEINDEX_API_KEY or not GROQ_API_KEY:
        raise RuntimeError("Missing API keys in credentials.py.")
    pi_client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    groq_client = AsyncGroq(api_key=GROQ_API_KEY)
    return pi_client, groq_client


async def call_vlm(
    groq_client: AsyncGroq,
    prompt: str,
    image_paths: List[str] = None,
    model: str = MODEL_NAME,
) -> str:
    messages = [{"role": "user", "content": prompt}]

    if image_paths:
        content = [{"type": "text", "text": prompt}]
        for image in image_paths:
            if not os.path.exists(image):
                continue
            with open(image, "rb") as image_file:
                image_data = image_file.read()
            b64 = base64.b64encode(image_data).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
        messages[0]["content"] = content

    response = await groq_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_completion_tokens=1024,
    )
    return response.choices[0].message.content.strip()


def parse_tree_search_result(raw_text: str) -> Dict:
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL)

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        thinking_match = re.search(
            r'"thinking"\s*:\s*"(.*?)"\s*,\s*"node_list"', text, flags=re.DOTALL
        )
        node_list_match = re.search(r'"node_list"\s*:\s*\[(.*?)\]', text, flags=re.DOTALL)

        thinking = thinking_match.group(1).replace("\\n", "\n") if thinking_match else raw_text.strip()
        node_list = []
        if node_list_match:
            node_list = re.findall(r'"([^"\\]+)"', node_list_match.group(1))

        return {"thinking": thinking, "node_list": node_list}


def extract_pdf_page_images(pdf_path: Path, output_dir: Path) -> Tuple[Dict[int, str], int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_document = fitz.open(str(pdf_path))

    page_images = {}
    total_pages = len(pdf_document)

    for page_number in range(total_pages):
        page = pdf_document.load_page(page_number)
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        image_path = output_dir / f"page_{page_number + 1}.jpg"
        pix.save(str(image_path))
        page_images[page_number + 1] = str(image_path)

    pdf_document.close()
    return page_images, total_pages


def get_page_images_for_nodes(
    node_list: List[str], node_map: Dict, page_images: Dict[int, str]
) -> List[str]:
    image_paths = []
    seen_pages = set()

    for node_id in node_list:
        if node_id not in node_map:
            continue
        node_info = node_map[node_id]
        for page_num in range(node_info["start_index"], node_info["end_index"] + 1):
            if page_num in page_images and page_num not in seen_pages:
                image_paths.append(page_images[page_num])
                seen_pages.add(page_num)

    return image_paths


def ensure_state() -> None:
    defaults = {
        "doc_cache": {},
        "query_count": 0,
        "messages": [],
        "selected_pdf_name": None,
        "last_uploaded_signature": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_pdf_files() -> List[Path]:
    if not DATA_DIR.exists():
        return []
    return sorted([p for p in DATA_DIR.glob("*.pdf") if p.is_file()])


def prepare_document(pi_client: PageIndexClient, pdf_path: Path) -> Dict:
    cache_key = str(pdf_path.resolve())
    if cache_key in st.session_state["doc_cache"]:
        return st.session_state["doc_cache"][cache_key]

    image_dir = IMAGE_ROOT / pdf_path.stem
    page_images, total_pages = extract_pdf_page_images(pdf_path, image_dir)

    doc_id = pi_client.submit_document(str(pdf_path))["doc_id"]
    for _ in range(30):
        if pi_client.is_retrieval_ready(doc_id):
            tree = pi_client.get_tree(doc_id, node_summary=True)["result"]
            node_map = utils.create_node_mapping(tree, include_page_ranges=True, max_page=total_pages)
            payload = {
                "doc_id": doc_id,
                "tree": tree,
                "page_images": page_images,
                "total_pages": total_pages,
                "node_map": node_map,
            }
            st.session_state["doc_cache"][cache_key] = payload
            return payload
        time.sleep(2)

    raise TimeoutError("Document indexing is still in progress. Please try again in a few moments.")


def build_search_prompt(query: str, tree_without_text: List[Dict]) -> str:
    return f"""
You are given a question and a tree structure of a document.
Each node contains a node id, node title, and a corresponding summary.
Your task is to find no more than 5 tree nodes that are likely to contain the answer to the question.

Question: {query}

Document tree structure:
{json.dumps(tree_without_text, indent=2)}

Please reply in the following JSON format:
{{
    "thinking": "<Your thinking process on which nodes are relevant to the question>",
    "node_list": ["node_id_1", "node_id_2", ..., "node_id_n"]
}}
Directly return the final JSON structure. Do not output anything else.
"""


def build_answer_prompt(query: str) -> str:
    return f"""
Answer the question based on the images of the document pages as context.

Question: {query}

Provide a clear answer based only on the context provided. Give your thinking process if possible. Make sure your answer is correct. If the answer cannot be found in the provided context, say "The answer is not found in the provided document."
"""


def main() -> None:
    st.set_page_config(page_title="Vectorless RAG", page_icon="📄", layout="wide")
    ensure_state()

    st.markdown("<h1 style='text-align:center;'>⚡ZeroVec AI</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center;color:gray;'>Retrieve relevant document nodes from a hierarchical tree summary, augment the user query with matched page-image context, and generate accurate answers using a VLM.</p>",
        unsafe_allow_html=True,
    )

    stack = [
        ("🦙", "Groq Llama 4 Scout", "VLM"),
        ("🗂️", "PageIndex", "Doc Indexing"),
        ("🔍", "Vectorless RAG", "Retrieval"),
    ]
    cols = st.columns(len(stack))
    for col, (icon, name, role) in zip(cols, stack):
        col.markdown(
            f"<div style='text-align:center;padding:4px 3px;border:1px solid #e0e0e0;border-radius:8px;line-height:1.15'>"
            f"<span style='font-size:1.1rem'>{icon}</span><br>"
            f"<span style='font-size:0.86rem;font-weight:600'>{name}</span><br>"
            f"<span style='font-size:0.68rem;color:gray'>{role}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    st.divider()

    if not PAGEINDEX_API_KEY or not GROQ_API_KEY:
        st.error("Missing API keys. Add PAGEINDEX_API_KEY and GROQ_API_KEY in credentials.py.")
        st.stop()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_ROOT.mkdir(parents=True, exist_ok=True)

    try:
        pi_client, groq_client = get_clients()
    except Exception as ex:
        st.error(f"Failed to initialize clients: {ex}")
        st.stop()

    with st.sidebar:
        st.header("Session")
        st.metric("Queries", st.session_state["query_count"])
        if st.button("Clear Chat", use_container_width=True):
            st.session_state["messages"] = []
            st.session_state["query_count"] = 0
            st.rerun()

        st.header("Document")
        uploaded_pdf = st.file_uploader(
            "Upload PDF", type=["pdf"], accept_multiple_files=False, key="pdf_uploader"
        )
        if uploaded_pdf is None:
            # Allow future uploads to be processed after clearing the widget.
            st.session_state["last_uploaded_signature"] = None
        else:
            upload_signature = f"{uploaded_pdf.name}:{uploaded_pdf.size}"
            if upload_signature != st.session_state.get("last_uploaded_signature"):
                file_name = Path(uploaded_pdf.name).name
                target_path = DATA_DIR / file_name

                # Avoid accidental overwrite by appending a numeric suffix.
                if target_path.exists():
                    stem, suffix = target_path.stem, target_path.suffix
                    n = 1
                    while True:
                        candidate = DATA_DIR / f"{stem}_{n}{suffix}"
                        if not candidate.exists():
                            target_path = candidate
                            break
                        n += 1

                target_path.write_bytes(uploaded_pdf.getbuffer())
                st.session_state["selected_pdf_name"] = target_path.name
                st.session_state["last_uploaded_signature"] = upload_signature
                st.success(f"Uploaded {target_path.name}")

        pdf_files = get_pdf_files()
        if not pdf_files:
            st.warning("No PDFs found in data/. Add a PDF to continue.")
            st.stop()

        default_index = 0
        selected_name = st.session_state.get("selected_pdf_name")
        if selected_name:
            for idx, pdf in enumerate(pdf_files):
                if pdf.name == selected_name:
                    default_index = idx
                    break

        selected_pdf = st.selectbox(
            "Choose a PDF", pdf_files, index=default_index, format_func=lambda p: p.name
        )
        st.session_state["selected_pdf_name"] = selected_pdf.name

    # Render existing chat history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant":
                meta = msg.get("meta", {})
                if meta.get("thinking"):
                    with st.expander("Reasoning over document tree"):
                        st.write(meta["thinking"])
                if meta.get("nodes"):
                    with st.expander("Retrieved Nodes"):
                        st.dataframe(meta["nodes"], use_container_width=True)
                if meta.get("images"):
                    with st.expander("Retrieved Page Images"):
                        st.image(meta["images"], caption=[Path(p).name for p in meta["images"]], width=280)

    # Sticky chat input at the bottom
    query = st.chat_input("Ask a question about the document...")

    if query:
        with st.chat_message("user"):
            st.write(query)
        st.session_state["messages"].append({"role": "user", "content": query})

        with st.spinner("Preparing document context..."):
            try:
                payload = prepare_document(pi_client, selected_pdf)
            except Exception as ex:
                st.error(str(ex))
                st.stop()

        tree = payload["tree"]
        node_map = payload["node_map"]
        page_images = payload["page_images"]

        tree_without_text = utils.remove_fields(copy.deepcopy(tree), fields=["text"])
        search_prompt = build_search_prompt(query, tree_without_text)

        with st.spinner("Reasoning over tree to retrieve relevant nodes..."):
            try:
                raw_search = run_async(call_vlm(groq_client, search_prompt))
                search_json = parse_tree_search_result(raw_search)
                retrieved_nodes = search_json.get("node_list", [])
            except Exception as ex:
                st.error(f"Retrieval step failed: {ex}")
                st.stop()

        retrieved_images = get_page_images_for_nodes(retrieved_nodes, node_map, page_images)
        answer_prompt = build_answer_prompt(query)

        with st.spinner("Generating final answer from PDF page images..."):
            try:
                answer = run_async(call_vlm(groq_client, answer_prompt, retrieved_images))
            except Exception as ex:
                st.error(f"Answer step failed: {ex}")
                st.stop()

        rows = []
        for node_id in retrieved_nodes:
            if node_id not in node_map:
                continue
            node_info = node_map[node_id]
            node = node_info["node"]
            start_page = node_info["start_index"]
            end_page = node_info["end_index"]
            page_range = f"{start_page}" if start_page == end_page else f"{start_page}-{end_page}"
            rows.append(
                {
                    "node_id": node.get("node_id", node_id),
                    "title": node.get("title", ""),
                    "pages": page_range,
                }
            )

        meta = {
            "thinking": search_json.get("thinking", ""),
            "nodes": rows,
            "images": retrieved_images,
        }

        with st.chat_message("assistant"):
            st.write(answer)
            if meta["thinking"]:
                with st.expander("Reasoning over document tree"):
                    st.write(meta["thinking"])
            if meta["nodes"]:
                with st.expander("Retrieved Nodes"):
                    st.dataframe(meta["nodes"], use_container_width=True)
            if meta["images"]:
                with st.expander("Retrieved Page Images"):
                    st.image(meta["images"], caption=[Path(p).name for p in meta["images"]], width=280)

        st.session_state["messages"].append({"role": "assistant", "content": answer, "meta": meta})
        st.session_state["query_count"] += 1


if __name__ == "__main__":
    main()
