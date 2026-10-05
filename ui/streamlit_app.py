"""Chat UI. It only talks to the HTTP API, so it exercises the same protected
routes any other client would use.

    streamlit run ui/streamlit_app.py
"""

import os

import httpx
import streamlit as st

API_URL = os.getenv("MDA_API_URL", "http://localhost:8000")
API_KEY = os.getenv("MDA_API_KEY", "dev-key-change-me")

st.set_page_config(page_title="Medical Document Assistant", layout="wide")


def api(method: str, path: str, **kwargs) -> httpx.Response:
    return httpx.request(
        method, f"{API_URL}{path}", headers={"X-API-Key": API_KEY}, timeout=300, **kwargs
    )


def error_detail(response: httpx.Response) -> str:
    try:
        return response.json().get("detail", response.text)
    except ValueError:
        return response.text


def show_sources(message: dict) -> None:
    if not message.get("citations"):
        return
    with st.expander(f"Sources ({len(message['citations'])})"):
        for c in message["citations"]:
            st.markdown(f"**[{c['ref']}] {c['filename']}, page {c['page']}**")
            st.caption(c["snippet"])


def show_meta(message: dict) -> None:
    if "trace_id" not in message:
        return
    if not message["grounded"]:
        warning = "Some statements could not be matched to a cited source. Check them."
        for sentence in message.get("unsupported", []):
            warning += f"\n\n> {sentence}"
        st.warning(warning)
    st.caption(
        f"{message['model']} · {message['latency_ms'] / 1000:.1f} s · trace `{message['trace_id']}`"
    )


try:
    health = httpx.get(f"{API_URL}/health", timeout=5).json()
except httpx.HTTPError:
    st.error(f"The API is not reachable at {API_URL}. Start it with `make api`.")
    st.stop()

with st.sidebar:
    st.header("Documents")
    uploaded = st.file_uploader("Upload a PDF or text file", type=["pdf", "txt", "md"])
    if uploaded and st.session_state.get("last_upload") != uploaded.file_id:
        with st.spinner("Extracting and indexing..."):
            response = api(
                "POST", "/documents", files={"file": (uploaded.name, uploaded.getvalue())}
            )
        st.session_state.last_upload = uploaded.file_id
        if response.is_success:
            doc = response.json()
            note = "indexed" if doc["created"] else "already indexed"
            st.success(f"{doc['filename']}: {doc['pages']} pages, {doc['chunks']} chunks ({note})")
        else:
            st.error(error_detail(response))

    documents = api("GET", "/documents").json()
    names = {d["id"]: d["filename"] for d in documents}
    scope = st.multiselect(
        "Search in",
        options=list(names),
        format_func=names.get,
        placeholder="All documents",
    )
    for d in documents:
        left, right = st.columns([5, 1])
        left.caption(f"{d['filename']} · {d['pages']} p · {d['chunks']} chunks")
        if right.button("✕", key=f"del-{d['id']}", help="Delete this document"):
            api("DELETE", f"/documents/{d['id']}")
            st.rerun()

    st.divider()
    st.caption(f"Model: {health['llm_provider']} / {health['llm_model']}")
    st.caption(f"Embeddings: {health['embedding_model']}")
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

st.title("Medical Document Assistant")
st.caption("Answers come only from the uploaded documents and cite the page they are based on.")

messages = st.session_state.setdefault("messages", [])
for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        show_sources(message)
        show_meta(message)

if not documents:
    st.info("Upload a document in the sidebar to get started.")

if question := st.chat_input("Ask a question about the documents", disabled=not documents):
    with st.chat_message("user"):
        st.markdown(question)

    # The API uses the last four exchanges at most, and rejects very long histories.
    history = [{"role": m["role"], "content": m["content"]} for m in messages[-8:]]
    with st.chat_message("assistant"):
        with st.spinner("Searching the documents..."):
            response = api(
                "POST",
                "/ask",
                json={"question": question, "document_ids": scope or None, "history": history},
            )
        if response.is_success:
            body = response.json()
            answer = {"role": "assistant", "content": body.pop("answer"), **body}
            st.markdown(answer["content"])
            show_sources(answer)
            show_meta(answer)
            messages.append({"role": "user", "content": question})
            messages.append(answer)
        else:
            st.error(error_detail(response))
