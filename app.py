import streamlit as st
from pathlib import Path
from haystack.nodes import Seq2SeqGenerator, DensePassageRetriever
from haystack.pipelines import GenerativeQAPipeline
from haystack.document_stores import InMemoryDocumentStore
from askem.preprocessing import TextProcessor, convert_files_to_docs

UPLOAD_DIR = Path("tmp/upload")
st.set_page_config(page_title="Long form question answering.", page_icon="📚")
st.title("Long form question answering")

if "text_processor" not in st.session_state:
    st.session_state["text_processor"] = TextProcessor()

# Initialization
if "document_store" not in st.session_state:
    st.session_state["document_store"] = InMemoryDocumentStore(embedding_dim=128)

if "retriever" not in st.session_state:
    st.session_state["retriever"] = DensePassageRetriever(
        document_store=st.session_state["document_store"],
        query_embedding_model="vblagoje/dpr-question_encoder-single-lfqa-wiki",
        passage_embedding_model="vblagoje/dpr-ctx_encoder-single-lfqa-wiki",
    )

if "generator" not in st.session_state:
    st.session_state["generator"] = Seq2SeqGenerator(
        model_name_or_path="vblagoje/bart_lfqa"
    )

if "qa_pipeline" not in st.session_state:
    st.session_state["qa_pipeline"] = GenerativeQAPipeline(
        st.session_state["generator"], st.session_state["retriever"]
    )

################################################################################
st.subheader("1. Upload one or more text document(s).")
"""
Currently the data is split into smaller paragraphs with a topic shift classification model, small experiment 

"""

uploaded_file = st.file_uploader("Upload text document(s)", type=["txt"])
# Write the uploaded file to a temporary directory
if uploaded_file is not None:
    with open(UPLOAD_DIR / uploaded_file.name, "w") as f:
        f.write(uploaded_file.read().decode("UTF-8"))


################################################################################
st.subheader("2. Press process document button.")


def process_docs():
    """Create embedding and delete temporary uploaded documents."""

    docs = convert_files_to_docs(
        dir_path=UPLOAD_DIR,
        clean_func=st.session_state.text_processor.to_paragraphs,
        split_paragraphs=True,
    )
    st.session_state.document_store.write_documents(docs)
    st.session_state.document_store.update_embeddings(st.session_state.retriever)

    for file in UPLOAD_DIR.iterdir():
        if file.is_file():
            file.unlink()


if st.button("Process document"):
    if not uploaded_file:
        st.warning("No file uploaded")
    with st.spinner("Processing documents..."):
        process_docs()


st.subheader("3. Enter your question then submit.")
################################################################################
question = st.text_input("What is your question?")

# When pressing submit button, execute count words
if st.button("Submit"):
    with st.spinner("Generating answer..."):
        y = st.session_state.qa_pipeline.run(question)
    y["answers"][0].answer
