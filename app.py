import streamlit as st
from pathlib import Path
from haystack.nodes import Seq2SeqGenerator, DensePassageRetriever
from haystack.pipelines import GenerativeQAPipeline
from haystack.document_stores import InMemoryDocumentStore
from askem.preprocessing import TextProcessor, convert_files_to_docs

UPLOAD_DIR = Path("tmp/upload")


def cleanup():
    for file in UPLOAD_DIR.iterdir():
        if file.is_file():
            file.unlink()


st.set_page_config(page_title="Question answering.", page_icon="📚")
st.title("Question answering")


### Cache static resources
@st.cache_resource
def load_text_processor():
    return TextProcessor()


@st.cache_resource
def load_generator():
    return Seq2SeqGenerator(model_name_or_path="vblagoje/bart_lfqa")


TEXT_PROCESSOR = load_text_processor()
GENERATOR = load_generator()

### Create dynamic session objects
if "document_store" not in st.session_state:
    st.session_state["document_store"] = InMemoryDocumentStore(embedding_dim=128)

if "retriever" not in st.session_state:
    st.session_state["retriever"] = DensePassageRetriever(
        document_store=st.session_state["document_store"],
        query_embedding_model="vblagoje/dpr-question_encoder-single-lfqa-wiki",
        passage_embedding_model="vblagoje/dpr-ctx_encoder-single-lfqa-wiki",
    )

if "qa_pipeline" not in st.session_state:
    st.session_state["qa_pipeline"] = GenerativeQAPipeline(
        GENERATOR, st.session_state["retriever"]
    )

################################################################################
st.subheader("Step 1. Upload one or more text document(s).")
"""
Step summary:
1. Save uploaded documents to a temporary directory.
"""

uploaded_file = st.file_uploader("Upload text document(s)", type=["txt"])
# Write the uploaded file to a temporary directory
if uploaded_file is not None:
    with open(UPLOAD_DIR / uploaded_file.name, "w") as f:
        f.write(uploaded_file.read().decode("UTF-8"))


################################################################################
st.subheader("Step 2. Start processing document(s).")
"""
Step summary:
1. Basic text preprocessing (e.g., remove extremely short sentences, excess blank spaces).
2. Split paragraphs into sentences.
3. Classify whether each sentence belongs to the same paragraph or not using a topic shift detection model.
4. Re-create paragraphs.
5. Write paragraphs to the document store.
6. Compute and store paragraphs embeddings.
7. Delete temporary uploaded documents.
"""


def process_docs():
    """Create embedding and delete temporary uploaded documents."""

    docs = convert_files_to_docs(
        dir_path=UPLOAD_DIR,
        clean_func=TEXT_PROCESSOR.to_paragraphs,
        split_paragraphs=True,
    )
    st.session_state.document_store.write_documents(docs)
    st.session_state.document_store.update_embeddings(st.session_state.retriever)


if st.button("Process document"):
    if not uploaded_file:
        st.warning("No file uploaded")
    with st.spinner("Processing documents..."):
        process_docs()
        cleanup()


################################################################################
st.subheader("Step 3. Enter your question then submit.")
"""
Step summary:
1. Convert question into embeddings.
2. Dot-product question embeddings with paragraph embeddings.
3. Get top-k related paragraphs.
4. Sent query with related paragraphs into question-answering model.
5. Return generated answer.
"""
question = st.text_input("Step 3: Ask your question.")

# When pressing submit button, execute count words
if st.button("Submit"):
    with st.spinner("Generating answer..."):
        y = st.session_state.qa_pipeline.run(question)

    if y["answers"]:
        st.success(y["answers"][0].answer)
