import asyncio

import streamlit as st
from base import AppSettings, render, search

app_settings = AppSettings(
    title="XDD-LLM Question answering",
    topics=["covid", "dolomites"],
    preset_questions_paths=[
        "preset_questions/preset_covid_q.txt",
        "preset_questions/preset_dolomites_q.txt",
    ],
    model_names=["gpt-3.5-turbo-16k", "gpt-4", "gpt-4-1106-preview"],
)


async def main(app_settings: AppSettings) -> None:
    """Streamlit app."""
    st.set_page_config(page_title=app_settings.title, page_icon="📖")
    st.title(app_settings.title)

    # Initialize states
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "search_settings" not in st.session_state:
        st.session_state.search_settings = {}

    # Sidebar
    search_settings = st.session_state["search_settings"]
    with st.sidebar:
        st.subheader("Topic")
        search_settings["topic"] = st.radio("Choose a topic", ["covid", "dolomites"])

        st.subheader("Ask preset questions")
        # Topic specific preset questions
        preset_questions = app_settings.preset_questions()[search_settings["topic"]]
        preset_question = st.selectbox("Ask preset questions", preset_questions)
        run_from_preset = st.button("Run")

        st.subheader("Advanced settings")
        st.markdown(
            "You can customize the QA system, all of these settings are available in the [API route](http://cosmos0001.chtc.wisc.edu:4502/docs) as well."
        )

        search_settings["model_name"] = st.radio("model", app_settings.model_names)
        search_settings["top_k"] = st.number_input("retriever top-k", value=5)
        search_settings["screening_top_k"] = st.number_input(
            "elastic search screening phase top-k", value=1000
        )
        search_settings["verbose"] = st.checkbox("verbose", value=True)

    # Chat history
    for message in st.session_state.messages:
        render(message)

    # Chat input
    if question := st.chat_input("Ask a question about COVID-19", key="question"):
        await search(question=question, **st.session_state.search_settings)

    if run_from_preset:
        await search(question=preset_question, **st.session_state.search_settings)


if __name__ == "__main__":
    asyncio.run(main(app_settings))
