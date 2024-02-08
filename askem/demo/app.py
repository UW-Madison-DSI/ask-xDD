import asyncio
import logging

import streamlit as st
from base import AppSettings, Topic, render, search

logging.basicConfig(level=logging.DEBUG)

ALL_TOPICS = [
    Topic(
        name="xdd-covid-19",
        label="COVID-19",
        preset_questions_path="preset_questions/preset_covid_q.txt",
    ),
    Topic(
        name="dolomites",
        label="Dolomites",
        preset_questions_path="preset_questions/preset_dolomites_q.txt",
    ),
    Topic(
        name="climate-change-modeling",
        label="Climate Change",
        preset_questions_path="preset_questions/preset_climate_change_q.txt",
    ),
    Topic(
        name="criticalmaas",
        label="CriticalMAAS",
        preset_questions_path="preset_questions/preset_critical_maas.txt",
    ),
    Topic(
        name="geoarchive",
        label="Geoarchive",
        preset_questions_path="preset_questions/preset_geoarchive_q.txt",
    ),
]


app_settings = AppSettings(
    title="ASK-xDD",
    topics=ALL_TOPICS,
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
        selected_topic = st.radio(
            "Choose a topic", app_settings.topics, format_func=lambda topic: topic.label
        )
        search_settings["topic"] = selected_topic.name

        st.subheader("Ask preset questions")
        # Topic specific preset questions
        preset_question = st.selectbox(
            "Ask preset questions", selected_topic.preset_questions
        )
        run_from_preset = st.button("Run")

        st.subheader("Advanced settings")
        st.markdown(
            "You can customize the QA system, all of these settings are available in the [API route](http://cosmos0001.chtc.wisc.edu:4502/docs) as well."
        )

        search_settings["model_name"] = st.radio("model", app_settings.model_names)
        search_settings["top_k"] = st.number_input("retriever top-k", value=5)
        search_settings["screening_top_k"] = st.number_input(
            "elastic search screening phase top-k", value=100
        )
        search_settings["verbose"] = st.checkbox("verbose", value=True)

    # Chat history
    for message in st.session_state.messages:
        render(message)

    # Chat input
    if question := st.chat_input("Ask a question.", key="question"):
        try:
            await search(question=question, **st.session_state.search_settings)
        except Exception as e:
            st.error(f"Error occur when searching relevant documents: {e}")

    if run_from_preset:
        try:
            await search(question=preset_question, **st.session_state.search_settings)
        except Exception as e:
            st.error(f"Error occur when searching relevant documents: {e}")


if __name__ == "__main__":
    asyncio.run(main(app_settings))
