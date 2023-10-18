import os

import requests
from langchain.agents import AgentType, initialize_agent
from langchain.agents.agent_iterator import AgentExecutorIterator
from langchain.llms import OpenAI
from langchain.tools import tool


@tool
def search_retriever(query: str) -> str:
    """Useful for when you need to answer questions about facts."""

    RETRIEVER_APIKEY = os.getenv("RETRIEVER_APIKEY")
    RETRIEVER_ENDPOINT = os.getenv("RETRIEVER_URL")
    REACT_RETRIEVER_TOP_K = os.getenv("REACT_RETRIEVER_TOP_K", 5)

    headers = {"Content-Type": "application/json", "Api-Key": RETRIEVER_APIKEY}
    data = {
        "question": query,
        "top_k": REACT_RETRIEVER_TOP_K,
        "doc_type": "paragraph",
    }

    response = requests.post(RETRIEVER_ENDPOINT, headers=headers, json=data)
    response.raise_for_status()
    return "\n\n".join([r["text"] for r in response.json()])


agent_executor = initialize_agent(
    [search_retriever],
    llm=OpenAI(model_name="gpt-4", temperature=0),
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)


def react_workflow(query: str) -> str:
    """React end-to-end workflow."""
    # Skip all the steps and return the final output
    return agent_executor.invoke({"input", query})["output"]


def get_iterator(query: str) -> AgentExecutorIterator:
    """Create ReAct chain iterator."""

    return agent_executor.iter(inputs={"input": query})
