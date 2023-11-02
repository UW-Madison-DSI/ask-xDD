import os

import requests
from langchain.agents import AgentType, initialize_agent
from langchain.agents.agent_iterator import AgentExecutorIterator
from langchain.chat_models import ChatOpenAI
from langchain.tools import StructuredTool
from tenacity import retry, stop_after_attempt, wait_random_exponential


@retry(wait=wait_random_exponential(min=3, max=15), stop=stop_after_attempt(6))
def get_llm(model_name: str):
    """Get LLM instance."""
    return ChatOpenAI(model_name=model_name, temperature=0)


class ReactManager:
    """Manage information in a single search chain."""

    def __init__(
        self,
        entry_query: str,
        retriever_endpoint: str,
        search_config: dict,
        model_name: str,
        verbose: bool = False,
    ):
        self.entry_query = entry_query
        self.retriever_endpoint = retriever_endpoint
        self.search_config = search_config
        self.model_name = model_name
        self.used_docs = []
        self.latest_used_docs = []

        # Retriever + ReAct agent
        self.agent_executor = initialize_agent(
            [StructuredTool.from_function(self.search_retriever)],
            llm=get_llm(self.model_name),
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=verbose,
        )

    def search_retriever(self, question: str) -> str:
        """Useful when you need to answer question about facts."""
        # Do NOT change the doc-string of this function, it will affect how ReAct works!

        headers = {
            "Content-Type": "application/json",
            "Api-Key": os.getenv("RETRIEVER_APIKEY"),
        }
        data = {"question": question}
        data.update(self.search_config)
        response = requests.post(self.retriever_endpoint, headers=headers, json=data)
        response.raise_for_status()

        # Collect used documents
        self.used_docs.extend(response.json())
        self.latest_used_docs = response.json()
        return "\n\n".join([r["text"] for r in response.json()])

    def get_iterator(self) -> AgentExecutorIterator:
        """ReAct iterator."""
        return self.agent_executor.iter({"input": self.entry_query})

    def run(self) -> str:
        """Run the chain until the end."""
        return self.agent_executor.invoke({"input", self.entry_query})["output"]
