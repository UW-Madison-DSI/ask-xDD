import numpy as np
from trulens_eval import Feedback, Select, Tru, TruCustomApp
from trulens_eval.feedback import Groundedness
from trulens_eval.feedback.provider.openai import OpenAI as fOpenAI
from trulens_eval.tru_custom_app import instrument

from askem.retriever.engine import react_search


class RAG:
    """Boilerplate wrapper for using Trulens."""

    def __init__(self, app_id: str, topic: str):
        self.app_id = app_id
        self.topic = topic
        self.used_docs = None
        self.answer = None

    def ask_question(self, question: str) -> None:
        """Ask question using react search."""

        results = react_search(question, topic=self.topic)
        self.used_docs = results["used_docs"]
        self.answer = results["answer"]

    @instrument
    def retrieve(self, query: str) -> list:
        return [doc.text_content for doc in self.used_docs]

    @instrument
    def generate_completion(self, query: str, context_str: list) -> str:
        return self.answer

    @instrument
    def query(self, query: str) -> str:
        self.ask_question(query)
        context_str = self.retrieve(query)
        completion = self.generate_completion(query, context_str)

        # Reset states
        self.used_docs = None
        self.answer = None

        return completion


class CustomGroundedness(Groundedness):
    def grounded_statements_aggregator(
        self, source_statements_multi_output: list[dict]
    ) -> float:
        """Aggregates multi-input, multi-output information from the groundedness_measure methods.

        Args:
            source_statements_multi_output (List[Dict]): A list of scores. Each list index is a context. The Dict is a per statement score.

        Returns:
            float: for each statement, gets the max groundedness, then max over that.
        """
        all_results = []
        statements_to_scores = {}

        print(source_statements_multi_output)
        # Ensure source_statements_multi_output is a list
        if not isinstance(source_statements_multi_output, list):
            source_statements_multi_output = [source_statements_multi_output]

        for multi_output in source_statements_multi_output:
            for k in multi_output:
                if k not in statements_to_scores:
                    statements_to_scores[k] = []
                statements_to_scores[k].append(multi_output[k])

        for k in statements_to_scores:
            all_results.append(np.max(statements_to_scores[k]))

        return np.max(all_results)  # Changed default from np.mean to np.max


class Eval:
    def __init__(self, model_engine: str = "gpt-4-1106-preview"):
        self.model_engine = model_engine

        self.tru = Tru("sqlite:///data/trulens_eval.db")
        self.fopenai = fOpenAI(model_engine=self.model_engine)

        # Custom max groundedness
        # self._grounded = CustomGroundedness(groundedness_provider=self.fopenai)

        # Default mean groundedness
        self._grounded = Groundedness(groundedness_provider=self.fopenai)

    def create_feedbacks(self) -> list[Feedback]:
        """Create feedbacks for Trulens."""

        # Define a groundedness feedback function
        f_groundedness = (
            Feedback(
                self._grounded.groundedness_measure_with_cot_reasons,
                name="Groundedness",
            )
            .on(Select.RecordCalls.retrieve.rets.collect())
            .on_output()
            .aggregate(self._grounded.grounded_statements_aggregator)
        )

        # Question/answer relevance between overall question and answer.
        f_qa_relevance = (
            Feedback(self.fopenai.relevance_with_cot_reasons, name="Answer Relevance")
            .on(Select.RecordCalls.retrieve.args.query)
            .on_output()
        )

        # Question/statement relevance between question and each context chunk.
        f_context_relevance = (
            Feedback(
                self.fopenai.qs_relevance_with_cot_reasons, name="Context Relevance"
            )
            .on(Select.RecordCalls.retrieve.args.query)
            .on(Select.RecordCalls.retrieve.rets.collect())
            .aggregate(np.mean)  # Aggregate over all context chunks (max relevance)
        )

        return [f_groundedness, f_qa_relevance, f_context_relevance]

    def run(self, rag: RAG, questions: list[str]) -> None:
        """Run evaluation on RAG model."""

        tru_rag = TruCustomApp(
            rag,
            app_id=rag.app_id,
            feedbacks=self.create_feedbacks(),
        )

        with tru_rag:
            [rag.query(question) for question in questions]
