import os
import json

from dotenv import load_dotenv
from groq import Groq

from deepeval import evaluate
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    ContextualRecallMetric,
    ContextualPrecisionMetric
)

from src.retriver import build_retriever


load_dotenv()


GOLDEN_PATH = "goldens/retriever_goldens.json"
JUDGE_MODEL = "openai/gpt-oss-120b"
THRESHOLD = 0.7


# -----------------------------
# GROQ MODEL
# -----------------------------

class GroqModel(DeepEvalBaseLLM):

    def __init__(self, model=JUDGE_MODEL):

        self.model_name = model

        self.client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        return response.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        return response.choices[0].message.content

    def get_model_name(self):
        return self.model_name


# -----------------------------
# LOAD GOLDENS
# -----------------------------

def load_goldens():

    with open(
        GOLDEN_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# -----------------------------
# RUN EVALUATION
# -----------------------------

def run():

    # Load golden questions
    goldens = load_goldens()

    # Build your existing retriever
    retriever = build_retriever()

    # Groq judge
    judge_model = GroqModel(
        model=JUDGE_MODEL
    )

    test_cases = []

    # Run every golden through retriever
    for g in goldens:

        retrieved = retriever.invoke(
            g["query"]
        )

        retrieval_context = [
            doc.page_content
            for doc in retrieved
        ]

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                expected_output=g["ideal_answer"],
                retrieval_context=retrieval_context,
                actual_output="(generator not evaluated)"
            )
        )

    # Metrics
    metrics = [

        ContextualRecallMetric(
            threshold=THRESHOLD,
            model=judge_model,
            include_reason=True
        ),

        ContextualPrecisionMetric(
            threshold=THRESHOLD,
            model=judge_model,
            include_reason=True
        )

    ]

    # Evaluate
    result = evaluate(
        test_cases=test_cases,
        metrics=metrics
    )

    return result


# -----------------------------
# MAIN
# -----------------------------

if __name__ == "__main__":

    run()