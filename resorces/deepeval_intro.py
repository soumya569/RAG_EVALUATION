import os

from dotenv import load_dotenv
from groq import Groq

from deepeval import evaluate
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric


load_dotenv()


# -----------------------------
# GROQ MODEL
# -----------------------------
class GroqModel(DeepEvalBaseLLM):

    def __init__(self, model="openai/gpt-oss-120b"):
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
# TEST CASE 1
# -----------------------------
case_1 = LLMTestCase(
    input="What is the capital of France?",
    actual_output="The capital of France is Paris.",
)


# -----------------------------
# TEST CASE 2
# -----------------------------
case_2 = LLMTestCase(
    input="What is the capital of France?",
    actual_output="France is a beautiful country famous for its food and wine.",
)


# -----------------------------
# GROQ JUDGE
# -----------------------------
groq_model = GroqModel(
    model="openai/gpt-oss-120b"
)


# -----------------------------
# ANSWER RELEVANCY
# -----------------------------
metric = AnswerRelevancyMetric(
    threshold=0.7,
    model=groq_model,
    include_reason=True
)


# -----------------------------
# RUN EVALUATION
# -----------------------------
evaluate(
    test_cases=[
        case_1,
        case_2
    ],
    metrics=[
        metric
    ]
)