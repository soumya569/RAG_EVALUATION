import os
import re
import glob
import json
import random

from dotenv import load_dotenv
from deepeval.synthesizer import Synthesizer
from deepeval.models import DeepEvalBaseLLM
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from groq import Groq

load_dotenv()


# -----------------------------
# GROQ MODEL FOR DEEPEVAL
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
# LOAD VTT + CHUNK
# -----------------------------
def load_chunks():

    texts = []

    for path in glob.glob("data/*.vtt"):

        with open(path, encoding="utf-8") as f:

            lines = [
                ln.strip()
                for ln in f
                if ln.strip()
                and ln.strip() != "WEBVTT"
                and "-->" not in ln
            ]

        texts.append(" ".join(lines))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    return splitter.split_text("\n\n".join(texts))


# -----------------------------
# GENERATE GOLDENS
# -----------------------------
chunks = load_chunks()

sample = random.sample(
    chunks,
    min(15, len(chunks))
)

contexts = [
    [c]
    for c in sample
]


# -----------------------------
# GROQ LLM
# -----------------------------
groq_llm = GroqModel(
    model="openai/gpt-oss-120b"
)


# -----------------------------
# DEEPEVAL SYNTHESIZER
# -----------------------------
synthesizer = Synthesizer(
    model=groq_llm
)


goldens = synthesizer.generate_goldens_from_contexts(
    contexts=contexts,
    include_expected_output=True,
    max_goldens_per_context=1
)


# -----------------------------
# CONVERT TO YOUR SCHEMA
# -----------------------------
rows = []

for i, g in enumerate(goldens, 1):

    rows.append({
        "id": f"g{i:03d}",
        "query": g.input,
        "ideal_answer": g.expected_output,
        "source": "TODO-verify"
    })


# -----------------------------
# SAVE JSON
# -----------------------------
os.makedirs("goldens", exist_ok=True)

with open(
    "goldens/retriever_deepeval_goldens.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        rows,
        f,
        indent=2,
        ensure_ascii=False
    )


print(
    f"wrote {len(rows)} DRAFT goldens "
    "-> goldens/retriever_deepeval_goldens.json"
)

print(
    "!! REVIEW EVERY ONE before using: "
    "check grounding, trim padding, fix leading questions."
)