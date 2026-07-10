import json
import re

from step6_llm_answer import OLLAMA_MODEL, call_ollama

JUDGE_MODEL = OLLAMA_MODEL

JUDGE_PROMPT_TEMPLATE = """You are a strict evaluator of RAG answers. Score the ANSWER using \
ONLY the CONTEXT below - do not use outside knowledge. Respond with ONLY a JSON object, no \
other text, no markdown fences, no <think> tags:
{{"faithfulness": <1-5>, "relevance": <1-5>, "completeness": <1-5>, "reasoning": "<one sentence>"}}

faithfulness: is every claim in the answer supported by the context (no fabrication)?
relevance: does the answer address the question that was asked?
completeness: does the answer use the relevant information the context provides?

Context:
{context}

Question: {question}

Answer:
{answer}

JSON:"""

JUDGE_LIMITATION = (
    "Judge model (qwen3:1.7b) is the same model used to generate the answers. "
    "Scores may reflect self-consistency bias rather than independent correctness; "
    "treat as a regression-detection signal, not ground truth."
)


def parse_judge_response(raw_text):
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in judge response: {raw_text!r}")
    data = json.loads(match.group(0))
    for axis in ("faithfulness", "relevance", "completeness"):
        data[axis] = max(1, min(5, int(data[axis])))
    return data
