"""
sloth_llm.py
Ollama client module supporting:
1. Question-to-Answer + Python Code generation (gemma4:e2b)
2. Pure-text code debugging & tweaking in-memory (gemma4:e2b)
3. Multimodal screenshot-assisted debugging via file path or in-memory bytes (phi3:vision)
"""

import base64
import re
from pathlib import Path
import requests

from config import (
    OLLAMA_TIMEOUT_SECONDS,
    OLLAMA_URL,
    TEXT_MODEL,
    VISION_MODEL,
)


def _strip_markdown_fences(raw_output: str) -> str:
    """
    Strictly extract raw Python code from LLM output, stripping any markdown
    code fences or conversational preamble if the model included them.
    """
    cleaned = raw_output.strip()

    fenced_pattern = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
    matches = fenced_pattern.findall(cleaned)
    if matches:
        best_match = max(matches, key=len)
        return best_match.strip() + "\n"

    lines = cleaned.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]

    return "\n".join(lines).strip() + "\n"


def _extract_explanation_and_code(raw_output: str) -> tuple[str, str]:
    """
    Split an LLM response into:
    1. The readable answer/explanation
    2. The extracted executable Python code block (if present).
    """
    cleaned = raw_output.strip()
    fenced_pattern = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
    matches = fenced_pattern.findall(cleaned)

    extracted_code = ""
    if matches:
        extracted_code = max(matches, key=len).strip() + "\n"
        explanation = fenced_pattern.sub("", cleaned).strip()
    else:
        explanation = cleaned

    if not explanation:
        explanation = "Here is the Python solution for your question:"

    return explanation, extracted_code


def _build_user_content(prompt: str, code: str) -> str:
    """Format the instructions and source code into a structured prompt."""
    return (
        f"{prompt.strip()}\n\n"
        "STRICT OUTPUT REQUIREMENT:\n"
        "- Output ONLY valid, executable Python code.\n"
        "- Do NOT include markdown backticks (```python or ```).\n"
        "- Do NOT include prose, commentary, or explanations before or after the code.\n\n"
        "=== PYTHON CODE START ===\n"
        f"{code}\n"
        "=== PYTHON CODE END ==="
    )


def answer_question(question: str) -> dict[str, str]:
    """
    Take any pasted question, problem statement, or coding prompt and return:
    - 'answer': Direct explanation and step-by-step answer
    - 'code': Complete, runnable Python code
    - 'raw': Full unfiltered Markdown response
    """
    endpoint = f"{OLLAMA_URL.rstrip('/')}/api/chat"
    prompt_content = (
        "You are Sloth-Compile, an expert Python Architect and Problem Solver.\n"
        "The user has pasted a question below. Provide a clear, direct, and accurate answer.\n\n"
        "STRUCTURE YOUR RESPONSE EXACTLY AS FOLLOWS:\n"
        "1. **Direct Answer & Explanation**: Clearly answer the question and explain how the solution works.\n"
        "2. **Python Code**: Provide a complete, self-contained, runnable Python script inside a ```python ... ``` "
        "code block (with print statements so the output is visible when executed).\n\n"
        f"USER QUESTION:\n{question.strip()}"
    )

    payload = {
        "model": TEXT_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt_content,
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.2,
        },
    }

    response = requests.post(endpoint, json=payload, timeout=OLLAMA_TIMEOUT_SECONDS)
    response.raise_for_status()
    data = response.json()

    raw_reply = data.get("message", {}).get("content", "").strip()
    if not raw_reply:
        raise ValueError(f"Empty response received from Ollama text model '{TEXT_MODEL}'.")

    explanation, extracted_code = _extract_explanation_and_code(raw_reply)
    return {
        "answer": explanation,
        "code": extracted_code,
        "raw": raw_reply,
    }


def generate_code_from_question(question: str) -> str:
    """
    Generate complete, runnable Python code directly from a user's natural language question.
    """
    result = answer_question(question)
    if result["code"].strip():
        return result["code"]
    return _strip_markdown_fences(result["raw"])


def call_text_model(prompt: str, code: str) -> str:
    """
    Send Python code and tweaking/debugging instructions to the text model (gemma4:e2b)
    and return the cleaned, tweaked Python source code.
    """
    endpoint = f"{OLLAMA_URL.rstrip('/')}/api/chat"
    full_content = _build_user_content(prompt, code)

    payload = {
        "model": TEXT_MODEL,
        "messages": [
            {
                "role": "user",
                "content": full_content,
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
        },
    }

    response = requests.post(endpoint, json=payload, timeout=OLLAMA_TIMEOUT_SECONDS)
    response.raise_for_status()
    data = response.json()

    raw_reply = data.get("message", {}).get("content", "")
    if not raw_reply.strip():
        raise ValueError(f"Empty response received from Ollama text model '{TEXT_MODEL}'.")

    return _strip_markdown_fences(raw_reply)


def call_vision_model_bytes(prompt: str, code: str, image_bytes: bytes) -> str:
    """
    Send in-memory image bytes and Python code to the multimodal model (phi3:vision)
    without writing any temporary image or code files to disk.
    """
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    endpoint = f"{OLLAMA_URL.rstrip('/')}/api/chat"
    full_content = _build_user_content(prompt, code)

    payload = {
        "model": VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": full_content,
                "images": [encoded_image],
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
        },
    }

    response = requests.post(endpoint, json=payload, timeout=OLLAMA_TIMEOUT_SECONDS)
    response.raise_for_status()
    data = response.json()

    raw_reply = data.get("message", {}).get("content", "")
    if not raw_reply.strip():
        raise ValueError(f"Empty response received from Ollama vision model '{VISION_MODEL}'.")

    return _strip_markdown_fences(raw_reply)


def call_vision_model(prompt: str, code: str, image_path: str) -> str:
    """
    Read an error screenshot from disk, convert it to base64, and send both the code
    and visual context to the multimodal model (phi3:vision) via Ollama `/api/chat`.
    """
    img_file = Path(image_path)
    if not img_file.is_file():
        raise FileNotFoundError(f"Error screenshot not found at path: {image_path}")

    return call_vision_model_bytes(prompt, code, img_file.read_bytes())