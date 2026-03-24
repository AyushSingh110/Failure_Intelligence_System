import requests

OLLAMA_URL = "http://localhost:11434/api/generate"


def generate_response(prompt: str) -> str:
    """
    Sends a prompt to the local Ollama model and returns the response.
    """

    payload = {
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)

    if response.status_code != 200:
        raise Exception(f"Ollama error: {response.text}")

    data = response.json()

    return data["response"]



from engine.rag.rag_pipeline import build_rag_prompt
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"


def generate_response(prompt: str):

    rag_prompt = build_rag_prompt(prompt)

    payload = {
        "model": "llama3",
        "prompt": rag_prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)

    return response.json()["response"]