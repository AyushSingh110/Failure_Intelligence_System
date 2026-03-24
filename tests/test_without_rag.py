import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

def generate_without_rag(prompt: str):

    payload = {
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)

    return response.json()["response"]


if __name__ == "__main__":

    prompt = input("Enter prompt: ")

    result = generate_without_rag(prompt)

    print("\n--- WITHOUT RAG ---\n")
    print(result)