from engine.models.ollama_client import generate_response

if __name__ == "__main__":

    prompt = input("Enter prompt: ")

    result = generate_response(prompt)

    print("\n--- WITH RAG ---\n")
    print(result)