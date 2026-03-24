from engine.models.ollama_client import generate_llama_response

prompt = "Who discovered relativity?"

response = generate_llama_response(prompt)

print("Model Response:")
print(response)