from engine.models.ollama_client import generate_llama_response
from engine.agents.knowledge_auditor import knowledge_auditor


def run_realtime_pipeline(prompt):

    print("Generating response from Llama3...")

    model_output = generate_llama_response(prompt)

    print("Model Output:")
    print(model_output)

    print("Running Knowledge Auditor...")

    verification = knowledge_auditor.audit(
        prompt=prompt,
        model_output=model_output
    )

    print("Verification Result:")
    print(verification)

    return verification