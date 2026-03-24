from engine.rag.retriever import fetch_wikipedia_summary


def build_rag_prompt(prompt: str) -> str:
    """
    Combines retrieved knowledge with user prompt.
    """

    context = fetch_wikipedia_summary(prompt)

    if context:
        return f"""
        Use the following context to answer:

        Context:
        {context}

        Question:
        {prompt}
        """
    else:
        return prompt