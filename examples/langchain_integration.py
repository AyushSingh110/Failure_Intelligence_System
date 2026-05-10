"""
FIE monitoring integrated with LangChain.
Attaches a custom callback that monitors every LLM call automatically.
"""
import os
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler
from fie_sdk import FIEClient

fie = FIEClient(api_key=os.getenv("FIE_API_KEY", "fie-your-key"))


class FIECallbackHandler(BaseCallbackHandler):
    """LangChain callback that sends every LLM response to FIE for monitoring."""

    def __init__(self):
        self._prompt = ""

    def on_llm_start(self, serialized, prompts, **kwargs):
        self._prompt = prompts[0] if prompts else ""

    def on_llm_end(self, response, **kwargs):
        outputs = [
            gen.text
            for gens in response.generations
            for gen in gens
            if gen.text
        ]
        if not outputs:
            return

        result = fie.monitor(
            prompt=self._prompt,
            model_outputs=outputs,
        )
        print(f"[FIE] archetype={result.archetype} risk={result.high_failure_risk}")


llm = ChatOpenAI(
    model="gpt-4o-mini",
    n=3,
    temperature=0.7,
    callbacks=[FIECallbackHandler()],
)

if __name__ == "__main__":
    response = llm.invoke("What year did the Berlin Wall fall?")
    print(f"Answer: {response.content}")
