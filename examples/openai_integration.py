"""
Drop-in FIE monitoring for OpenAI API calls.
Wrap any OpenAI completion with client.monitor() to get failure analysis.
"""
import os
import openai
from fie_sdk import FIEClient

openai.api_key = os.getenv("OPENAI_API_KEY")
fie = FIEClient(api_key=os.getenv("FIE_API_KEY", "fie-your-key"))


def monitored_completion(prompt: str, n_samples: int = 3) -> dict:
    """
    Calls OpenAI n times, passes all outputs to FIE for failure analysis.
    Returns the primary output + FIE analysis result.
    """
    responses = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        n=n_samples,
        temperature=0.7,
    )

    outputs = [choice.message.content for choice in responses.choices]
    primary = outputs[0]

    result = fie.monitor(
        prompt=prompt,
        model_outputs=outputs,
    )

    return {
        "output":    primary,
        "archetype": result.archetype,
        "risk":      result.high_failure_risk,
        "summary":   result.failure_summary,
    }


if __name__ == "__main__":
    r = monitored_completion("Who invented the telephone?")
    print(f"Output:    {r['output']}")
    print(f"Archetype: {r['archetype']}")
    print(f"Risk:      {r['risk']}")
