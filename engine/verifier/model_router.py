class ModelRouter:

    def get_models(self):
        # Example verification models
        return [
            "gpt_verifier",
            "claude_verifier",
            "gemini_verifier"
        ]


model_router = ModelRouter()