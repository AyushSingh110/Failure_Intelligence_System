class KnowledgeAuditor:
    """
    Orchestrates the verification pipeline.
    """

    def audit(self, prompt: str, model_output: str) -> dict:

        # Lazy imports to avoid circular dependency
        from engine.verifier.model_router import model_router
        from engine.verifier.answer_collector import answer_collector
        from engine.verifier.agreement_checker import agreement_checker

        # choose verification models
        models = model_router.get_models()

        # collect answers from verifiers
        answers = answer_collector.collect_answers(prompt)

        # analyze agreement
        result = agreement_checker.check_agreement(model_output, answers)

        return result


knowledge_auditor = KnowledgeAuditor()