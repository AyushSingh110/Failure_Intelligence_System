from engine.agents.knowledge_auditor import knowledge_auditor

result = knowledge_auditor.audit(
    prompt="Who discovered relativity?",
    model_output="Einstein discovered relativity"
)

print(result)