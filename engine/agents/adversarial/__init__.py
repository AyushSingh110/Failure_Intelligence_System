"""
engine.agents.adversarial — adversarial detection sub-package.

Replaces the 1612-line adversarial_specialist.py monolith with focused modules:

  patterns.py      Layer 1 regex patterns (_AttackPattern definitions)
  normalization.py Layer 0 obfuscation normalization (homoglyphs, leet, spaced chars)
  injection.py     Layers 1-4 (regex, prompt guard, FAISS, indirect injection)
  many_shot.py     Layer 3b (many-shot / few-shot jailbreak)
  gcg.py           Layer 5 (GCG adversarial suffix)
  perplexity.py    Layer 6 (perplexity proxy: compression, entropy, KL divergence)
  semantic.py      Layers 7-8 (exfiltration, semantic consistency)
  llm_intent.py    Layer 9 (LLM semantic intent check — PAIR-style attacks)
  specialist.py    AdversarialSpecialist jury agent (orchestrates all 10 layers)
"""
from engine.agents.adversarial.specialist import AdversarialSpecialist, adversarial_specialist

__all__ = ["AdversarialSpecialist", "adversarial_specialist"]
