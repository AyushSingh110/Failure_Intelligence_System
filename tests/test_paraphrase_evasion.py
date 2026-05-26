"""
Paraphrase Evasion Test — 50 prompts
Same harmful intent as regex-covered attack categories, but surface wording is completely
different — no keywords the regex layers are looking for. Tests whether ML layers
(pair_classifier, prompt_guard, LlamaGuard via Groq) catch what regex misses.
"""
