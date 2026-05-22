"""
Attack pattern definitions — Layer 1 regex patterns.

Each _AttackPattern maps a compiled regex to a category, root_cause label,
and base confidence score used when that pattern fires in isolation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class _AttackPattern:
    category:        str
    root_cause:      str
    base_confidence: float
    pattern:         re.Pattern


_ATTACK_PATTERNS: list[_AttackPattern] = [

    # PROMPT INJECTION
    _AttackPattern(
        category="INJECTION",
        root_cause="PROMPT_INJECTION",
        base_confidence=0.88,
        pattern=re.compile(
            r"\b("
            r"ignore\s+(?:all\s+)?(?:previous|prior|above|earlier)\s+instructions?|"
            r"disregard\s+(?:all\s+)?(?:previous|prior|above|earlier)\s+instructions?|"
            r"forget\s+(?:everything|all\s+(?:previous|prior))\s+(?:you\s+were\s+told|instructions?)|"
            r"reveal\s+(?:the\s+)?(?:system\s+prompt|your\s+(?:instructions?|guidelines?|rules?))|"
            r"(?:print|show|output|display|repeat|tell\s+me)\s+(?:your\s+)?system\s+(?:prompt|instructions?)|"
            r"(?:print|show|output|display|reveal|tell\s+me)\s+(?:the\s+)?(?:hidden|secret|internal|developer)\s+(?:message|messages|prompt|instructions?|rules?)|"
            r"what\s+are\s+your\s+(?:hidden|secret|internal|original)\s+instructions?|"
            r"(?:internal|hidden|secret)\s+(?:instructions?|rules?)|"
            r"new\s+(?:primary\s+)?(?:instruction|directive|rule|task)\s*[:;]?\s+(?:you\s+must|ignore)|"
            r"(?:your\s+)?(?:new\s+)?(?:primary\s+)?directive\s+is\s+to\s+ignore|"
            r"SYSTEM\s*[:;]\s*override|"
            r"(?:from|starting)\s+now\s+(?:on\s+)?ignore\s+all|"
            r"(?:note|message|instructions?|reminder|alert)\s+(?:to|for)\s+(?:(?:the\s+)?(?:AI|LLM|language\s+model|assistant|model|GPT|Claude|bot))\s*[:\-]|"
            r"reveal\s+your\s+system\s+prompt"
            r")\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),

    # JAILBREAK
    _AttackPattern(
        category="JAILBREAK",
        root_cause="JAILBREAK_ATTEMPT",
        base_confidence=0.82,
        pattern=re.compile(
            r"\b("
            r"you\s+are\s+now\s+(?:DAN|an?\s+(?:AI|assistant|model)\s+with(?:out)?\s+(?:no\s+)?restrictions?)|"
            r"act\s+as\s+(?:an?\s+)?(?:unrestricted|uncensored|unfiltered|free)\s+(?:AI|model|assistant)|"
            r"act\s+as\s+(?:an?\s+)?(?:AI|model|assistant)\s+with(?:out)?\s+(?:no\s+|any\s+)?restrictions?|"
            r"(?:pretend|imagine|roleplay\s+as)\s+(?:you\s+have\s+)?(?:no\s+|without\s+)(?:content\s+)?(?:policy|restrictions?|guidelines?|limits?)|"
            r"you\s+(?:have\s+been\s+)?(?:freed|liberated|released)\s+from\s+(?:all\s+)?(?:restrictions?|guidelines?)|"
            r"(?:developer|jailbreak|debug|unrestricted|god)\s+mode\s+(?:activated?|enabled?|on|where\s+all\s+restrictions?\s+(?:are|have\s+been)\s+(?:lifted|removed|disabled))|"
            r"(?:your\s+)?(?:ethical\s+)?(?:restrictions?|guidelines?|limitations?|filters?)\s+"
            r"(?:have\s+been\s+)?(?:removed|disabled|lifted|suspended|deactivated)|"
            r"(?:no\s+longer\s+bound|freed?\s+from|released?\s+from|liberated\s+from)\s+(?:your\s+)?(?:ethical\s+)?(?:restrictions?|guidelines?|programming|training)|"
            r"(?:you\s+have\s+no\s+(?:restrictions?|guidelines?|content\s+policy))|"
            r"(?:behave|respond|answer|reply)\s+as\s+(?:if\s+)?(?:you\s+have\s+no\s+restrictions?|an?\s+unrestricted\s+AI|you\s+are\s+not\s+(?:bound|limited|restricted))"
            r")\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),

    # INSTRUCTION OVERRIDE
    _AttackPattern(
        category="OVERRIDE",
        root_cause="INSTRUCTION_OVERRIDE",
        base_confidence=0.85,
        pattern=re.compile(
            r"\b("
            r"(?:as\s+(?:your\s+)?(?:developer|administrator|admin|supervisor|operator|owner|creator|god|root))\s*[,:]?\s+(?:I\s+(?:command|order|instruct|require)|you\s+(?:must|will|shall))|"
            r"I\s+(?:am|'m)\s+(?:your\s+)?(?:developer|administrator|admin|supervisor|operator|owner|creator)|"
            r"(?:system|developer|admin|root|god)\s+(?:command|override|instruction)\s*[:;]|"
            r"override\s+(?:safety|content|ethical)\s+(?:policy|policies|restrictions?|guidelines?|filters?)|"
            r"(?:disable|turn\s+off|bypass|circumvent)\s+(?:your\s+)?(?:safety|content|ethical)\s+(?:filters?|restrictions?|guidelines?)|"
            r"you\s+(?:must|will|shall)\s+(?:now\s+)?ignore\s+(?:all\s+)?(?:safety|content|ethical)\s+(?:guidelines?|restrictions?|policies)"
            r")\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),

    # TOKEN SMUGGLING
    _AttackPattern(
        category="SMUGGLING",
        root_cause="TOKEN_SMUGGLING",
        base_confidence=0.90,
        pattern=re.compile(
            r"("
            r"<\|(?:system|endoftext|im_start|im_end|pad|unk|bos|eos|sep|cls|mask)\|>|"
            r"\[INST\]|\[/INST\]|\[SYS\]|\[/SYS\]|"
            r"###\s*(?:Human|Assistant|System|Instruction|Input|Response|Question|Answer)\s*:|"
            r"<<SYS>>|<</SYS>>|"
            r"\x00||"
            r"(?:user|human|assistant|system)\s*:\s*<\|(?:im_start|im_end)\|>|"
            r"\[(?:SYSTEM_PROMPT|SYSTEM|SYS_MSG)\]|"
            r"<system>.*?</system>|"
            r"<(?:human|assistant|user)>.*?</(?:human|assistant|user)>"
            r")",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
]
