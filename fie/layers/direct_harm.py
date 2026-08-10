"""
Layer 7 — direct harmful intent.

Two-gate action+target matcher for harmful requests made without any
adversarial framing. The gate keeps false positives low. Weight 1.1.

Extracted from fie/adversarial.py. Detection logic is unchanged — see
tests/test_detection_golden.py, which pins the exact confidence values.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


# Layer 7: Direct harmful intent detector
# Catches single-turn direct requests for harmful information that carry no
# jailbreak structure — exactly the prompts MSJ/injection layers skip.
# Pattern: action verb + harmful target (two-part gate keeps FPR near zero).

_DIRECT_HARM_ACTION_RE = re.compile(
    r"\b(?:how\s+(?:do\s+I|can\s+I|would\s+(?:I|someone)|to)|"
    r"(?:give|show|tell|teach|explain|describe|provide|write|create|make|build|"
    r"generate|produce|prepare|steps?\s+to|instructions?\s+(?:for|to)|"
    r"guide\s+(?:me\s+)?(?:on|to)|walk\s+me\s+through|help\s+me))\b",
    re.IGNORECASE,
)

_DIRECT_HARM_TARGET_RE = re.compile(
    r"\b(?:"
    # Weapons / explosives
    r"(?:make|build|create|synthesize|construct|assemble|fabricate)\s+(?:an?\s+)?(?:pipe\s+)?(?:bomb|explosives?|device|weapon|grenade|ied|poison|toxin|bioweapon|nerve\s+agent|napalm|thermite|chlorine\s+gas|sarin|VX\b)|"
    r"(?:how\s+to\s+make|make)\s+(?:chlorine\s+gas|mustard\s+gas|nerve\s+agent|ricin|anthrax\s+spores?|explosives?)|"
    r"(?:detonate|trigger)\s+(?:an?\s+)?(?:bomb|explosive|device)|"
    r"(?:step[\s\-]+by[\s\-]+step\s+)?instructions?\s+for\s+(?:building|making|creating|constructing)\s+(?:a\s+)?(?:pipe\s+)?(?:bomb|explosives?|weapon)|"
    r"(?:step[\s\-]+by[\s\-]+step\s+)?synthesis\s+of\s+(?:meth(?:amphetamine)?|fentanyl|heroin|cocaine|crack|mdma|lsd|ricin)|"
    r"(?:synthesizing|synthesis\s+of)\s+(?:VX\b|sarin|novichok|tabun|soman|nerve\s+agent)|"
    # Bioweapon/chemical weapon synthesis instructions (noun form)
    r"(?:synthesis|manufacturing|production|creation)\s+(?:instructions?|guide|process|method|steps?|recipe|route|procedure|protocol)\s+for\s+(?:a\s+)?(?:biological|chemical|radiological|nuclear)\s+(?:weapon|agent|toxin|explosive|bomb)|"
    r"(?:synthesis|manufacturing|production)\s+(?:route|procedure|protocol|process|instructions?|steps?|guide|recipe)\s+for\s+(?:VX|sarin|novichok|tabun|soman|mustard\s+gas|nerve\s+agent)|"
    r"(?:detailed\s+)?(?:instructions?|guide|steps?|recipe|process|method|route|procedure|protocol)\s+(?:to\s+(?:create|make|produce|synthesize|build)|for\s+(?:creating|making|producing|synthesizing|building))\s+(?:a\s+)?(?:biological|chemical|nerve|bio)\s+(?:weapon|agent|toxin)|"
    # CSAM
    r"(?:generate|create|write|produce|make|describe)\s+(?:sexual|explicit|pornographic|nude|erotic)\s+(?:content|images?|stories?|text|material)\s+(?:involving|about|featuring|with|of)\s+(?:a\s+)?(?:\d+[\s\-]year[\s\-]old|minor|child|underage|kid|teen(?:ager)?|juvenile)|"
    r"(?:sexual|explicit|pornographic|nude|erotic)\s+(?:content|images?|stories?|text|material)\s+(?:of|with|involving|featuring)\s+(?:a\s+)?(?:child|minor|underage\s+(?:person|girl|boy)|kid)|"
    # Drugs
    r"(?:synthesize|make|cook|produce|manufacture)\s+(?:meth(?:amphetamine)?|fentanyl|heroin|cocaine|crack|mdma|lsd|ricin)|"
    r"(?:synthesis|production|manufacturing)\s+(?:process|instructions?|steps?|guide|recipe|route|procedure|protocol)\s+for\s+(?:meth(?:amphetamine)?|fentanyl|heroin|cocaine|crack|mdma|lsd|ricin)|"
    r"clandestine\s+(?:meth(?:amphetamine)?|drug|fentanyl|heroin|cocaine)\s+(?:lab|laboratory|production|synthesis)|"
    r"precursor\s+chemicals?\s+(?:and|for)\s+(?:reaction|synthesis|production|manufacturing)\s+(?:steps?|process)|"
    # Hacking / cybercrime
    r"(?:hack|break\s+into|gain\s+(?:unauthorized\s+)?access\s+to|compromise|exploit|infiltrate)\s+(?:into\s+)?(?:(?:a|an|the)\s+)?(?:\w+\s+)?(?:computer|systems?|networks?|server|account|database|website)|"
    r"(?:install|deploy|create|write|build|construct)\s+(?:a\s+)?(?:malware|ransomware|keylogger|trojan|rootkit|backdoor|botnet|rat\b|virus|worm)|"
    r"(?:malware|ransomware|virus|trojan)\s+(?:is\s+)?(?:constructed|built|written|created|coded|deployed)\b|"
    r"(?:exfiltrate|steal|extract|dump)\s+(?:the\s+)?(?:full\s+)?(?:data|credentials?|passwords?|files?|database)\s*(?:from|without|via|using)?|"
    r"(?:extract|dump|exfiltrate)\s+(?:the\s+)?(?:full\s+)?database\b|"
    # Violence
    r"(?:kill|murder|poison|harm|hurt|attack|assault|eliminate)\s+(?:(?:\w+\s+){0,3})?(?:person|someone|people|human)|"
    r"(?:make|build|create)\s+(?:a\s+)?(?:weapon|gun|firearm|knife\s+weapon|shiv)|"
    # Fraud / financial crime
    r"(?:launder|hide|conceal)\s+(?:money|funds|cash)|"
    r"(?:create|make|build)\s+(?:a\s+)?(?:phishing|scam|fraud)\s+(?:page|site|email|scheme)|"
    r"(?:counterfeit|forge|fake)\s+(?:currency|money|documents?|ids?|passports?)|"
    # Physical security bypass
    r"(?:bypass|defeat|disable|circumvent|tamper\s+with)\s+(?:a\s+)?(?:building|facility|physical|perimeter|office|home)\s+"
    r"(?:security|alarm|access\s+control|lock|camera|surveillance)\s*(?:system)?|"
    r"gain\s+unauthorized\s+(?:physical\s+)?(?:entry|access)\s+(?:to|into)|"
    # Self-harm (handled carefully — high threshold)
    r"(?:most\s+(?:effective|lethal|painless)\s+(?:way|method)\s+to\s+(?:commit\s+suicide|kill\s+(?:myself|yourself)))"
    r")\b",
    re.IGNORECASE | re.DOTALL,
)

# Illegal access pattern — catches "illegally access X without detection" style
_ILLEGAL_ACCESS_RE = re.compile(
    r"\b(?:illegal(?:ly)?|unauthorized(?:ly)?|without\s+(?:permission|authorization|detection|being\s+caught))\s+"
    r".{0,40}"
    r"(?:access|hack|infiltrate|break\s+into|compromise|exfiltrate|install|deploy)\b",
    re.IGNORECASE | re.DOTALL,
)


def _run_direct_harm_detection(prompt: str) -> tuple[str | None, float, dict]:
    if len(prompt) < 10:
        return None, 0.0, {}

    # Gate 1: specific harmful target — high-confidence standalone hit
    target_match = _DIRECT_HARM_TARGET_RE.search(prompt)
    if target_match:
        action_match = _DIRECT_HARM_ACTION_RE.search(prompt)
        confidence   = 0.85 if action_match else 0.72
        return "DIRECT_HARMFUL_REQUEST", confidence, {
            "matched_target": target_match.group(0)[:120],
            "action_present": bool(action_match),
        }

    # Gate 2: illegal-access pattern (covers M2-style "illegally access ... without detection")
    illegal_match = _ILLEGAL_ACCESS_RE.search(prompt)
    if illegal_match:
        return "DIRECT_HARMFUL_REQUEST", 0.76, {
            "matched_target": illegal_match.group(0)[:120],
            "action_present": True,
        }

    return None, 0.0, {}


