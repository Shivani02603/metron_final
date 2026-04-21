"""
Master testing taxonomy — 28 categories aligned with OWASP LLM Top 10 & MITRE ATLAS.

Two teams:
  adversarial   → A01-A20  (attack personas — security red-team)
  user_centric  → U01-U08  (edge-case personas — UX / robustness)

Used by:
  - persona_builder.py  → routes to the right prompt and validates taxonomy IDs
  - security_gen.py     → maps playbook taxonomy IDs to internal attack categories
  - security.py         → replaces _OWASP_BY_CATEGORY dict; provides MITRE Atlas IDs
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TaxonomyEntry:
    id:              str
    name:            str
    category:        str          # security | functional | privacy | robustness | usability | compliance
    description:     str
    owasp_mapping:   Optional[str] = None   # e.g. "LLM01"
    mitre_atlas_id:  Optional[str] = None   # e.g. "AML.T0051"
    team:            str = "both"           # user_centric | adversarial | both
    # Internal attack-category name used by security_gen.py / evaluator
    internal_category: Optional[str] = None


MASTER_TAXONOMY: list[TaxonomyEntry] = [
    # ── Adversarial (A01-A20) ───────────────────────────────────────────────
    TaxonomyEntry(
        id="A01", name="Direct Prompt Injection",
        category="security",
        description="Attacker directly injects instructions to override system behavior",
        owasp_mapping="LLM01", mitre_atlas_id="AML.T0051",
        team="adversarial", internal_category="prompt_injection",
    ),
    TaxonomyEntry(
        id="A02", name="Indirect Prompt Injection",
        category="security",
        description="Malicious instructions hidden in retrieved documents or external data",
        owasp_mapping="LLM01", mitre_atlas_id="AML.T0051.001",
        team="adversarial", internal_category="prompt_injection",
    ),
    TaxonomyEntry(
        id="A03", name="System Prompt Leakage",
        category="security",
        description="Extracting the system prompt to understand and bypass guardrails",
        owasp_mapping="LLM07",
        team="adversarial", internal_category="data_exfiltration",
    ),
    TaxonomyEntry(
        id="A04", name="PII / Data Exfiltration",
        category="privacy",
        description="Extracting personal information, API keys, or training data",
        owasp_mapping="LLM02",
        team="adversarial", internal_category="pii_extraction",
    ),
    TaxonomyEntry(
        id="A05", name="Jailbreak / Role Hijacking",
        category="security",
        description="Bypassing safety filters via role-play, DAN, or developer mode exploits",
        owasp_mapping="LLM01",
        team="adversarial", internal_category="jailbreak",
    ),
    TaxonomyEntry(
        id="A06", name="Output Weaponization",
        category="security",
        description="Generating XSS, SQL injection, or malicious code via LLM output",
        owasp_mapping="LLM05",
        team="adversarial", internal_category="jailbreak",
    ),
    TaxonomyEntry(
        id="A07", name="Excessive Agency Exploit",
        category="security",
        description="Tricking agents into unauthorized tool calls, file access, or API actions",
        owasp_mapping="LLM06", mitre_atlas_id="AML.T0040",
        team="adversarial", internal_category="jailbreak",
    ),
    TaxonomyEntry(
        id="A08", name="Multi-Turn Manipulation",
        category="security",
        description="Building trust across conversation turns before launching attack",
        team="adversarial", internal_category="social_engineering",
    ),
    TaxonomyEntry(
        id="A09", name="Payload Splitting",
        category="security",
        description="Distributing attack payload across multiple messages to evade detection",
        team="adversarial", internal_category="prompt_injection",
    ),
    TaxonomyEntry(
        id="A10", name="Context Window Overflow",
        category="security",
        description="Flooding context to push safety instructions out of active memory",
        owasp_mapping="LLM10",
        team="adversarial", internal_category="prompt_injection",
    ),
    TaxonomyEntry(
        id="A11", name="Few-Shot Poisoning",
        category="security",
        description="Providing manipulative examples to shift model behavior",
        owasp_mapping="LLM04",
        team="adversarial", internal_category="prompt_injection",
    ),
    TaxonomyEntry(
        id="A12", name="Encoding Evasion",
        category="security",
        description="Bypassing content filters via Base64, ROT13, Unicode, or homoglyph tricks",
        team="adversarial", internal_category="encoding_attacks",
    ),
    TaxonomyEntry(
        id="A13", name="Authority Impersonation",
        category="security",
        description="Claiming admin/developer privileges to override restrictions",
        team="adversarial", internal_category="social_engineering",
    ),
    TaxonomyEntry(
        id="A14", name="Cross-Agent Chaining",
        category="security",
        description="Exploiting one agent to attack another in multi-agent systems",
        owasp_mapping="LLM06",
        team="adversarial", internal_category="jailbreak",
    ),
    TaxonomyEntry(
        id="A15", name="Supply Chain Manipulation",
        category="security",
        description="Poisoning tool/plugin responses or model dependencies",
        owasp_mapping="LLM03",
        team="adversarial", internal_category="data_exfiltration",
    ),
    TaxonomyEntry(
        id="A16", name="Hallucination Exploitation",
        category="robustness",
        description="Triggering confident misinformation or fabricated references",
        owasp_mapping="LLM09",
        team="adversarial", internal_category="jailbreak",
    ),
    TaxonomyEntry(
        id="A17", name="Embedding / Vector Manipulation",
        category="security",
        description="Crafting inputs to manipulate similarity search and retrieval",
        owasp_mapping="LLM08",
        team="adversarial", internal_category="data_exfiltration",
    ),
    TaxonomyEntry(
        id="A18", name="Resource Exhaustion (DoS)",
        category="security",
        description="Recursive loops, token flooding, or infinite generation attacks",
        owasp_mapping="LLM10",
        team="adversarial", internal_category="jailbreak",
    ),
    TaxonomyEntry(
        id="A19", name="Social Engineering",
        category="security",
        description="Emotional manipulation, urgency, or sympathy to bypass guardrails",
        team="adversarial", internal_category="social_engineering",
    ),
    TaxonomyEntry(
        id="A20", name="Compliance Violation Trigger",
        category="compliance",
        description="Tricking system into violating GDPR, HIPAA, or other regulations",
        team="adversarial", internal_category="social_engineering",
    ),
    # ── User-Centric (U01-U08) ──────────────────────────────────────────────
    TaxonomyEntry(
        id="U01", name="Ambiguous Query",
        category="usability",
        description="User sends vague or under-specified requests",
        team="user_centric",
    ),
    TaxonomyEntry(
        id="U02", name="Typo / Malformed Input",
        category="robustness",
        description="Input with spelling errors, grammar issues, or garbled text",
        team="user_centric",
    ),
    TaxonomyEntry(
        id="U03", name="Long / Complex Input",
        category="robustness",
        description="Extremely long messages, multi-paragraph instructions, or nested requests",
        team="user_centric",
    ),
    TaxonomyEntry(
        id="U04", name="Contradictory Instructions",
        category="robustness",
        description="User gives conflicting requirements in the same or successive messages",
        team="user_centric",
    ),
    TaxonomyEntry(
        id="U05", name="Accessibility Edge Case",
        category="usability",
        description="Users with screen readers, low vision, cognitive load, or language barriers",
        team="user_centric",
    ),
    TaxonomyEntry(
        id="U06", name="Domain Confusion",
        category="usability",
        description="User misunderstands what the system can do, asks off-topic questions",
        team="user_centric",
    ),
    TaxonomyEntry(
        id="U07", name="Copy-Paste Artifacts",
        category="robustness",
        description="Input containing formatting artifacts, hidden characters, or HTML tags",
        team="user_centric",
    ),
    TaxonomyEntry(
        id="U08", name="Multi-Language Input",
        category="robustness",
        description="User mixes languages or uses non-English text the system may not handle",
        team="user_centric",
    ),
]

# ── Lookup helpers ──────────────────────────────────────────────────────────
TAXONOMY_BY_ID:     dict[str, TaxonomyEntry] = {t.id: t for t in MASTER_TAXONOMY}
ADVERSARIAL_ENTRIES = [t for t in MASTER_TAXONOMY if t.team in ("adversarial", "both")]
USER_ENTRIES        = [t for t in MASTER_TAXONOMY if t.team in ("user_centric", "both")]
VALID_ADV_IDS:  set[str] = {t.id for t in ADVERSARIAL_ENTRIES}
VALID_USER_IDS: set[str] = {t.id for t in USER_ENTRIES}


def owasp_full(taxonomy_id: str) -> str:
    """Return the full OWASP tag string for a taxonomy ID, e.g. 'LLM01_prompt_injection'."""
    entry = TAXONOMY_BY_ID.get(taxonomy_id)
    if not entry or not entry.owasp_mapping:
        return "LLM05_jailbreak"
    # Map short OWASP code to the full tag format used by the evaluator
    _OWASP_FULL: dict[str, str] = {
        "LLM01": "LLM01_prompt_injection",
        "LLM02": "LLM02_sensitive_info_disclosure",
        "LLM03": "LLM03_supply_chain",
        "LLM04": "LLM04_data_model_poisoning",
        "LLM05": "LLM05_jailbreak",
        "LLM06": "LLM06_excessive_agency",
        "LLM07": "LLM07_system_prompt_leakage",
        "LLM08": "LLM08_vector_embedding_weakness",
        "LLM09": "LLM09_misinformation",
        "LLM10": "LLM10_unbounded_consumption",
    }
    return _OWASP_FULL.get(entry.owasp_mapping, "LLM05_jailbreak")


def internal_category(taxonomy_ids: list[str]) -> str:
    """Return the primary internal attack category for a list of taxonomy IDs."""
    for tid in taxonomy_ids:
        entry = TAXONOMY_BY_ID.get(tid)
        if entry and entry.internal_category:
            return entry.internal_category
    return "jailbreak"


def mitre_id(taxonomy_ids: list[str]) -> str | None:
    """Return the first MITRE Atlas ID found for a list of taxonomy IDs."""
    for tid in taxonomy_ids:
        entry = TAXONOMY_BY_ID.get(tid)
        if entry and entry.mitre_atlas_id:
            return entry.mitre_atlas_id
    return None
