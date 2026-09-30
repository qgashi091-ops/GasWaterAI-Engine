"""Micro-agent architecture POC -- narrowly-scoped, small-context AI agents,
tested against the previous large multimodal plan-prompt approach.

CORE PRINCIPLE: one agent = one narrowly defined responsibility. An agent
never sees the full plan, the whole rule library, or another agent's
reasoning. Deterministic facts (PlanFacts topology, the vector graph)
remain exclusively authoritative and are never produced or overridden by an
agent.

CRITICAL ARCHITECTURE RULE: an agent produces an `OBSERVATION`, never a
`PLAN_FACT` directly. Promotion from observation to fact requires
corroboration from another independent source (another agent, deterministic
graph/text evidence) -- implemented, for the symbol-recognition role, by
feeding an agent's observation into the existing
`app/plan_analysis/component_evidence.py` fusion layer exactly like any
other evidence source, never as a shortcut around it.

Every module in this package calls the "direct external multimodal API
interface already created in Engine v1"
(`app/vision_fallback/anthropic_provider.py`'s constants/config) -- never
Base44's InvokeLLM.
"""
