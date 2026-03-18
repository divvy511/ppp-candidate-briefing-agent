from __future__ import annotations
import time
import logging
from collections.abc import Callable
from agent.config import INTER_CANDIDATE_DELAY_SECONDS
from agent.models import CandidateInput, CandidateBriefing, OutputSchema
from agent.enrichment import enrich_candidate
from agent.briefing import generate_briefing
from agent.utils import extract_role_title_from_spec

logger = logging.getLogger(__name__)

DEFAULT_ROLE_SPEC = """
Role: Head of Distribution / National BDM
Firm: Mid-tier Australian active asset manager, AUM $5–20B, institutional and wholesale focus

We are looking for:
- 10+ years in Australian funds management distribution, sales, or investor relations
- Proven track record managing and growing relationships with institutional investors, platforms, and IFAs
- Deep networks across superannuation funds, family offices, and financial planning dealer groups
- Experience managing and developing a national sales team (3–8 direct reports)
- Strong investment product knowledge across equities, fixed income, and/or alternatives
- Existing profile and brand in the Australian wholesale and institutional market

Ideal background: current or recent Head of Distribution, National Sales Manager, or Senior Institutional BDM at a comparable firm. Career progression from BDM to senior leadership preferred.
""".strip()


def process_candidate(
    candidate: CandidateInput,
    role_spec: str,
    role_title: str,
    anthropic_client,
    tavily_client,
    status_callback: Callable[[str], None] | None = None,
    preview_callback: Callable[[CandidateBriefing], None] | None = None,
) -> CandidateBriefing:
    """Run a single candidate through the full two-step pipeline."""
    if status_callback:
        status_callback(f"Enriching {candidate.full_name} — starting web research...")

    enrichment = enrich_candidate(
        candidate=candidate,
        anthropic_client=anthropic_client,
        tavily_client=tavily_client,
        status_callback=status_callback,
    )

    search_count = enrichment.get("total_searches", 0)
    if status_callback:
        status_callback(f"Completed {search_count} searches — generating briefing...")

    briefing = generate_briefing(
        candidate_input=candidate,
        enrichment_context=enrichment,
        role_spec=role_spec,
        role_title=role_title,
        anthropic_client=anthropic_client,
    )

    if preview_callback:
        preview_callback(briefing)

    return briefing


def run_pipeline(
    candidates: list[CandidateInput],
    role_spec: str,
    anthropic_client,
    tavily_client,
    status_callback: Callable[[str], None] | None = None,
    preview_callback: Callable[[CandidateBriefing], None] | None = None,
) -> OutputSchema:
    """
    Process candidates sequentially to stay within Tier 1 rate limits
    (50 RPM, 30K input tokens/min).

    Within each candidate, Tavily searches are parallelised when Claude batches
    multiple tool_use blocks — reducing wall time without adding Claude API calls.

    An inter-candidate pause (INTER_CANDIDATE_DELAY_SECONDS) keeps total request
    rate well inside the 50 RPM cap across a full 5-candidate run.
    """
    briefings: list[CandidateBriefing] = []
    total = len(candidates)
    role_title = extract_role_title_from_spec(role_spec)

    for i, candidate in enumerate(candidates):
        if status_callback:
            status_callback(f"[{i + 1}/{total}] {candidate.full_name} — {candidate.current_employer}")

        try:
            briefing = process_candidate(
                candidate=candidate,
                role_spec=role_spec,
                role_title=role_title,
                anthropic_client=anthropic_client,
                tavily_client=tavily_client,
                status_callback=status_callback,
                preview_callback=preview_callback,
            )
            briefings.append(briefing)

        except Exception as e:
            logger.error(f"Unhandled pipeline error for {candidate.full_name}: {e}", exc_info=True)
            if status_callback:
                status_callback(f"Failed to process {candidate.full_name} — adding placeholder")
            briefings.append(_placeholder_briefing(candidate, role_title))

        if i < total - 1:
            if status_callback:
                status_callback(f"Pausing {INTER_CANDIDATE_DELAY_SECONDS}s before next candidate...")
            time.sleep(INTER_CANDIDATE_DELAY_SECONDS)

    return OutputSchema(candidates=briefings)


def _placeholder_briefing(candidate: CandidateInput, role_title: str) -> CandidateBriefing:
    return CandidateBriefing(
        candidate_id=candidate.candidate_id,
        full_name=candidate.full_name,
        current_role={"title": candidate.current_title, "employer": candidate.current_employer, "tenure_years": 0.0},
        career_narrative="[PIPELINE_ERROR — manual review required]. Please re-run or review manually.",
        experience_tags=["[error]", "[error]", "[error]"],
        firm_aum_context="[PIPELINE_ERROR — manual review required]",
        mobility_signal={"score": 1, "rationale": "[PIPELINE_ERROR — manual review required]"},
        role_fit={"role": role_title, "score": 1, "justification": "[PIPELINE_ERROR — manual review required]"},
        outreach_hook="[PIPELINE_ERROR — manual review required]",
    )
