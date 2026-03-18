from __future__ import annotations
import logging
import anthropic
from agent.config import CLAUDE_MODEL
from agent.models import CandidateBriefing, CandidateInput
from agent.prompts import BRIEFING_SYSTEM_PROMPT, build_briefing_user_prompt

logger = logging.getLogger(__name__)

# Claude is forced to call this tool — it can never produce plain text instead.
# The input_schema is the Pydantic model's own JSON schema, so the API validates
# structure at the transport layer before we even attempt Pydantic validation.
_BRIEFING_TOOL = {
    "name": "submit_briefing",
    "description": (
        "Submit the completed candidate briefing. "
        "Call this exactly once with all fields populated."
    ),
    "input_schema": CandidateBriefing.model_json_schema(),
}


def generate_briefing(
    candidate_input: CandidateInput,
    enrichment_context: dict,
    role_spec: str,
    role_title: str,
    anthropic_client: anthropic.Anthropic,
) -> CandidateBriefing:
    """
    Step 2 of the pipeline: synthesise enrichment data into a validated briefing.

    Uses tool_choice={"type": "tool"} to force Claude to call submit_briefing.
    The response arrives as a structured dict — no JSON parsing or fence stripping.
    On Pydantic validation failure, the error is sent back via tool_result so Claude
    can self-correct without re-running the enrichment step.

    Never raises — returns a flagged placeholder on total failure.
    """
    user_prompt = build_briefing_user_prompt(
        candidate=candidate_input,
        enrichment_context=enrichment_context,
        role_spec=role_spec,
        role_title=role_title,
    )
    messages: list[dict] = [{"role": "user", "content": user_prompt}]

    # Attempt 1
    data, response, api_error = _call_briefing(anthropic_client, messages)
    briefing, val_error = _validate(data, candidate_input.candidate_id)
    if briefing:
        return briefing

    logger.warning(
        f"First briefing attempt failed for {candidate_input.full_name}: "
        f"{val_error or api_error}. Retrying."
    )

    # Attempt 2 — continue the conversation with structured error feedback.
    # Append the assistant's tool_use response, then a tool_result with is_error=True
    # so Claude understands exactly which fields to fix.
    if response is not None:
        messages.append({"role": "assistant", "content": response.content})
        tool_use_id = next(
            (b.id for b in response.content if hasattr(b, "type") and b.type == "tool_use"),
            "unknown",
        )
        messages.append({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": (
                    f"Validation failed: {val_error or api_error}. "
                    "Correct only the invalid fields and resubmit using the submit_briefing tool."
                ),
                "is_error": True,
            }],
        })
    else:
        # API call itself failed — start a fresh retry with a plain instruction
        messages.append({
            "role": "user",
            "content": (
                f"The previous call failed: {api_error}. "
                "Please call the submit_briefing tool now with all required fields."
            ),
        })

    data2, _, error2 = _call_briefing(anthropic_client, messages)
    briefing2, val_error2 = _validate(data2, candidate_input.candidate_id)
    if briefing2:
        return briefing2

    logger.error(
        f"Both briefing attempts failed for {candidate_input.full_name}: "
        f"{val_error2 or error2}"
    )
    return _error_placeholder(candidate_input, role_title)


def _call_briefing(
    client: anthropic.Anthropic,
    messages: list,
) -> tuple[dict | None, object | None, str | None]:
    """
    Call Claude with tool_choice forced to submit_briefing.
    Returns (tool_input_dict, raw_response, error_string).
    Exactly one of (tool_input_dict, error_string) will be None.
    """
    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2048,
            temperature=0,
            system=BRIEFING_SYSTEM_PROMPT,
            tools=[_BRIEFING_TOOL],
            tool_choice={"type": "tool", "name": "submit_briefing"},
            messages=messages,
        )
        tool_block = next(
            (b for b in response.content if hasattr(b, "type") and b.type == "tool_use"),
            None,
        )
        if tool_block is None:
            return None, response, "Claude did not call the submit_briefing tool"
        return tool_block.input, response, None
    except Exception as e:
        logger.error(f"Briefing API call failed: {e}")
        return None, None, str(e)


def _validate(
    data: dict | None,
    candidate_id: str,
) -> tuple[CandidateBriefing | None, str | None]:
    """
    Validate a tool_input dict against CandidateBriefing.
    Returns (briefing, None) on success or (None, error_message) on failure.
    """
    if data is None:
        return None, "No data returned from tool call"
    try:
        data["candidate_id"] = candidate_id
        return CandidateBriefing(**data), None
    except Exception as e:
        return None, str(e)


def _error_placeholder(candidate: CandidateInput, role_title: str) -> CandidateBriefing:
    """Return a flagged placeholder so the pipeline never silently drops a candidate."""
    return CandidateBriefing(
        candidate_id=candidate.candidate_id,
        full_name=candidate.full_name,
        current_role={
            "title": candidate.current_title,
            "employer": candidate.current_employer,
            "tenure_years": 0.0,
        },
        career_narrative=(
            "[GENERATION_ERROR — manual review required for this candidate]. "
            "Please re-run or review manually."
        ),
        experience_tags=["[error]", "[error]", "[error]"],
        firm_aum_context="[GENERATION_ERROR — manual review required]",
        mobility_signal={"score": 1, "rationale": "[GENERATION_ERROR — manual review required]"},
        role_fit={
            "role": role_title,
            "score": 1,
            "justification": "[GENERATION_ERROR — manual review required]",
        },
        outreach_hook="[GENERATION_ERROR — manual review required]",
    )
