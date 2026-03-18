from __future__ import annotations
import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
import anthropic
from agent.config import CLAUDE_MODEL, MAX_TOOL_ITERATIONS
from agent.tools import ALL_TOOLS, execute_tool
from agent.models import CandidateInput
from agent.prompts import ENRICHMENT_SYSTEM_PROMPT, build_enrichment_user_prompt

logger = logging.getLogger(__name__)


def enrich_candidate(
    candidate: CandidateInput,
    anthropic_client: anthropic.Anthropic,
    tavily_client,
    status_callback: Callable[[str], None] | None = None,
) -> dict:
    """
    Agentic tool-use loop: Claude decides what to search; Tavily executes searches.

    Claude API calls are sequential (respects 50 RPM Tier 1 limit).
    When Claude batches multiple tool_use blocks in a single response, those
    Tavily searches run concurrently — safe because Tavily has a 30/s limit,
    and we never parallelise Claude calls themselves.

    status_callback(str) pushes live status strings to the Streamlit UI.
    """
    messages = [
        {"role": "user", "content": build_enrichment_user_prompt(candidate)}
    ]

    all_search_results: list[dict] = []
    iteration = 0

    while iteration < MAX_TOOL_ITERATIONS:
        iteration += 1

        try:
            response = _call_claude_with_retry(
                client=anthropic_client,
                system=ENRICHMENT_SYSTEM_PROMPT,
                tools=ALL_TOOLS,
                messages=messages,
                max_tokens=3000,
            )
        except Exception as e:
            logger.error(f"Enrichment API call failed for {candidate.full_name}: {e}")
            break

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason in ("end_turn", "max_tokens"):
            # max_tokens means the summary was cut at the token limit — still usable.
            if response.stop_reason == "max_tokens":
                logger.warning(f"Enrichment hit max_tokens for {candidate.full_name} — summary may be truncated")
            break

        if response.stop_reason == "tool_use":
            tool_blocks = [b for b in response.content if b.type == "tool_use"]

            # Emit status messages on the main thread BEFORE dispatching to workers.
            # Streamlit's session context only exists on the main thread — calling
            # run_status.write() from a ThreadPoolExecutor worker raises NoSessionContext.
            if status_callback:
                for block in tool_blocks:
                    query_display = block.input.get("query") or block.input.get("firm_name", "")
                    status_callback(f"Searching: {str(query_display)[:80]}")

            # Execute all tool_use blocks from this response concurrently.
            # These are Tavily searches only — no additional Claude API calls.
            def _run_tool(block):
                result = execute_tool(
                    tool_name=block.name,
                    tool_input=block.input,
                    tavily_client=tavily_client,
                    logger=logger,
                )
                return block, result

            workers = min(len(tool_blocks), 3)
            if workers > 1:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    block_results = list(pool.map(_run_tool, tool_blocks))
            else:
                block_results = [_run_tool(b) for b in tool_blocks]

            tool_results = []
            for block, result_text in block_results:
                all_search_results.append({
                    "tool": block.name,
                    "input": block.input,
                    "result": result_text,
                })
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                })

            messages.append({"role": "user", "content": tool_results})

        else:
            logger.warning(f"Unexpected stop_reason '{response.stop_reason}' for {candidate.full_name}")
            break

    enrichment_summary = _extract_final_text(messages)

    return {
        "candidate_id": candidate.candidate_id,
        "full_name": candidate.full_name,
        "search_results": all_search_results,
        "enrichment_summary": enrichment_summary,
        "total_searches": len(all_search_results),
        "iterations": iteration,
    }


def _call_claude_with_retry(
    client: anthropic.Anthropic,
    system: str,
    tools: list,
    messages: list,
    max_tokens: int,
    max_retries: int = 4,
):
    """Call Claude API with exponential backoff on rate limit and connection errors."""
    for attempt in range(max_retries):
        try:
            return client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=max_tokens,
                system=system,
                tools=tools,
                messages=messages,
            )
        except (anthropic.RateLimitError, anthropic.APIConnectionError) as e:
            if attempt == max_retries - 1:
                raise
            wait = 3 * (2 ** attempt)  # 3s, 6s, 12s, 24s
            logger.warning(
                f"{type(e).__name__} — retrying in {wait}s "
                f"(attempt {attempt + 1}/{max_retries}): {e}"
            )
            time.sleep(wait)
        except anthropic.APIStatusError as e:
            if e.status_code == 529 and attempt < max_retries - 1:
                wait = 3 * (2 ** attempt)
                logger.warning(f"API overloaded, retrying in {wait}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
            else:
                logger.error(f"API status error {e.status_code}: {e.message}")
                raise


def _extract_final_text(messages: list) -> str:
    """Extract text from the last assistant message in history."""
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", [])
        if isinstance(content, str):
            return content
        for block in content:
            if hasattr(block, "type") and block.type == "text":
                return block.text
    return ""
