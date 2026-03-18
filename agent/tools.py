from __future__ import annotations
import logging
from tavily import TavilyClient


SEARCH_CANDIDATE_TOOL = {
    "name": "search_candidate_profile",
    "description": (
        "Search the web for publicly available career information about a specific "
        "individual in the Australian funds management industry. Use this to find: "
        "career history, how long they've been in their current role, previous employers, "
        "career progression pattern, any awards or conference appearances, and whether "
        "they have a visible public profile (LinkedIn activity, media mentions)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "The search query. Always include both full name AND current employer. "
                    "Example: 'Cathy Hales Perpetual Limited Head of Retail Sales'"
                )
            },
            "intent": {
                "type": "string",
                "enum": ["career_timeline", "public_profile", "recent_activity", "previous_employers"],
                "description": "What you are specifically trying to find."
            }
        },
        "required": ["query", "intent"]
    }
}

SEARCH_FIRM_TOOL = {
    "name": "search_firm_context",
    "description": (
        "Search for firm-level context about an Australian asset management company: "
        "AUM size, firm type (active manager, passive, boutique, global house, super fund), "
        "ownership structure, and recent corporate changes. Use this to establish the "
        "context a candidate has been operating in."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "firm_name": {
                "type": "string",
                "description": "Full name of the asset management firm."
            },
            "focus": {
                "type": "string",
                "enum": ["aum_and_size", "firm_type_and_strategy", "ownership_and_structure"],
                "description": "Aspect of the firm to research."
            }
        },
        "required": ["firm_name", "focus"]
    }
}

SEARCH_CORPORATE_EVENTS_TOOL = {
    "name": "search_corporate_events",
    "description": (
        "Search for recent corporate events at a candidate's employer that may affect "
        "career mobility: mergers, acquisitions, demergers, restructures, leadership "
        "changes, rebrands, fund closures, or strategic reviews. This is critical for "
        "assessing whether a candidate may be open to moving. Events in the past 1-3 years "
        "are most relevant."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "firm_name": {
                "type": "string",
                "description": "Name of the firm to research."
            },
            "event_type": {
                "type": "string",
                "enum": ["merger_acquisition", "restructure_layoffs", "leadership_change", "strategic_review", "fund_performance"],
                "description": "Type of corporate event to search for."
            }
        },
        "required": ["firm_name", "event_type"]
    }
}

ALL_TOOLS = [SEARCH_CANDIDATE_TOOL, SEARCH_FIRM_TOOL, SEARCH_CORPORATE_EVENTS_TOOL]


def execute_tool(
    tool_name: str,
    tool_input: dict,
    tavily_client: TavilyClient,
    logger: logging.Logger,
) -> str:
    """
    Execute a tool call. Never raises — always returns a string result.
    On failure, returns an explicit failure notice so Claude can reason about it.
    """
    try:
        query = _build_query(tool_name, tool_input)

        # Candidate profile searches need deeper results; firm/events are well-indexed
        depth = "advanced" if tool_name == "search_candidate_profile" else "basic"

        results = tavily_client.search(
            query=query,
            max_results=3,
            search_depth=depth,
            include_answer=True,  # Tavily's synthesised answer — useful for AUM/firm facts
        )

        output_parts = []

        # Include Tavily's answer if present (often most useful for AUM/firm facts)
        if results.get("answer"):
            output_parts.append(f"Summary: {results['answer']}")

        for r in results.get("results", [])[:3]:
            output_parts.append(
                f"Source: {r.get('url', 'unknown')}\n"
                f"Title: {r.get('title', '')}\n"
                f"Content: {r.get('content', '')[:400]}"
            )

        if not output_parts:
            return f"[NO_RESULTS] No results found for query: {query}"

        return "\n\n---\n\n".join(output_parts)

    except Exception as e:
        logger.warning(f"Tool {tool_name} failed: {e}")
        return f"[TOOL_ERROR] Search unavailable: {str(e)}"


def _build_query(tool_name: str, tool_input: dict) -> str:
    """Build a search query string from tool name and input."""
    if tool_name == "search_candidate_profile":
        return tool_input["query"]

    elif tool_name == "search_firm_context":
        firm = tool_input["firm_name"]
        focus = tool_input["focus"]
        focus_terms = {
            "aum_and_size": "AUM assets under management size",
            "firm_type_and_strategy": "investment strategy boutique active passive",
            "ownership_and_structure": "ownership structure parent company",
        }
        return f"{firm} Australia {focus_terms.get(focus, focus)}"

    elif tool_name == "search_corporate_events":
        firm = tool_input["firm_name"]
        event = tool_input["event_type"]
        event_terms = {
            "merger_acquisition": "merger acquisition takeover 2022 2023 2024 2025",
            "restructure_layoffs": "restructure redundancies headcount reduction 2023 2024 2025",
            "leadership_change": "CEO managing director leadership change appointment 2023 2024 2025",
            "strategic_review": "strategic review demerger separation 2023 2024 2025",
            "fund_performance": "fund performance outflows net flows AUM decline 2023 2024",
        }
        return f"{firm} Australia {event_terms.get(event, event)}"

    return tool_input.get("query", str(tool_input))
