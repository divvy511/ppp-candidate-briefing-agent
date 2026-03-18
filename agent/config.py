"""
agent/config.py
Central configuration — all operational constants live here.
Change values here to affect the whole application.
"""

# Anthropic model to use for all API calls
CLAUDE_MODEL = "claude-sonnet-4-6"

# Max tool-use iterations per candidate during enrichment
# Research converges by iteration 4-5; 8 was wasteful diminishing returns
MAX_TOOL_ITERATIONS = 5

# Seconds to pause between candidates to respect Tier 1 rate limits (50 RPM / 30K TPM)
# At ~9 Claude calls per candidate, 8s gap keeps us safely under 50 RPM for 5 candidates
INTER_CANDIDATE_DELAY_SECONDS = 8

# Max candidates per run (operational safety limit)
MAX_CANDIDATES_PER_RUN = 20
