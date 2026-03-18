# PPP Candidate Briefing Agent

## What This Does

The PPP Candidate Briefing Agent is an AI-powered research tool built exclusively for Platinum Pacific Partners, a specialist executive search firm focused on the Australian funds management industry. Given a list of candidate names and employers, the agent runs live web research on each person and their firm, then synthesises a structured briefing containing: estimated tenure, firm AUM context, corporate event signals, a mobility score (1–5), a role fit score (1–10), and a personalised outreach hook a consultant can use in the first 30 seconds of a call. Every briefing is grounded in real web data — not generic LLM knowledge.

## Prerequisites

- Python 3.11+
- Anthropic API key → [console.anthropic.com](https://console.anthropic.com)
- Tavily Search API key → [app.tavily.com](https://app.tavily.com) (free tier, no credit card required)

## Setup

```bash
git clone <repo>
cd ppp-recruiter-agent
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env — paste in your ANTHROPIC_API_KEY and TAVILY_API_KEY
streamlit run app.py
```

## How to Use

1. Open the app in your browser (Streamlit will print the local URL, usually `http://localhost:8501`)
2. **Step 1** — Upload your `candidates.csv` file (must have columns: `full_name`, `current_employer`, `current_title`, `linkedin_url`)
3. **Step 2** — Review or edit the role specification — this is what every candidate is scored against; change it for each new mandate
4. **Step 3** — Click **Generate Briefings** and watch live search activity in real time
5. When complete, review each candidate card (mobility score, role fit, outreach hook), then download `output.json`

## Architecture

```
candidates.csv
     │
     ▼
 CSV Parser (agent/utils.py)
     │
     ▼ for each candidate
┌────────────────────────────────────────────┐
│  STEP 1: ENRICHMENT (agent/enrichment.py)  │
│                                            │
│  Claude (claude-sonnet-4-6)                │
│    ├── search_candidate_profile ──► Tavily │
│    ├── search_firm_context      ──► Tavily │
│    └── search_corporate_events  ──► Tavily │
│                                            │
│  Agentic loop: 3–6 searches per candidate  │
│  Output: structured research summary       │
└────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────┐
│  STEP 2: BRIEFING (agent/briefing.py)      │
│                                            │
│  Claude (claude-sonnet-4-6)                │
│    Input: research summary + role spec     │
│    Output: validated JSON briefing         │
│    Schema enforced by Pydantic v2          │
└────────────────────────────────────────────┘
     │
     ▼
output.json (OutputSchema: list of CandidateBriefing)
```

## Why Three Tools?

Most recruiter tools search candidate profiles and stop there. The third tool — `search_corporate_events` — is what separates a good mobility signal from a guess. When Pendal was acquired by Perpetual in January 2023, every senior distribution professional at Pendal became a potential mover — their team was being restructured, their brand was disappearing, and their bonus pool was uncertain. When Magellan's flagship fund lost 40% of AUM in 2022–2023, senior BDMs faced an existential threat to their comp. These events don't show up in a LinkedIn profile search; they show up when you specifically search for corporate restructures, M&A activity, and leadership changes at the firm. The third tool ensures these signals are captured systematically, not by luck.

## Known Limitations

- **LinkedIn not directly scraped** — LinkedIn blocks automated access; the agent uses general web search and public mentions instead. Tenure figures may be estimated from press releases, conference bios, and news articles rather than exact start dates. All uncertain figures are marked `[UNVERIFIED]`.
- **Tavily free tier** — Limited to ~1,000 searches/month on the free plan. For production use with 20+ candidates/week, a paid Tavily plan is recommended.
- **Tenure is estimated** — Without access to LinkedIn's API or a data provider like Proxycurl, tenure is inferred from publicly available information and may be off by 6–12 months.


## What to Build Next

1. **Proxycurl integration** — Replace estimated tenure with verified LinkedIn data (exact start dates, connection count as network proxy, endorsement count). This single change would materially improve mobility score accuracy.

2. **Candidate database + caching layer** — Once a firm's AUM and corporate events have been researched, those results should be cached. If PPP is working three mandates that all involve Perpetual or BlackRock candidates, re-running the same firm searches burns API quota unnecessarily. A simple SQLite or Supabase cache keyed by firm name + date would eliminate redundant searches and cut per-candidate cost by ~40%.

3. **Automated mandate intake + longlist generator** — When a client calls PPP with a new role, the consultant today spends 2–3 hours manually building a longlist from memory and prior searches. An intake form → role spec → automatic query against PPP's accumulated candidate database → re-scored mobility check → draft longlist in under 15 minutes would compress that cycle to one-tenth the time and make PPP faster to market than any competitor.

## Submitting Your Results

`output.json` is not gitignored — it will be included automatically when you stage all files. After running the agent on `candidates.csv`, commit it before submitting:

```bash
git add output.json
git commit -m "Add output.json — submission deliverable"
git push
```

> The brief states output.json is "the most important deliverable — it is what we evaluate directly." Do not submit without it.
