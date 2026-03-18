# Design Note — PPP Candidate Briefing Agent

## 1. Architecture Choices and Why

**Two API calls, not one.** The pipeline separates enrichment (web research) from synthesis (briefing generation) into two distinct Claude invocations. A tool-use loop that simultaneously manages search state and produces schema-valid JSON creates tension the model resolves inconsistently. More critically: when JSON validation fails, a single-call design forces you to re-run the entire research phase just to retry the cheap synthesis step. Separating the two means a self-correction retry costs a few hundred tokens, not another full research cycle.

**Three tools because mobility lives in corporate events, not profiles.** I could have built a generic web search tool. I didn't because the five candidates in this dataset sit at firms that have each been through significant disruption — Pendal acquired by Perpetual, Magellan's AUM collapse, Challenger's distribution restructure, Fidelity's global cost programme. None of that surfaces when you search a person's name and employer. It requires an explicit query targeting corporate events at the firm. The third tool makes this systematic rather than dependent on whatever a general search happens to return.

**Forced structured output via tool_use.** The briefing call uses `tool_choice` to force Claude to call a `submit_briefing` tool whose `input_schema` is the Pydantic model's own JSON schema. The response arrives as a structured dict — no JSON parsing, no markdown-fence stripping. Pydantic v2 validators enforce field-level constraints (score ranges, sentence counts, single-line hooks), and any failure is returned to Claude as a `tool_result` with `is_error: true` so self-correction targets the exact failing field.

**`agent/config.py` as single source of truth.** Model string, iteration limits, rate-limit delays — these change together when upgrading API tiers. Scattering them as inline literals is a maintenance trap.

---

## 2. What I Would Do Differently With More Time

- **Proxycurl integration for verified tenure.** The weakest output in every briefing is tenure estimation. The agent infers start dates from press releases and bios — often right within a year, but "approximately 4 years" is a weaker signal than "joined January 2021, confirmed via LinkedIn." Proxycurl returns structured role history with exact dates, directly improving mobility score accuracy.

- **Firm-level caching.** Every invocation re-searches Perpetual's AUM and ownership structure from scratch. A SQLite cache keyed by `(firm_name, date)` with a 30-day TTL for stable data (AUM, firm type) and a 7-day TTL for corporate events would eliminate redundant searches and cut per-candidate API cost by ~40%.

- **CRM integration.** Right now the output lands in a JSON file that sits outside PPP's workflow. When a briefing is generated it should automatically create or update the candidate record in PPP's CRM (Bullhorn, Salesforce, or equivalent) — with scores, tags, and the outreach hook pre-populated. The research is only useful if it flows into the tools consultants already open every morning.

---

## 3. One Additional Automation: Candidate-to-Mandate Matching at Intake

The most time-consuming step this tool doesn't touch is the front end of a new mandate: a client calls, describes a role, and the consultant spends 2–3 hours building a longlist from memory. As PPP accumulates briefings across mandates, those JSON files become a structured database of researched, scored candidates. An intake automation would let a consultant fill a short form — role title, seniority, channel focus, firm type — and immediately receive a ranked shortlist from prior research, re-scored for current mobility using a fresh corporate events check. Speed of longlist delivery is a direct competitive differentiator: if PPP can present five pre-researched names within an hour of intake while competitors take two days, that is a tangible reason for clients to return.
