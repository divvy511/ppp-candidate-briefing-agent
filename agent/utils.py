from __future__ import annotations
import pandas as pd
import logging
from agent.config import MAX_CANDIDATES_PER_RUN
from agent.models import CandidateInput, CandidateBriefing, OutputSchema

logger = logging.getLogger(__name__)

REQUIRED_CSV_COLUMNS = {"full_name", "current_employer", "current_title", "linkedin_url"}


def extract_role_title_from_spec(role_spec: str) -> str:
    """
    Extract the role title from the first line of the role spec.

    Handles these formats:
      "Role: Head of Distribution / National BDM"   → "Head of Distribution / National BDM"
      "Head of Distribution / National BDM"          → "Head of Distribution / National BDM"
      "Role Title: Senior Portfolio Manager"         → "Senior Portfolio Manager"

    Falls back to "Unspecified Role" if no title can be extracted.
    """
    if not role_spec or not role_spec.strip():
        return "Unspecified Role"

    for line in role_spec.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip common prefixes
        for prefix in ("Role:", "Role Title:", "Title:", "Position:"):
            if line.lower().startswith(prefix.lower()):
                title = line[len(prefix):].strip()
                if title:
                    return title
        # No recognised prefix — first non-empty line is the title
        return line

    return "Unspecified Role"


def parse_csv(file) -> tuple[list[CandidateInput], str | None]:
    """
    Parse uploaded CSV (Streamlit UploadedFile or file path string).
    Returns (candidates, error_message). error_message is None on success.
    """
    try:
        df = pd.read_csv(file)
        # Normalise column names: strip whitespace, lowercase, underscores
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(r"\s+", "_", regex=True)
        )

        missing_cols = REQUIRED_CSV_COLUMNS - set(df.columns)
        if missing_cols:
            return [], f"CSV missing required columns: {', '.join(sorted(missing_cols))}"

        df = df.dropna(subset=["full_name"]).reset_index(drop=True)
        if len(df) == 0:
            return [], "CSV contains no valid candidate rows."
        if len(df) > MAX_CANDIDATES_PER_RUN:
            return [], f"CSV has {len(df)} rows — maximum is {MAX_CANDIDATES_PER_RUN} per run."

        candidates = [
            CandidateInput(
                full_name=str(row["full_name"]).strip(),
                current_employer=str(row["current_employer"]).strip(),
                current_title=str(row["current_title"]).strip(),
                linkedin_url=str(row.get("linkedin_url", "")).strip(),
                candidate_id=f"candidate_{i + 1}",
            )
            for i, row in df.iterrows()
        ]

        return candidates, None

    except Exception as e:
        return [], f"Failed to parse CSV: {e}"


def briefings_to_summary_df(briefings: list[CandidateBriefing]) -> pd.DataFrame:
    """Build summary DataFrame for the Streamlit results table."""
    rows = []
    for b in briefings:
        is_error = "[error]" in b.experience_tags
        rows.append({
            "Name": b.full_name,
            "Current Role": b.current_role.title,
            "Employer": b.current_role.employer,
            "Tenure (yrs)": "—" if is_error else b.current_role.tenure_years,
            "Mobility": "—" if is_error else f"{b.mobility_signal.score}/5",
            "Role Fit": "—" if is_error else f"{b.role_fit.score}/10",
            "Status": "⚠️ Error" if is_error else "✅ Complete",
        })
    return pd.DataFrame(rows)


def validate_and_serialize(output: OutputSchema) -> tuple[str, bool, str]:
    """
    Serialise to JSON and round-trip validate.
    Returns (json_string, is_valid, message).
    """
    try:
        json_str = output.model_dump_json(indent=2)
        # Round-trip validation
        OutputSchema.model_validate_json(json_str)
        count = len(output.candidates)
        errors = sum(1 for c in output.candidates if "[error]" in c.experience_tags)
        msg = f"✓ Valid JSON — {count} candidates"
        if errors:
            msg += f" ({errors} with errors — manual review needed)"
        return json_str, True, msg
    except Exception as e:
        return "{}", False, f"Validation failed: {e}"
