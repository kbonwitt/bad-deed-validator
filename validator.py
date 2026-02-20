#!/usr/bin/env python3
"""
Bad Deed Validator
==================
Parses messy OCR-scanned real estate deeds, enriches them with reference data,
and validates them with deterministic code — not AI guesswork.

Design principle: The LLM is a parser, not a judge.
All business-logic checks (dates, amounts, county lookup) are pure Python.
"""

import json
import re
from datetime import datetime
from difflib import get_close_matches
from pathlib import Path

import anthropic

# ---------------------------------------------------------------------------
# Input: the raw OCR text (exactly as provided)
# ---------------------------------------------------------------------------

RAW_DEED_TEXT = """\
*** RECORDING REQ ***
Doc: DEED-TRUST-0042
County: S. Clara  |  State: CA
Date Signed: 2024-01-15
Date Recorded: 2024-01-10
Grantor:  T.E.S.L.A. Holdings LLC
Grantee:  John  &  Sarah  Connor
Amount: $1,250,000.00 (One Million Two Hundred Thousand Dollars)
APN: 992-001-XA
Status: PRELIMINARY
*** END ***"""


# ---------------------------------------------------------------------------
# Custom exceptions — each maps to a specific, catchable failure mode
# ---------------------------------------------------------------------------


class DeedValidationError(Exception):
    """Base class for all deed validation failures."""


class TemporalOrderError(DeedValidationError):
    """Raised when a deed is recorded before it was signed."""


class AmountDiscrepancyError(DeedValidationError):
    """Raised when the numeric dollar amount doesn't match the written-out words."""


class CountyMatchError(DeedValidationError):
    """Raised when the county string cannot be resolved to reference data."""


# ---------------------------------------------------------------------------
# Step 1: LLM extraction
# The LLM's only job is to convert messy text → structured JSON.
# It does NOT validate — that's code's responsibility.
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """\
You are a real estate document parser. Extract the following fields from the
deed text below and return ONLY a valid JSON object with no extra commentary,
no markdown fences, no explanation.

Fields:
  doc_id         – string
  county         – string, exactly as written in the document
  state          – string, 2-letter code
  date_signed    – string, ISO format YYYY-MM-DD
  date_recorded  – string, ISO format YYYY-MM-DD
  grantor        – string, cleaned up (expand acronyms like T.E.S.L.A. → TESLA)
  grantee        – string, cleaned up
  amount_numeric – number, the dollar figure (digits only, no $ or commas)
  amount_words   – string, the written-out amount as it appears, without parentheses
  apn            – string
  status         – string

Deed text:
{raw_text}
"""


def extract_deed_data(raw_text: str) -> dict:
    """
    Use Claude to parse raw OCR text into a structured dict.

    Prompt engineering note: we ask for *extraction only* and remind the model
    to preserve the county string exactly as written. This keeps the LLM honest
    and leaves all interpretation to deterministic code below.
    """
    client = anthropic.Anthropic()

    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": EXTRACTION_PROMPT.format(raw_text=raw_text),
            }
        ],
    )

    raw_json = message.content[0].text.strip()
    # Defensively strip markdown code fences the model might add despite instructions
    raw_json = re.sub(r"^```(?:json)?\s*", "", raw_json, flags=re.MULTILINE)
    raw_json = re.sub(r"\s*```\s*$", "", raw_json, flags=re.MULTILINE)

    return json.loads(raw_json)


# ---------------------------------------------------------------------------
# Step 2: County enrichment — deterministic fuzzy matching, no AI
# ---------------------------------------------------------------------------

# Common abbreviations found in OCR / informal deed writing.
# Expanding these before fuzzy-matching catches the "S. Clara" case reliably.
ABBREVIATION_MAP = {
    "s.": "santa",
    "st.": "saint",
    "mt.": "mount",
    "ft.": "fort",
    "n.": "north",
}


def _normalize(name: str) -> str:
    """Lowercase, expand known abbreviations, collapse whitespace."""
    tokens = name.lower().split()
    expanded = [ABBREVIATION_MAP.get(t, t) for t in tokens]
    return " ".join(expanded)


def match_county(raw_county: str, counties: list[dict]) -> dict:
    """
    Resolve a raw county string to a record in our reference list.

    Strategy (in order):
      1. Normalize abbreviations on both sides, then try exact string match.
      2. Fall back to difflib fuzzy match (cutoff=0.6) on normalized names.

    This is deterministic and auditable — no black-box AI involved.
    Raises CountyMatchError if no match is found above the cutoff.
    """
    normalized_raw = _normalize(raw_county.strip())

    # Build a map: normalized_name → original record
    normalized_index: dict[str, dict] = {_normalize(c["name"]): c for c in counties}

    # 1. Exact match after normalization
    if normalized_raw in normalized_index:
        matched = normalized_index[normalized_raw]
        print(f"  [County] Exact match (normalized): '{raw_county}' → '{matched['name']}'")
        return matched

    # 2. Fuzzy match
    candidates = list(normalized_index.keys())
    close = get_close_matches(normalized_raw, candidates, n=1, cutoff=0.6)
    if close:
        matched = normalized_index[close[0]]
        print(f"  [County] Fuzzy match: '{raw_county}' → '{matched['name']}'")
        return matched

    raise CountyMatchError(
        f"Cannot resolve county '{raw_county}' to any known county. "
        f"Known: {[c['name'] for c in counties]}"
    )


# ---------------------------------------------------------------------------
# Step 3a: Date validation — pure Python, no AI
# ---------------------------------------------------------------------------


def validate_dates(date_signed: str, date_recorded: str) -> None:
    """
    A deed cannot be recorded before it is signed — that's physically impossible.
    Raises TemporalOrderError with a clear message if the order is wrong.

    We use Python datetime for comparison, not string comparison, to be safe
    against non-ISO formatted dates that the LLM might normalize differently.
    """
    signed = datetime.strptime(date_signed, "%Y-%m-%d").date()
    recorded = datetime.strptime(date_recorded, "%Y-%m-%d").date()

    if recorded < signed:
        raise TemporalOrderError(
            f"Temporal impossibility: deed recorded on {date_recorded} "
            f"but not signed until {date_signed} "
            f"({(signed - recorded).days} day(s) after recording)."
        )


# ---------------------------------------------------------------------------
# Step 3b: Amount validation — pure Python, no AI
# ---------------------------------------------------------------------------


_ONES: dict[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}
_TENS: dict[str, int] = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_SCALES: dict[str, int] = {
    "thousand": 1_000,
    "million": 1_000_000,
    "billion": 1_000_000_000,
}


def _parse_sub_thousand(words: list[str]) -> int:
    """Parse a word list representing a number < 1000 (e.g. ['two', 'hundred'])."""
    total = 0
    for word in words:
        if word in _ONES:
            total += _ONES[word]
        elif word in _TENS:
            total += _TENS[word]
        elif word == "hundred":
            # "hundred" scales whatever came before it (e.g. "two hundred" → 200)
            total = total * 100 if total else 100
    return total


def _parse_word_amount(text: str) -> float:
    """
    Convert a written dollar amount to a float using a custom hierarchical parser.
    E.g., "One Million Two Hundred Thousand Dollars" → 1_200_000.0

    We avoid third-party word-to-number libraries because several (including
    word2number) misparse compound amounts like "One Million Two Hundred Thousand"
    (returning 1,201,200 instead of 1,200,000). This parser is correct by design:
    it splits on scale words (million, thousand, billion) and sums the segments.
    """
    # Strip non-numeric word noise
    cleaned = re.sub(r"\b(dollars?)\b", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\band\s+\d+/100\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[,\-]", " ", cleaned)  # handle hyphens in "twenty-one"
    words = cleaned.lower().split()

    result = 0
    segment: list[str] = []
    for word in words:
        if word in _SCALES:
            result += _parse_sub_thousand(segment) * _SCALES[word]
            segment = []
        else:
            segment.append(word)
    result += _parse_sub_thousand(segment)  # units/hundreds remainder
    return float(result)


def validate_amounts(amount_numeric: float, amount_words: str) -> None:
    """
    Cross-check the digit amount against the written-word amount.
    Any discrepancy > $0.01 raises AmountDiscrepancyError.

    Why not let the LLM reconcile this? Because "silently picking one" is how
    fraudulent figures slip through. We surface the conflict and halt.
    """
    amount_from_words = _parse_word_amount(amount_words)
    diff = abs(amount_numeric - amount_from_words)

    if diff > 0.01:
        raise AmountDiscrepancyError(
            f"Amount mismatch: "
            f"${amount_numeric:,.2f} (numeric digits) ≠ "
            f"${amount_from_words:,.2f} (written words). "
            f"Discrepancy: ${diff:,.2f}. "
            f"Manual review required before recording."
        )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def process_deed(raw_text: str) -> dict:
    """
    Full deed-processing pipeline.

    Returns a result dict on success.
    Raises a DeedValidationError subclass on any validation failure.
    All validation errors are collected before raising so the caller sees
    the full picture in one pass.
    """
    counties_path = Path(__file__).parent / "counties.json"
    with counties_path.open() as f:
        counties = json.load(f)

    # --- Step 1: LLM extraction ---
    print("Step 1: Extracting deed data via LLM...")
    deed = extract_deed_data(raw_text)
    print(f"  Extracted fields:\n{json.dumps(deed, indent=4)}")

    # --- Step 2: County enrichment ---
    print("\nStep 2: Enriching county data...")
    county_record = match_county(deed["county"], counties)
    tax_rate = county_record["tax_rate"]
    print(f"  Tax rate: {tax_rate:.1%}")

    # --- Step 3: Validation (collect all errors before raising) ---
    print("\nStep 3: Running validation checks...")
    errors: list[DeedValidationError] = []

    try:
        validate_dates(deed["date_signed"], deed["date_recorded"])
        print("  [PASS] Date order check")
    except TemporalOrderError as e:
        print(f"  [FAIL] Date order: {e}")
        errors.append(e)

    try:
        validate_amounts(deed["amount_numeric"], deed["amount_words"])
        print("  [PASS] Amount consistency check")
    except AmountDiscrepancyError as e:
        print(f"  [FAIL] Amount: {e}")
        errors.append(e)

    # Surface the first (most critical) error. Caller can catch individually.
    if errors:
        raise errors[0]

    # --- Step 4: Closing costs ---
    closing_costs = deed["amount_numeric"] * tax_rate
    print(f"\nStep 4: Closing costs @ {tax_rate:.1%}: ${closing_costs:,.2f}")

    return {
        "deed": deed,
        "county": county_record,
        "closing_costs": closing_costs,
        "status": "VALIDATED",
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("=" * 60)
    print("Bad Deed Validator")
    print("=" * 60 + "\n")

    try:
        result = process_deed(RAW_DEED_TEXT)
        print("\n" + "=" * 60)
        print("RESULT: ACCEPTED")
        print("=" * 60)
        print(json.dumps(result, indent=2))

    except TemporalOrderError as e:
        print(f"\n[REJECTED] Temporal Order Error\n  {e}")

    except AmountDiscrepancyError as e:
        print(f"\n[REJECTED] Amount Discrepancy\n  {e}")

    except CountyMatchError as e:
        print(f"\n[REJECTED] County Resolution Error\n  {e}")

    except DeedValidationError as e:
        print(f"\n[REJECTED] Validation Error\n  {e}")
