# Bad Deed Validator

A "paranoid" pipeline for parsing and validating OCR-scanned real estate deeds before they touch a blockchain.

## The Core Design Principle

> **The LLM is a parser, not a judge.**

The LLM does one thing: convert messy OCR text into a clean JSON object. Every piece of business logic — date ordering, amount consistency, county resolution — is implemented in deterministic Python code. This means failures are reproducible, auditable, and not subject to hallucination.

## Architecture

```
Raw OCR Text
     │
     ▼
┌─────────────────────────────┐
│  Step 1: LLM Extraction     │  Claude parses text → structured dict
│  (anthropic SDK)            │  Prompt: "extract only, do not validate"
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Step 2: County Enrichment  │  "S. Clara" → "Santa Clara" → tax_rate 0.012
│  (pure Python, difflib)     │  Abbreviation expansion + fuzzy match
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Step 3: Validation         │  All checks are code, not AI
│  ├─ Date order check        │  recorded >= signed? (datetime comparison)
│  └─ Amount consistency      │  digits vs. words (word2number library)
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Step 4: Closing Costs      │  amount × tax_rate
└─────────────────────────────┘
```

## How Each Problem Is Solved

### Date Error (`recorded < signed`)

**Code, not AI.** `validate_dates()` parses both ISO strings with `datetime.strptime` and does a direct comparison. If `date_recorded < date_signed`, it raises `TemporalOrderError` with the exact delta in days. This cannot be fooled by a confident-sounding hallucination.

```
[FAIL] Date order: Temporal impossibility: deed recorded on 2024-01-10
       but not signed until 2024-01-15 (5 day(s) after recording).
```

### Amount Discrepancy (`$1,250,000` vs. `"One Million Two Hundred Thousand"`)

**Code, not AI.** `validate_amounts()` uses the `word2number` library to parse the written-word amount into a float independently, then subtracts. Any gap > $0.01 raises `AmountDiscrepancyError`. The system does not silently pick one version — it halts and demands human review.

```
[FAIL] Amount: $1,250,000.00 (numeric digits) ≠ $1,200,000.00 (written words).
       Discrepancy: $50,000.00. Manual review required before recording.
```

### County Matching (`"S. Clara"` → `"Santa Clara"`)

**Deterministic fuzzy matching, not AI.** `match_county()` runs a two-stage lookup:

1. **Abbreviation expansion**: a small lookup table maps `"s."` → `"santa"`, `"st."` → `"saint"`, etc. Both the raw input and the reference names are normalized before comparison.
2. **Fuzzy match fallback**: `difflib.get_close_matches` handles anything the abbreviation table misses (typos, OCR artifacts), with a 0.6 similarity cutoff.

This approach is transparent, tweakable, and produces the same result every run.

## Running It

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...
python validator.py
```

Expected output (this deed has two validation failures):

```
Step 1: Extracting deed data via LLM...
Step 2: Enriching county data...
  [County] Exact match (normalized): 'S. Clara' → 'Santa Clara'
Step 3: Running validation checks...
  [FAIL] Date order: Temporal impossibility: deed recorded on 2024-01-10
         but not signed until 2024-01-15 (5 day(s) after recording).
  [FAIL] Amount: $1,250,000.00 (numeric digits) ≠ $1,200,000.00 (written words).
         Discrepancy: $50,000.00. Manual review required before recording.

[REJECTED] Temporal Order Error
```

## Error Taxonomy

| Exception | Trigger |
|---|---|
| `TemporalOrderError` | `date_recorded < date_signed` |
| `AmountDiscrepancyError` | numeric ≠ written-word amount |
| `CountyMatchError` | county string can't be resolved to reference data |

All three are subclasses of `DeedValidationError`, so callers can catch them individually or with a single broad handler.

## Files

| File | Purpose |
|---|---|
| `validator.py` | Main script — all logic lives here |
| `counties.json` | Reference data: county names and tax rates |
| `requirements.txt` | `anthropic` |
