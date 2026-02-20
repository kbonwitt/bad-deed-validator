# Bad Deed Validator

Parses and validates OCR-scanned real estate deeds using an LLM for extraction and deterministic Python for all business logic.

**Core principle:** The LLM is a parser, not a judge. It extracts structured data from messy text. All validation (dates, amounts, county matching) is pure code — reproducible, auditable, and hallucination-proof.

## How It Works

1. **LLM Extraction** — Claude parses raw OCR text into clean JSON (extraction only, no validation)
2. **County Enrichment** — Abbreviation expansion (`"S. Clara"` -> `"Santa Clara"`) + `difflib` fuzzy matching to look up the tax rate
3. **Validation** — Code-only checks that reject the deed on failure:
   - **Date order**: `datetime` comparison catches recording before signing
   - **Amount consistency**: Custom word-to-number parser cross-checks digits vs. written words, flags the $50k discrepancy
4. **Closing Costs** — `amount * tax_rate` (only reached if validation passes)

## Running It

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...
python validator.py
```

Use `python validator.py --mock` to test without an API key.

## Expected Output

The sample deed has two intentional errors, both caught:

```
Step 2: Enriching county data...
  [County] Exact match (normalized): 'S. Clara' -> 'Santa Clara'
Step 3: Running validation checks...
  [FAIL] Date order: deed recorded on 2024-01-10 but not signed until 2024-01-15
  [FAIL] Amount: $1,250,000.00 (numeric) != $1,200,000.00 (written). Discrepancy: $50,000.00

[REJECTED] Temporal Order Error
```
