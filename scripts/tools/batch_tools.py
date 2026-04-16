"""
OpenAI Batch API tools for Phase 1 (summarization) and Phase 2 (population).

Provides helpers to:
  - Build JSONL request files (one line per node)
  - Upload and submit batches
  - Poll for completion
  - Download and parse results
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Output directories for batch artifacts
# ---------------------------------------------------------------------------

_BATCH_DIR = Path(__file__).resolve().parent.parent.parent / "output" / "batches"
_BATCH_IDS_DIR = _BATCH_DIR / "batch_ids"
_BATCH_INPUTS_DIR = _BATCH_DIR / "inputs"
_BATCH_OUTPUTS_DIR = _BATCH_DIR / "outputs"

for _d in (_BATCH_IDS_DIR, _BATCH_INPUTS_DIR, _BATCH_OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL = "gpt-4o-mini"
BATCH_SIZE = 5_000
POLL_INTERVAL_SECONDS = 10

# Batch API pricing (50% off standard)
INPUT_COST_PER_M = 0.075
OUTPUT_COST_PER_M = 0.30


# ---------------------------------------------------------------------------
# Phase 1: Summarization request builders
# ---------------------------------------------------------------------------

_SUMMARIZE_PROMPT_TEMPLATE = (
    'Provide a concise biological/chemical summary of "{entity_name}" '
    "(entity type: {entity_type}).\n\n"
    "Include all of the following that apply:\n"
    "- What it is and its primary biological role or function\n"
    "- Organism(s) it is found in\n"
    "- Tissue(s) and cell type(s) where it is located or most relevant\n"
    "- Subcellular compartment\n"
    "- Organ system(s) involved\n"
    "- Relevant biological processes and pathways\n"
    "- Molecular function\n"
    "- Mechanism of action\n"
    "- Disease associations and clinical relevance\n"
    "- Chemical/drug classification (if applicable)\n"
    "- Regulatory role and interaction types\n"
    "- Inheritance patterns (if applicable)\n"
    "- Developmental stage relevance\n"
    "- Taxonomic domain\n"
    "- Expression context\n\n"
    "Be factual and specific. Cover as many dimensions as possible."
)


def build_summarize_request(node: dict, custom_id: str) -> dict:
    """Build a single Batch API request line for entity summarization.

    Parameters
    ----------
    node : dict
        Node dict with at least 'name' and 'label' keys.
    custom_id : str
        Unique identifier for this request (e.g. the node ID).

    Returns
    -------
    dict suitable for writing as one JSONL line.
    """
    prompt = _SUMMARIZE_PROMPT_TEMPLATE.format(
        entity_name=node["name"],
        entity_type=node["label"],
    )
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL,
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": prompt}],
        },
    }


# ---------------------------------------------------------------------------
# Phase 2: Population request builders
# ---------------------------------------------------------------------------


def build_populate_request(
    node: dict,
    summary_text: str,
    schema: dict,
    custom_id: str,
) -> dict:
    """Build a single Batch API request line for schema-field mapping.

    Parameters
    ----------
    node : dict
        Node dict with at least 'id', 'name', 'label'.
    summary_text : str
        The LLM-generated summary from Phase 1.
    schema : dict
        The current schema (used to list fields and vocabulary terms).
    custom_id : str
        Unique identifier for this request.

    Returns
    -------
    dict suitable for writing as one JSONL line.
    """
    fields = schema.get("fields", [])
    vocabs = schema.get("controlled_vocabularies", {})

    field_specs = []
    for f in fields:
        if f.get("field_type") != "controlled":
            continue
        vocab_name = f.get("controlled_vocabulary", "")
        terms = vocabs.get(vocab_name, [])
        field_specs.append(
            f'- {f["name"]}: {f.get("description", "")}  '
            f"Terms: {json.dumps(terms)}"
        )

    fields_block = "\n".join(field_specs)

    prompt = (
        f"You are mapping biological context to schema fields.\n\n"
        f'Entity: "{node["name"]}" (ID: {node["id"]}, type: {node["label"]})\n\n'
        f"Summary:\n{summary_text}\n\n"
        f"Schema fields and their allowed controlled-vocabulary terms:\n{fields_block}\n\n"
        f"For EACH field listed above, return a JSON object where:\n"
        f"- Each key is a field name (use EXACTLY the field names listed above — do not add new fields)\n"
        f"- Each value is a LIST of matching vocabulary terms (the entity may match multiple)\n"
        f"- Use null if the field does not apply or cannot be determined\n"
        f"- Do NOT use placeholder values like \"not_applicable\", \"unknown\", \"none\", "
        f"\"not_specified\", \"none_known\", or similar — use null instead\n"
        f"- Only use terms from the provided vocabulary lists\n"
        f'- If a concept fits but no existing term matches, use the CLOSEST existing term '
        f'and also add a "suggested_additions" key mapping field names to suggested new vocabulary terms\n\n'
        f"Return valid JSON only. Example:\n"
        f'{{"organism": ["Homo sapiens"], "tissue_location": ["blood", "liver"], '
        f'"cell_type": null, "suggested_additions": {{"tissue_location": ["bone_marrow_stroma"]}}}}'
    )

    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL,
            "max_tokens": 2000,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": prompt}],
        },
    }


# ---------------------------------------------------------------------------
# JSONL file writing
# ---------------------------------------------------------------------------


def write_jsonl(requests: list[dict], path: Path) -> None:
    """Write a list of request dicts as a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")
    print(f"[batch] Wrote {len(requests)} requests to {path}")


# ---------------------------------------------------------------------------
# Batch submission, polling, and download
# ---------------------------------------------------------------------------


def submit_batch(
    client: OpenAI,
    jsonl_path: Path,
    phase: str,
    batch_number: int,
) -> str:
    """Upload a JSONL file and create a batch. Returns the batch ID.

    Also saves a metadata JSON file to output/batches/batch_ids/.
    """
    # Upload the input file
    with open(jsonl_path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")

    # Create the batch
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    # Save batch metadata
    meta = {
        "batch_id": batch.id,
        "input_file_id": file_obj.id,
        "phase": phase,
        "batch_number": batch_number,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "status": batch.status,
        "jsonl_input": str(jsonl_path),
    }
    meta_path = _BATCH_IDS_DIR / f"{phase}_batch_{batch_number:03d}.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[batch] Submitted {phase} batch {batch_number:03d} → {batch.id} (status: {batch.status})")
    return batch.id


def poll_batch(client: OpenAI, batch_id: str) -> dict:
    """Poll a batch until it reaches a terminal state. Returns the batch object as a dict."""
    last_completed = 0
    with tqdm(desc=f"Batch {batch_id[:12]}…", unit=" poll", dynamic_ncols=True) as pbar:
        while True:
            batch = client.batches.retrieve(batch_id)
            status = batch.status
            total = batch.request_counts.total if batch.request_counts else 0
            completed = batch.request_counts.completed if batch.request_counts else 0

            pbar.set_postfix_str(f"{completed}/{total} req | {status}")
            pbar.update(1)

            if completed > last_completed:
                tqdm.write(f"  [{batch_id[:12]}] {completed}/{total} requests completed")
                last_completed = completed

            if status in ("completed", "failed", "expired", "cancelled"):
                return {
                    "batch_id": batch_id,
                    "status": status,
                    "output_file_id": getattr(batch, "output_file_id", None),
                    "error_file_id": getattr(batch, "error_file_id", None),
                    "request_counts": {
                        "total": total,
                        "completed": completed,
                        "failed": batch.request_counts.failed if batch.request_counts else 0,
                    },
                }

            time.sleep(POLL_INTERVAL_SECONDS)


def download_batch_results(
    client: OpenAI,
    output_file_id: str,
    dest_path: Path,
) -> list[dict]:
    """Download batch output JSONL and parse into a list of result dicts."""
    content = client.files.content(output_file_id)
    raw = content.text
    dest_path.write_text(raw, encoding="utf-8")
    print(f"[batch] Downloaded results to {dest_path}")

    results = []
    for line in raw.strip().split("\n"):
        if line.strip():
            results.append(json.loads(line))
    return results


# ---------------------------------------------------------------------------
# Result parsing helpers
# ---------------------------------------------------------------------------


def parse_phase1_results(results: list[dict]) -> dict[str, str]:
    """Parse Phase 1 batch output into {custom_id: summary_text}."""
    summaries = {}
    for item in results:
        cid = item.get("custom_id", "")
        response = item.get("response", {})
        body = response.get("body", {})
        choices = body.get("choices", [])
        if choices:
            summaries[cid] = choices[0].get("message", {}).get("content", "")
        else:
            summaries[cid] = ""
    return summaries


def parse_phase2_results(results: list[dict]) -> tuple[list[dict], dict[str, list[str]]]:
    """Parse Phase 2 batch output into (populated_nodes, suggested_additions).

    Returns
    -------
    populated_nodes : list of dicts, each with schema fields populated
    all_suggestions : dict of field_name -> [suggested new terms]
    """
    populated = []
    all_suggestions: dict[str, list[str]] = {}

    for item in results:
        cid = item.get("custom_id", "")
        response = item.get("response", {})
        body = response.get("body", {})
        choices = body.get("choices", [])

        if not choices:
            populated.append({"id": cid, "error": "no_response"})
            continue

        content = choices[0].get("message", {}).get("content", "{}")
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            populated.append({"id": cid, "error": "invalid_json"})
            continue

        # Extract suggested additions (LLM may return null or a list instead of a dict)
        suggestions = result.pop("suggested_additions", {}) or {}
        if not isinstance(suggestions, dict):
            suggestions = {}
        for field_name, terms in suggestions.items():
            if isinstance(terms, list):
                existing = all_suggestions.setdefault(field_name, [])
                for t in terms:
                    if t not in existing:
                        existing.append(t)

        result["_custom_id"] = cid
        populated.append(result)

    return populated, all_suggestions


def estimate_batch_cost(num_nodes: int) -> float:
    """Estimate total Batch API cost for Phase 1 + Phase 2."""
    # Phase 1: ~300 input, ~800 output per node
    p1_in = num_nodes * 300
    p1_out = num_nodes * 800
    # Phase 2: ~3000 input, ~600 output per node
    p2_in = num_nodes * 3000
    p2_out = num_nodes * 600

    total_in = p1_in + p2_in
    total_out = p1_out + p2_out

    return (
        total_in * INPUT_COST_PER_M / 1_000_000
        + total_out * OUTPUT_COST_PER_M / 1_000_000
    )
