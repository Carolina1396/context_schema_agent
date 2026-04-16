"""
Knowledge Graph Schema Refinement Agent

Three-phase pipeline run over multiple iterations:
  Phase 1 — Summarize: one summarization request per node.
  Phase 2 — Populate: one schema-mapping request per node.
  Phase 3 — Refine: Synchronous agent loop reviews results and modifies the schema.

Each iteration processes 100 new diverse nodes and refines the schema.
The finalized schema from iteration N feeds as input to iteration N+1.

Modes:
  batch — OpenAI Batch API (cheap, slow).  Default.
  async — Direct async API calls (full price, fast).

Usage:
    source .venv/bin/activate
    python schema_agent.py --mode async --iterations 10
"""

import argparse
import os
import sys
import json
import asyncio
import random
import subprocess
import time
from pathlib import Path

from tqdm import tqdm
from openai import RateLimitError

from openai import OpenAI, AsyncOpenAI

from tools.graph_tools import (
    get_type_distribution,
    _ensure_loaded,
    _nodes,
    _nodes_by_type,
)
from tools.batch_tools import (
    build_summarize_request,
    build_populate_request,
    write_jsonl,
    submit_batch,
    poll_batch,
    download_batch_results,
    parse_phase1_results,
    parse_phase2_results,
    estimate_batch_cost,
    _BATCH_INPUTS_DIR,
    _BATCH_OUTPUTS_DIR,
    MODEL,
)
from tools.async_tools import (
    async_phase1_summarize,
    async_phase2_populate,
    estimate_async_cost,
)
from tools.schema_tools import (
    save_schema,
    finalize_schema,
    write_summary,
    write_nodes,
    load_latest_schema,
    set_run_number,
    set_archive_dir,
    cleanup_checkpoints,
    is_null_like,
    load_rejected_suggestions,
    save_rejected_suggestions,
    update_rejected_suggestions,
    filter_suggestions,
    load_vocabulary_history,
)
from tools.cell_context_tools import expand_drug_nodes, DRUG_CELL_TYPES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_TURNS = 40
NUM_NODES = 100  # overridden at runtime by --num-nodes

# Standard pricing for Phase 3 agent loop (non-batch)
INPUT_COST_PER_M = 0.15
OUTPUT_COST_PER_M = 0.60

# Schema is loaded from output/archive/schema_final_N.json (highest N)
# via load_latest_schema() at pipeline start.


# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------


class CostTracker:
    def __init__(self, budget: float):
        self.budget = budget
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        # Batch tokens tracked separately (50% off pricing)
        self.batch_input_tokens = 0
        self.batch_output_tokens = 0

    @property
    def cost(self) -> float:
        # Standard-rate tokens (Phase 3 agent loop)
        standard = (
            self.total_input_tokens * INPUT_COST_PER_M / 1_000_000
            + self.total_output_tokens * OUTPUT_COST_PER_M / 1_000_000
        )
        # Batch-rate tokens (Phase 1 & 2) — 50% off
        batch = (
            self.batch_input_tokens * (INPUT_COST_PER_M / 2) / 1_000_000
            + self.batch_output_tokens * (OUTPUT_COST_PER_M / 2) / 1_000_000
        )
        return standard + batch

    @property
    def remaining(self) -> float:
        return self.budget - self.cost

    def record(self, usage):
        """Record usage from a standard (non-batch) API call."""
        self.total_input_tokens += usage.prompt_tokens
        self.total_output_tokens += usage.completion_tokens

    def record_batch(self, results: list[dict]):
        """Record usage from batch API results."""
        for item in results:
            usage = item.get("response", {}).get("body", {}).get("usage", {})
            self.batch_input_tokens += usage.get("prompt_tokens", 0)
            self.batch_output_tokens += usage.get("completion_tokens", 0)

    def check(self) -> bool:
        """Check whether Phase 3 agent spending is still within budget.

        Only counts agent (Phase 3) tokens — batch costs from Phase 1 & 2
        are fixed and already spent, so they should not block Phase 3 from running.
        """
        agent_cost = (
            self.total_input_tokens * INPUT_COST_PER_M / 1_000_000
            + self.total_output_tokens * OUTPUT_COST_PER_M / 1_000_000
        )
        return agent_cost < self.budget

    def summary(self) -> str:
        return (
            f"Batch tokens — in: {self.batch_input_tokens:,}  out: {self.batch_output_tokens:,} | "
            f"Agent tokens — in: {self.total_input_tokens:,}  out: {self.total_output_tokens:,} | "
            f"Cost: ${self.cost:.4f} / ${self.budget:.2f}"
        )


def estimate_per_iteration_cost(mode: str = "batch") -> float:
    """Estimate cost for a single 100-node iteration: Phase 1 + Phase 2 + Phase 3."""
    if mode == "async":
        p12_cost = estimate_async_cost(NUM_NODES)
    else:
        p12_cost = estimate_batch_cost(NUM_NODES)
    # Phase 3: agent loop — ~200k in, ~60k out at standard rates
    p3_cost = 200_000 * INPUT_COST_PER_M / 1_000_000 + 60_000 * OUTPUT_COST_PER_M / 1_000_000
    return round(p12_cost + p3_cost, 2)


# ---------------------------------------------------------------------------
# Node selection — pick 100 diverse nodes
# ---------------------------------------------------------------------------


def select_diverse_nodes(node_type: str | None = None) -> list[dict]:
    """Pick 100 diverse nodes, mixing high-degree and low-degree.

    If node_type is given, sample only from that entity type.
    Otherwise sample across all 9 entity types.
    """
    _ensure_loaded()

    def _degree(n: dict) -> int:
        xrefs = n.get("xrefs", "")
        return len(xrefs.split("|")) if xrefs else 0

    if node_type:
        # Single-type mode: mix high / low / random within that type
        pool = [n for n in _nodes_by_type.get(node_type, []) if len(n.get("name", "")) > 3]
        if not pool:
            raise ValueError(f"No nodes found for type '{node_type}'")
        n_high = NUM_NODES // 3
        n_low = NUM_NODES // 3
        n_rand = NUM_NODES - n_high - n_low
        high = sorted(pool, key=_degree, reverse=True)[:n_high]
        low = sorted(pool, key=_degree)[:n_low]
        used_ids = {n["id"] for n in high + low}
        remaining = [n for n in pool if n["id"] not in used_ids]
        rand = random.sample(remaining, min(n_rand, len(remaining)))
        selected = high + low + rand
    else:
        entity_types = [
            "Disease", "MacromolecularMachine", "ChemicalSubstance",
            "BiologicalProcessOrActivity", "OrganismTaxon", "GeneFamily",
            "PhenotypicFeature", "Pathway", "AnatomicalEntity",
        ]
        selected = []
        per_type = max(NUM_NODES // len(entity_types), 1)
        for etype in entity_types:
            candidates = [n for n in _nodes_by_type.get(etype, []) if len(n.get("name", "")) > 3]
            if not candidates:
                continue
            high = sorted(candidates, key=_degree, reverse=True)[:max(per_type // 3, 1)]
            low = sorted(candidates, key=_degree)[:max(per_type // 3, 1)]
            used = {n["id"] for n in high + low}
            remaining = [n for n in candidates if n["id"] not in used]
            rand = random.sample(remaining, min(per_type - len(high) - len(low), len(remaining)))
            selected.extend(high + low + rand)

        # Deduplicate
        seen: set[str] = set()
        deduped = []
        for n in selected:
            if n["id"] not in seen:
                seen.add(n["id"])
                deduped.append(n)
        selected = deduped

        if len(selected) < NUM_NODES:
            extras = [n for n in _nodes if len(n.get("name", "")) > 3 and n["id"] not in seen]
            selected.extend(random.sample(extras, min(NUM_NODES - len(selected), len(extras))))

    return selected[:NUM_NODES]


# ---------------------------------------------------------------------------
# Phase 1: Summarize via Batch API
# ---------------------------------------------------------------------------


def phase1_summarize(nodes: list[dict], client: OpenAI) -> tuple[list[dict], dict[str, str]]:
    """Submit a summarization batch and return (raw_results, summaries_by_id)."""
    from tools.cell_context_tools import build_cell_context_summarize_prompt
    print("Building Phase 1 JSONL...")
    requests = []
    for i, node in enumerate(nodes):
        cid = f"{i:04d}_{node['id']}"
        if node.get("cell_type_context"):
            # Cell-type-conditional prompt — override the generic one
            prompt = build_cell_context_summarize_prompt(node)
            req = {
                "custom_id": cid,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL,
                    "max_tokens": 1000,
                    "messages": [{"role": "user", "content": prompt}],
                },
            }
        else:
            req = build_summarize_request(node, custom_id=cid)
        requests.append(req)

    jsonl_path = _BATCH_INPUTS_DIR / "phase1_batch_001.jsonl"
    write_jsonl(requests, jsonl_path)

    batch_id = submit_batch(client, jsonl_path, phase="phase1", batch_number=1)
    result = poll_batch(client, batch_id)

    if result["status"] != "completed":
        print(f"ERROR: Phase 1 batch failed with status: {result['status']}")
        sys.exit(1)

    output_path = _BATCH_OUTPUTS_DIR / "phase1_batch_001_output.jsonl"
    raw_results = download_batch_results(client, result["output_file_id"], output_path)
    summaries = parse_phase1_results(raw_results)

    success = sum(1 for s in summaries.values() if s)
    print(f"Phase 1 complete: {success}/{len(nodes)} summaries obtained.")
    return raw_results, summaries


# ---------------------------------------------------------------------------
# Phase 2: Populate via Batch API
# ---------------------------------------------------------------------------


def phase2_populate(
    nodes: list[dict],
    summaries: dict[str, str],
    schema: dict,
    client: OpenAI,
) -> tuple[list[dict], list[dict], dict[str, list[str]]]:
    """Submit a population batch and return (raw_results, populated_nodes, suggestions)."""
    from tools.cell_context_tools import build_cell_context_populate_prompt
    print("Building Phase 2 JSONL...")
    requests = []
    for i, node in enumerate(nodes):
        cid = f"{i:04d}_{node['id']}"
        summary_text = summaries.get(cid, "")
        if not summary_text:
            continue
        if node.get("cell_type_context"):
            prompt = build_cell_context_populate_prompt(node, summary_text, schema)
            req = {
                "custom_id": cid,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL,
                    "max_tokens": 2000,
                    "response_format": {"type": "json_object"},
                    "messages": [{"role": "user", "content": prompt}],
                },
            }
        else:
            req = build_populate_request(node, summary_text, schema, custom_id=cid)
        requests.append(req)

    jsonl_path = _BATCH_INPUTS_DIR / "phase2_batch_001.jsonl"
    write_jsonl(requests, jsonl_path)

    batch_id = submit_batch(client, jsonl_path, phase="phase2", batch_number=1)
    result = poll_batch(client, batch_id)

    if result["status"] != "completed":
        print(f"ERROR: Phase 2 batch failed with status: {result['status']}")
        sys.exit(1)

    output_path = _BATCH_OUTPUTS_DIR / "phase2_batch_001_output.jsonl"
    raw_results = download_batch_results(client, result["output_file_id"], output_path)
    populated, suggestions = parse_phase2_results(raw_results)

    # Restore identity fields from nodes using _custom_id set by parse_phase2_results
    node_by_cid = {f"{i:04d}_{n['id']}": n for i, n in enumerate(nodes)}
    for p in populated:
        cid = p.pop("_custom_id", "")
        if cid in node_by_cid:
            src = node_by_cid[cid]
            p["id"] = src["id"]
            p["name"] = src["name"]
            p["label"] = src["label"]
            # Preserve composite-node metadata (cell_type_context, base_id)
            for meta_field in ("cell_type_context", "base_id"):
                if meta_field in src:
                    p[meta_field] = src[meta_field]

    print(f"Phase 2 complete: {len(populated)} nodes populated.")
    return raw_results, populated, suggestions


# ---------------------------------------------------------------------------
# Phase 2 analysis — build aggregate stats for the agent
# ---------------------------------------------------------------------------


def count_responded(populated: list[dict], schema: dict) -> dict[str, int]:
    """Count how many nodes have non-null values per field (before cleaning)."""
    fields = [f for f in schema.get("fields", []) if f.get("field_type") == "controlled"]
    responded: dict[str, int] = {}
    for f in fields:
        fn = f["name"]
        responded[fn] = sum(1 for n in populated if n.get(fn) is not None)
    return responded


def clean_populated_nodes(populated: list[dict], schema: dict) -> None:
    """Remove null-like placeholder values from populated node fields in-place."""
    controlled_fields = {
        f["name"]
        for f in schema.get("fields", [])
        if f.get("field_type") == "controlled"
    }
    for node in populated:
        for fn in controlled_fields:
            val = node.get(fn)
            if isinstance(val, list):
                cleaned = [t for t in val if not is_null_like(t)]
                node[fn] = cleaned if cleaned else None
            elif isinstance(val, str) and is_null_like(val):
                node[fn] = None


def analyze_population_results(
    populated: list[dict],
    schema: dict,
    suggestions: dict[str, list[str]],
    responded: dict[str, int],
) -> str:
    """Build a text summary of Phase 2 results for the agent."""
    fields = [f for f in schema.get("fields", []) if f.get("field_type") == "controlled"]
    field_names = [f["name"] for f in fields]

    coverage: dict[str, int] = {fn: 0 for fn in field_names}
    term_freq: dict[str, dict[str, int]] = {fn: {} for fn in field_names}

    for node in populated:
        for fn in field_names:
            val = node.get(fn)
            if val is not None and val != []:
                coverage[fn] += 1
                if isinstance(val, list):
                    for term in val:
                        term_freq[fn][term] = term_freq[fn].get(term, 0) + 1

    total = len(populated)
    lines = [
        f"## Phase 2 Population Results ({total} nodes)\n",
        "### Per-field coverage:",
    ]
    for fn in field_names:
        pct = coverage[fn] / total * 100 if total else 0
        resp = responded.get(fn, 0)
        app_pct = coverage[fn] / resp * 100 if resp else 0
        lines.append(
            f"  - {fn}: {coverage[fn]}/{total} ({pct:.0f}%) | "
            f"applicable: {coverage[fn]}/{resp} ({app_pct:.0f}%)"
        )

    lines.append("\n### Most frequent terms per field (top 5):")
    for fn in field_names:
        top = sorted(term_freq[fn].items(), key=lambda x: -x[1])[:5]
        if top:
            terms_str = ", ".join(f"{t} ({c})" for t, c in top)
            lines.append(f"  - {fn}: {terms_str}")
        else:
            lines.append(f"  - {fn}: (no values)")

    if suggestions:
        lines.append("\n### Suggested vocabulary additions from Phase 2:")
        for fn, terms in sorted(suggestions.items()):
            lines.append(f"  - {fn}: {terms}")

    # Show 5 example nodes
    lines.append("\n### Example populated nodes (5 of 100):")
    examples = populated[:5]
    for ex in examples:
        lines.append(f"\n**{ex.get('name', '?')}** ({ex.get('id', '?')}):")
        for fn in field_names:
            val = ex.get(fn)
            if val is not None:
                lines.append(f"  {fn}: {json.dumps(val)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Phase 3: Synchronous agent loop for schema refinement
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "update_vocabulary",
            "description": (
                "Replace the term list for ONE controlled vocabulary field. "
                "Call this once per field you want to change. "
                "Max 20 terms per vocabulary."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "field_name": {
                        "type": "string",
                        "description": "The vocabulary key to update (e.g. 'tissue_location').",
                    },
                    "terms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "The complete new term list for this vocabulary (max 20 terms).",
                    },
                },
                "required": ["field_name", "terms"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_checkpoint",
            "description": "Save the current schema state as a versioned checkpoint.",
            "parameters": {
                "type": "object",
                "properties": {
                    "version": {
                        "type": "string",
                        "description": "Version label (e.g. '2.0', '2.1').",
                    },
                },
                "required": ["version"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_summary",
            "description": (
                "Write the refinement summary to output/archive/refinement_summary_N.md. "
                "Call this BEFORE finalize_schema."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The full markdown content for refinement_summary.md.",
                    },
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finalize_schema",
            "description": (
                "Save the final schema as schema_final_N.json and end the session. "
                "Call this AFTER write_summary."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


def make_dispatch_tool(base_schema: dict):
    """Return a dispatch_tool closure that applies incremental vocabulary updates."""
    import copy
    current_schema = copy.deepcopy(base_schema)

    def dispatch_tool(name: str, input_args: dict) -> str:
        try:
            if name == "update_vocabulary":
                field_name = input_args.get("field_name", "")
                terms = input_args.get("terms")
                if not field_name or terms is None:
                    result = {"error": "Missing 'field_name' or 'terms'."}
                elif field_name not in current_schema.get("controlled_vocabularies", {}):
                    result = {"error": f"Unknown vocabulary: '{field_name}'."}
                else:
                    current_schema["controlled_vocabularies"][field_name] = terms
                    result = {"updated": field_name, "term_count": len(terms)}
            elif name == "save_checkpoint":
                version = input_args.get("version", "unknown")
                result = save_schema(schema=current_schema, version=version)
            elif name == "finalize_schema":
                result = finalize_schema(schema=current_schema)
            elif name == "write_summary":
                content = input_args.get("content", "")
                if not content:
                    result = {"error": "Missing required 'content' argument."}
                else:
                    result = write_summary(content=content)
            else:
                result = {"error": f"Unknown tool: {name}"}
        except Exception as e:
            result = {"error": f"Tool '{name}' raised: {type(e).__name__}: {e}"}

        return json.dumps(result, ensure_ascii=False)

    return dispatch_tool


def build_refinement_prompt(
    type_summary: str,
    starting_schema: dict,
    analysis: str,
    vocab_history: str = "",
) -> str:
    """Build the system prompt for Phase 3 (schema refinement)."""
    schema_json = json.dumps(starting_schema, indent=2)

    # Build optional history section (Idea D)
    history_section = ""
    if vocab_history:
        history_section = f"""
{vocab_history}

IMPORTANT: Review the history above before making changes. If a term was
added in one iteration and removed in the next (or vice-versa), this is
oscillation — do NOT repeat the cycle. Only make a change if you have a
clear reason that differs from the previous rationale. Fields that have been
stable (no changes) across recent iterations should generally stay unchanged
unless the population data strongly justifies a modification.
"""

    return f"""You are a schema refinement agent for a biomedical knowledge graph.

## Context
You have a starting schema with 21 controlled-vocabulary fields. We have already:
1. Summarized 100 diverse nodes using LLM knowledge (Phase 1, Batch API)
2. Populated every schema field for all 100 nodes (Phase 2, Batch API)
3. Written the populated nodes to output/nodes.json (already done)

Your job is to REVIEW the population results and REFINE the controlled
vocabularies so they best fit the data without being too granular. Each
controlled-vocabulary field returns a LIST of labels (an entity may match
multiple).

CRITICAL CONSTRAINT: The schema fields are FIXED. Do NOT add, remove, or
rename any fields. Do NOT change field descriptions, field_type, applies_to_types,
or required flags. You may ONLY modify the term lists inside
`controlled_vocabularies`.

## Graph composition
{type_summary}

## Current schema
```json
{schema_json}
```

## Population analysis (from 100 nodes)
{analysis}
{history_section}
## Your tasks
1. Review the coverage stats, term frequencies, and suggested additions above.
2. REFINE the controlled vocabularies ONLY:
   - ADD new vocabulary terms where the LLM suggested additions or where coverage is low
   - MERGE or RENAME terms that are redundant or too granular
   - REMOVE vocabulary terms that are never used and don't add value
   - Do NOT add, remove, or rename fields
   - Do NOT change field descriptions, applies_to_types, or type_specific_fields
   - Each controlled vocabulary MUST NOT exceed 20 unique terms. This is a HARD
     LIMIT enforced programmatically — any vocabulary over 20 terms will be
     truncated on save. If a vocabulary is at 20 and you need to add a term,
     you MUST remove or merge an existing term first. Prefer making terms more
     general over exceeding the cap.
   - Do NOT include any placeholder or null-like values in vocabularies (e.g.
     "not_applicable", "unknown", "none", "not_specified", "none_known",
     "unclassified", "other", "not_a_drug", "not_organism_specific"). These
     will be automatically stripped on save. If no vocabulary term fits a node,
     the field value should be null — not a vocabulary term representing absence.
3. For each vocabulary you want to change, call update_vocabulary(field_name, terms)
   with the COMPLETE new term list for that field. Call it once per field.
   Fields you do not call update_vocabulary for remain unchanged.
4. Call save_checkpoint at least twice during your work (e.g. after the first
   batch of updates and again before finalizing).
5. Write a refinement summary (write_summary) with the following structure for
   EACH controlled-vocabulary field, ranked by coverage percentage (highest first):
     - **field_name** — coverage: XX% | applicable coverage: YY%
       - Terms added: term1, term2, ...
       - Count of terms added: N
       - Terms removed: term1, term2, ...
       - Count of terms removed: N
   Where "coverage" = nodes with values / total nodes, and "applicable coverage"
   = nodes with values / nodes where the LLM responded (excludes nodes where
   the field is not applicable). Use the coverage stats from the analysis above.
   Include ALL 21 fields in the summary, even if no changes were made (show
   "Terms added: none" / "Terms removed: none" in that case). Do not include
   anything else in the summary — no overview, no examples, no commentary.
6. Call finalize_schema to save the final schema and end the session.

## Rules
- Be opinionated about vocabularies: remove unused terms, merge redundant ones
- Don't be too granular — prefer broader, well-populated vocabulary terms
- Maximum 20 terms per controlled vocabulary
- No null-like placeholder values in any vocabulary
- NEVER change the fields array, type_specific_fields, or field metadata
- Process everything and finalize in this session"""


def phase3_refine(
    starting_schema: dict,
    type_summary: str,
    analysis: str,
    client: OpenAI,
    tracker: CostTracker,
    vocab_history: str = "",
) -> None:
    """Run the synchronous agent loop for schema refinement."""
    system = build_refinement_prompt(type_summary, starting_schema, analysis, vocab_history)
    dispatch_tool = make_dispatch_tool(starting_schema)

    messages: list[dict] = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": (
                "Review the population results and modify the schema. "
                "Save checkpoints, write the summary, then finalize."
            ),
        },
    ]

    finalized = False

    for turn in range(1, MAX_TURNS + 1):
        if not tracker.check():
            print(f"\n{'='*60}")
            print(f"BUDGET EXHAUSTED after turn {turn - 1}.")
            print(tracker.summary())
            break

        print(f"\n{'='*60}")
        print(f"Phase 3 — Turn {turn}/{MAX_TURNS}  |  {tracker.summary()}")
        print(f"{'='*60}")

        for attempt in range(6):
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    max_tokens=16384,
                    tools=TOOLS,
                    messages=messages,
                )
                break
            except RateLimitError:
                if attempt == 5:
                    raise
                wait = min(60.0, (2 ** attempt) + random.uniform(0, 1))
                print(f"  [rate limit] retrying in {wait:.1f}s...")
                time.sleep(wait)
        tracker.record(response.usage)

        choice = response.choices[0]
        message = choice.message

        if message.content:
            preview = message.content[:500]
            if len(message.content) > 500:
                preview += "..."
            print(f"\n[Agent] {preview}")

        messages.append(message.model_dump(exclude_none=True))

        tool_calls = message.tool_calls or []
        tool_messages = []

        for tc in tool_calls:
            func_name = tc.function.name
            try:
                func_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                func_args = {}

            print(f"\n[Tool call] {func_name}({json.dumps(func_args, ensure_ascii=False)[:200]}...)")
            result_str = dispatch_tool(func_name, func_args)
            preview = result_str[:300] + "..." if len(result_str) > 300 else result_str
            print(f"[Tool result] {preview}")

            tool_messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_str,
            })

            if func_name == "finalize_schema":
                result_data = json.loads(result_str)
                if result_data.get("finalized") or result_data.get("run_number"):
                    finalized = True

        if finalized:
            messages.extend(tool_messages)
            print(f"\n{'='*60}")
            print("Schema finalized! Agent session complete.")
            print(tracker.summary())
            break

        if tool_messages:
            messages.extend(tool_messages)
        elif choice.finish_reason == "stop":
            messages.append({
                "role": "user",
                "content": (
                    "Don't just describe your plan — execute it now using your "
                    "tools. Save a checkpoint, write the summary, then "
                    "finalize the schema."
                ),
            })

    else:
        print(f"\n{'='*60}")
        print(f"Reached max turns ({MAX_TURNS}).")
        print(tracker.summary())


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    budget: float,
    mode: str = "batch",
    node_type: str | None = None,
    cell_context: bool = False,
) -> None:
    """Run the full 3-phase pipeline.

    Parameters
    ----------
    budget : float
        Maximum USD to spend.
    mode : str
        "batch" for OpenAI Batch API (cheap, slow) or
        "async" for direct async API calls (full price, fast).
    node_type : str or None
        If set, restrict node sampling to this entity type only.
    cell_context : bool
        If True, expand ChemicalSubstance nodes into (drug, cell_type)
        composite nodes before Phase 1. Forces node_type=ChemicalSubstance.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    tracker = CostTracker(budget)

    # Load the latest schema from output/archive/schema_final_N.json
    try:
        starting_schema, latest_n = load_latest_schema()
    except FileNotFoundError:
        print("ERROR: No schema found in output/archive/. Place a schema_final_N.json there first.")
        sys.exit(1)

    next_run = latest_n + 1
    set_run_number(next_run)
    print(f"Starting run {next_run} (based on schema from run {latest_n})")
    print(f"Mode: {mode}")

    # Pre-load graph data
    _ensure_loaded()
    type_dist = get_type_distribution()
    type_lines = "\n".join(
        f"  - {t}: {c:,} nodes" for t, c in type_dist["type_counts"].items()
    )
    type_summary = (
        f"Total nodes: {type_dist['total_nodes']:,}\n"
        f"Entity types ({type_dist['num_types']}):\n{type_lines}"
    )

    # Select 100 diverse nodes (optionally filtered to one entity type)
    effective_type = "ChemicalSubstance" if cell_context else node_type
    diverse_nodes = select_diverse_nodes(node_type=effective_type)
    label = effective_type if effective_type else "all types"
    print(f"\nSelected {len(diverse_nodes)} nodes ({label}):")
    type_counts: dict[str, int] = {}
    for n in diverse_nodes:
        type_counts[n["label"]] = type_counts.get(n["label"], 0) + 1
    for t, c in sorted(type_counts.items()):
        print(f"  {t}: {c}")

    # Expand drug nodes into (drug, cell_type) composite nodes
    if cell_context:
        diverse_nodes = expand_drug_nodes(diverse_nodes)
        print(f"\nCell context expansion: {len(diverse_nodes)} composite nodes "
              f"({NUM_NODES} drugs × {len(DRUG_CELL_TYPES)} cell types)")
        for ct in DRUG_CELL_TYPES:
            print(f"  ::{ct}")
    print()

    if mode == "async":
        p1_results, summaries, p2_results, populated_nodes, suggestions = asyncio.run(
            _run_async_phases(diverse_nodes, starting_schema, api_key)
        )
    else:
        # ---- Phase 1: Summarize via Batch API ----
        print("=" * 60)
        print("PHASE 1: Summarizing entities via Batch API")
        print("=" * 60)
        p1_results, summaries = phase1_summarize(diverse_nodes, client)
        print(tracker.summary())

        # ---- Phase 2: Populate via Batch API ----
        print("\n" + "=" * 60)
        print("PHASE 2: Populating schema fields via Batch API")
        print("=" * 60)
        p2_results, populated_nodes, suggestions = phase2_populate(
            diverse_nodes, summaries, starting_schema, client
        )

    tracker.record_batch(p1_results)
    tracker.record_batch(p2_results)
    print(tracker.summary())

    # ---- Count responses before cleaning (for applicable coverage) ----
    responded = count_responded(populated_nodes, starting_schema)

    # ---- Clean null-like placeholder values from populated nodes ----
    clean_populated_nodes(populated_nodes, starting_schema)
    print("Cleaned null-like placeholder values from populated nodes.")

    # ---- Write populated nodes to archive ----
    print(f"\nWriting {len(populated_nodes)} populated nodes to output/archive/nodes_{next_run}.json...")
    write_nodes(populated_nodes)

    # ---- Idea B: Filter suggestions against previously-rejected terms ----
    rejected = load_rejected_suggestions()
    original_suggestion_count = sum(len(v) for v in suggestions.values())
    suggestions = filter_suggestions(suggestions, rejected)
    filtered_count = sum(len(v) for v in suggestions.values())
    print(f"Suggestions: {original_suggestion_count} raw → {filtered_count} after filtering")

    # ---- Phase 2 analysis ----
    analysis = analyze_population_results(populated_nodes, starting_schema, suggestions, responded)
    print("\n" + analysis)

    # ---- Idea D: Load vocabulary change history for Phase 3 context ----
    vocab_history = load_vocabulary_history(last_n=3)
    if vocab_history:
        print(f"\nLoaded vocabulary change history (last 3 iterations)")

    # ---- Phase 3: Agent refinement (synchronous) ----
    # Note: budget check is NOT applied here — Phase 1+2 batch costs are fixed
    # and Phase 3 costs ~$0.01. The budget gate inside the agent loop handles
    # runaway spending within Phase 3 itself.
    print("\n" + "=" * 60)
    print("PHASE 3: Agent-driven schema refinement")
    print("=" * 60)
    phase3_refine(starting_schema, type_summary, analysis, client, tracker, vocab_history)

    # ---- Idea B: Persist rejected suggestions after Phase 3 ----
    final_schema, _ = load_latest_schema()
    rejected = update_rejected_suggestions(suggestions, final_schema, rejected)
    save_rejected_suggestions(rejected)

    # ---- Cleanup checkpoint intermediates ----
    cleanup_result = cleanup_checkpoints()
    if cleanup_result["count"]:
        print(f"\nCleaned up {cleanup_result['count']} checkpoint file(s).")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(tracker.summary())
    print("=" * 60)


async def _run_async_phases(
    nodes: list[dict],
    schema: dict,
    api_key: str,
) -> tuple[list[dict], dict[str, str], list[dict], list[dict], dict[str, list[str]]]:
    """Run Phase 1 and Phase 2 using direct async API calls."""
    async_client = AsyncOpenAI(api_key=api_key)

    print("=" * 60)
    print("PHASE 1: Summarizing entities via async API")
    print("=" * 60)
    p1_results, summaries = await async_phase1_summarize(nodes, async_client)

    print("\n" + "=" * 60)
    print("PHASE 2: Populating schema fields via async API")
    print("=" * 60)
    p2_results, populated, suggestions = await async_phase2_populate(
        nodes, summaries, schema, async_client
    )

    return p1_results, summaries, p2_results, populated, suggestions


def generate_plots() -> None:
    """Run all plotting scripts after the final iteration."""
    scripts_dir = Path(__file__).resolve().parent
    plot_scripts = [
        scripts_dir / "plot_pca.py",
        scripts_dir / "plot_node_types.py",
        scripts_dir / "plot_term_changes.py",
    ]
    for script in plot_scripts:
        if script.exists():
            print(f"\nRunning {script.name}...")
            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=str(scripts_dir),
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                print(result.stdout.strip())
            else:
                print(f"  WARNING: {script.name} failed:\n{result.stderr.strip()}")
        else:
            print(f"  WARNING: {script.name} not found, skipping.")


def main():
    parser = argparse.ArgumentParser(description="Knowledge Graph Schema Refinement Agent")
    parser.add_argument(
        "--mode",
        choices=["batch", "async"],
        default="batch",
        help="batch = OpenAI Batch API (cheap, slow); async = direct async calls (full price, fast)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of refinement iterations to run (default 10)",
    )
    parser.add_argument(
        "--node-type",
        type=str,
        default=None,
        help=(
            "Restrict node sampling to one entity type "
            "(e.g. Disease, ChemicalSubstance). "
            "Outputs go to output/archive_<node_type>/"
        ),
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=None,
        help=(
            "Number of base nodes to sample per iteration (default 100). "
            "With --cell-context, total composite nodes = num_nodes × cell_types. "
            "Use 9 for Tier 1 async cell-context runs (9 × 11 = 99 nodes)."
        ),
    )
    parser.add_argument(
        "--cell-context",
        action="store_true",
        default=False,
        help=(
            "Expand ChemicalSubstance nodes into (drug, cell_type) composite nodes. "
            "Each drug is annotated separately per cell type for context-specific "
            "drug-disease repositioning. Implies --node-type ChemicalSubstance. "
            "Outputs go to output/archive_<exp-name>/"
        ),
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default="cell_context",
        help=(
            "Name for the cell-context experiment archive (default: cell_context). "
            "Use a different name (e.g. expB) to run a new experiment without "
            "overwriting existing results. Archive: output/archive_<exp-name>/"
        ),
    )
    args = parser.parse_args()
    mode = args.mode
    iterations = args.iterations
    node_type = args.node_type
    cell_context = args.cell_context

    if args.num_nodes is not None:
        global NUM_NODES
        NUM_NODES = args.num_nodes

    # Configure archive directory for this run
    exp_name = args.exp_name if cell_context else None
    if cell_context:
        archive_dir_name = f"archive_{exp_name}"
        archive_path = Path(__file__).resolve().parent.parent / "output" / archive_dir_name
        archive_path.mkdir(parents=True, exist_ok=True)
        set_archive_dir(archive_path)
        print(f"Cell context mode  →  experiment: {exp_name}  →  archive: output/{archive_dir_name}/")
        # Seed archive with a starting schema if none exists yet.
        # If the experiment dir already has schema_final_0.json (e.g. expB), keep it.
        # Otherwise copy the latest schema from the main disease archive.
        if not list(archive_path.glob("schema_final_*.json")):
            import shutil
            main_archive = Path(__file__).resolve().parent.parent / "output" / "archive"
            main_schemas = sorted(
                main_archive.glob("schema_final_*.json"),
                key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
            )
            if main_schemas:
                seed_src = main_schemas[-1]
                seed_dst = archive_path / "schema_final_0.json"
                shutil.copy(seed_src, seed_dst)
                print(f"Seeded {archive_dir_name}/ with {seed_src.name} → schema_final_0.json")
            else:
                print(f"WARNING: No schema found to seed {archive_dir_name}/ from.")
    elif node_type:
        archive_subdir = f"archive_{node_type.lower()}"
        archive_path = Path(__file__).resolve().parent.parent / "output" / archive_subdir
        set_archive_dir(archive_path)
        print(f"Node type filter: {node_type}  →  archive: output/{archive_subdir}/")

    est_per = estimate_per_iteration_cost(mode)
    est_total = round(est_per * iterations, 2)
    print("=" * 60)
    print("Knowledge Graph Schema Refinement Agent")
    print("=" * 60)
    mode_label = "Batch API (50% off)" if mode == "batch" else "Async direct API (standard pricing)"
    print(f"\nMode: {mode_label}")
    print(f"Iterations: {iterations}")
    if cell_context:
        print(f"Cell context mode: {NUM_NODES} drugs × {len(DRUG_CELL_TYPES)} cell types = "
              f"{NUM_NODES * len(DRUG_CELL_TYPES)} composite nodes per iteration")
    elif node_type:
        print(f"Node type: {node_type}")
    print(f"Estimated cost per iteration ({NUM_NODES} nodes): ${est_per:.2f}")
    print(f"Estimated total cost ({iterations} iterations): ${est_total:.2f}")
    print(f"  Phase 1 (summarize): {NUM_NODES} requests per iteration")
    print(f"  Phase 2 (populate):  {NUM_NODES} requests per iteration")
    print(f"  Phase 3 (refine):    synchronous agent loop (~{MAX_TURNS} turns max)")
    print(f"  Model: {MODEL}")
    print()

    budget_input = input(f"Enter per-iteration budget cap in USD (default ${est_per:.2f}): ").strip()
    if budget_input:
        try:
            budget = float(budget_input)
        except ValueError:
            print("Invalid number. Using estimate as budget.")
            budget = est_per
    else:
        budget = est_per

    print(f"\nPer-iteration budget: ${budget:.2f}")
    print(f"Max total spend: ${budget * iterations:.2f}")
    print(f"Starting {iterations} iterations...\n")

    for i in tqdm(range(1, iterations + 1), desc="Iterations", unit=" iter"):
        print("\n" + "#" * 60)
        print(f"# ITERATION {i} / {iterations}")
        print("#" * 60 + "\n")
        run_pipeline(budget, mode=mode, node_type=node_type, cell_context=cell_context)

    # Generate plots after all iterations
    print("\n" + "#" * 60)
    print("# GENERATING PLOTS")
    print("#" * 60)
    generate_plots()

    print("\n" + "#" * 60)
    print(f"# ALL {iterations} ITERATIONS COMPLETE")
    print("#" * 60)


if __name__ == "__main__":
    main()
