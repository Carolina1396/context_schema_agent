"""
Async direct-API tools for Phase 1 (summarization) and Phase 2 (population).

Drop-in alternative to batch_tools.py: same prompts, same output formats,
but uses AsyncOpenAI with concurrent requests instead of the Batch API.
"""

import asyncio
import json
import random
from pathlib import Path

from openai import AsyncOpenAI, RateLimitError, InternalServerError, APIConnectionError, APITimeoutError
from tqdm.asyncio import tqdm as async_tqdm

from tools.batch_tools import (
    MODEL,
    _SUMMARIZE_PROMPT_TEMPLATE,
    _BATCH_INPUTS_DIR,
    _BATCH_OUTPUTS_DIR,
)
from tools.cell_context_tools import (
    build_cell_context_summarize_prompt,
    build_cell_context_populate_prompt,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_CONCURRENT = 5   # semaphore limit — keep low to stay under TPM limits
MAX_RETRIES = 8      # max retries on transient errors

# Rate limit errors get a longer base wait to let the TPM window reset
_RATE_LIMIT_BASE_WAIT = 15.0   # seconds — minimum wait after a 429
_OTHER_ERROR_BASE_WAIT = 2.0   # seconds — minimum wait for 5xx / connection errors

# Chunked gathering — prevents thundering herd by processing tasks in batches
# with a mandatory sleep between chunks instead of firing all at once
CHUNK_SIZE = 20          # requests per chunk
CHUNK_DELAY_P1 = 8.0    # seconds between Phase 1 chunks (~22k tokens/chunk at 1,100 tok/req)
CHUNK_DELAY_P2 = 15.0   # seconds between Phase 2 chunks (~36k tokens/chunk at 1,800 tok/req)

# Standard pricing (no batch discount)
INPUT_COST_PER_M = 0.15
OUTPUT_COST_PER_M = 0.60


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------


_TRANSIENT_ERRORS = (RateLimitError, InternalServerError, APIConnectionError, APITimeoutError)


async def _with_backoff(coro_fn, *args, **kwargs):
    """Call an async function with exponential backoff on transient API errors.

    RateLimitError uses a longer base wait (_RATE_LIMIT_BASE_WAIT) to allow the
    TPM window to reset.  Other transient errors use a shorter base wait.
    """
    for attempt in range(MAX_RETRIES):
        try:
            return await coro_fn(*args, **kwargs)
        except _TRANSIENT_ERRORS as e:
            if attempt == MAX_RETRIES - 1:
                raise
            base = _RATE_LIMIT_BASE_WAIT if isinstance(e, RateLimitError) else _OTHER_ERROR_BASE_WAIT
            wait = min(120.0, base * (2 ** attempt) + random.uniform(0, 2))
            from tqdm import tqdm
            tqdm.write(f"  [{type(e).__name__}] attempt {attempt + 1}/{MAX_RETRIES}, retrying in {wait:.1f}s...")
            await asyncio.sleep(wait)


# ---------------------------------------------------------------------------
# Chunked gather helper
# ---------------------------------------------------------------------------


async def _chunked_gather(tasks: list, chunk_delay: float) -> list:
    """Process async tasks in chunks with a delay between each chunk.

    Prevents thundering-herd rate-limit errors by spreading token consumption
    across time instead of firing all tasks at once via gather(*all_tasks).

    Uses return_exceptions=True so one bad node (e.g. malformed name causing
    a 400 BadRequestError) doesn't crash the entire chunk.
    """
    from tqdm import tqdm as sync_tqdm
    results = []
    n = len(tasks)
    with sync_tqdm(total=n, unit=" nodes") as pbar:
        for i in range(0, n, CHUNK_SIZE):
            chunk = tasks[i : i + CHUNK_SIZE]
            chunk_results = await asyncio.gather(*chunk, return_exceptions=True)
            for r in chunk_results:
                if isinstance(r, Exception):
                    sync_tqdm.write(f"  [skipped] {type(r).__name__}: {str(r)[:120]}")
                    results.append(None)
                else:
                    results.append(r)
            pbar.update(len(chunk))
            if i + CHUNK_SIZE < n:
                pbar.set_postfix_str(f"waiting {chunk_delay:.0f}s")
                await asyncio.sleep(chunk_delay)
                pbar.set_postfix_str("")
    return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# Phase 1: Async summarization
# ---------------------------------------------------------------------------


async def _summarize_one(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    node: dict,
    custom_id: str,
) -> dict:
    """Send a single summarization request with retry. Returns a result dict matching batch output format."""
    if node.get("cell_type_context"):
        prompt = build_cell_context_summarize_prompt(node)
    else:
        prompt = _SUMMARIZE_PROMPT_TEMPLATE.format(
            entity_name=node["name"],
            entity_type=node["label"],
        )

    async def _call():
        async with sem:
            return await client.chat.completions.create(
                model=MODEL,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}],
            )

    response = await _with_backoff(_call)
    choice = response.choices[0]
    return {
        "custom_id": custom_id,
        "response": {
            "body": {
                "choices": [{"message": {"content": choice.message.content}}],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                },
            },
        },
    }


async def async_phase1_summarize(
    nodes: list[dict],
    client: AsyncOpenAI,
) -> tuple[list[dict], dict[str, str]]:
    """Run Phase 1 summarization with concurrent async requests.

    Returns (raw_results, summaries_by_custom_id) — same shape as batch mode.
    """
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    tasks = []
    for i, node in enumerate(nodes):
        cid = f"{i:04d}_{node['id']}"
        tasks.append(_summarize_one(client, sem, node, cid))

    print(f"[async] Phase 1: sending {len(tasks)} summarization requests "
          f"(concurrency={MAX_CONCURRENT}, chunk_size={CHUNK_SIZE}, delay={CHUNK_DELAY_P1}s)...")
    results = await _chunked_gather(tasks, CHUNK_DELAY_P1)

    # Parse into summaries dict (same as parse_phase1_results)
    summaries: dict[str, str] = {}
    for item in results:
        cid = item["custom_id"]
        choices = item["response"]["body"].get("choices", [])
        summaries[cid] = choices[0]["message"]["content"] if choices else ""

    success = sum(1 for s in summaries.values() if s)
    print(f"[async] Phase 1 complete: {success}/{len(nodes)} summaries obtained.")
    return list(results), summaries


# ---------------------------------------------------------------------------
# Phase 2: Async population
# ---------------------------------------------------------------------------


def _build_populate_prompt(node: dict, summary_text: str, schema: dict) -> str:
    """Build the populate prompt — mirrors batch_tools.build_populate_request."""
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

    return (
        f"You are mapping biological context to schema fields.\n\n"
        f'Entity: "{node["name"]}" (ID: {node["id"]}, type: {node["label"]})\n\n'
        f"Summary:\n{summary_text}\n\n"
        f"Schema fields and their allowed controlled-vocabulary terms:\n{fields_block}\n\n"
        f"For EACH field listed above, return a JSON object where:\n"
        f"- Each key is a field name (use EXACTLY the field names listed above — do not add new fields)\n"
        f"- Each value is a LIST of matching vocabulary terms (the entity may match multiple)\n"
        f"- Use null if the field does not apply or cannot be determined\n"
        f'- Do NOT use placeholder values like "not_applicable", "unknown", "none", '
        f'"not_specified", "none_known", or similar — use null instead\n'
        f"- Only use terms from the provided vocabulary lists\n"
        f'- If a concept fits but no existing term matches, use the CLOSEST existing term '
        f'and also add a "suggested_additions" key mapping field names to suggested new vocabulary terms\n\n'
        f"Return valid JSON only. Example:\n"
        f'{{"organism": ["Homo sapiens"], "tissue_location": ["blood", "liver"], '
        f'"cell_type": null, "suggested_additions": {{"tissue_location": ["bone_marrow_stroma"]}}}}'
    )


async def _populate_one(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    node: dict,
    summary_text: str,
    schema: dict,
    custom_id: str,
) -> dict:
    """Send a single population request with retry. Returns a result dict matching batch output format."""
    if node.get("cell_type_context"):
        prompt = build_cell_context_populate_prompt(node, summary_text, schema)
    else:
        prompt = _build_populate_prompt(node, summary_text, schema)

    async def _call():
        async with sem:
            return await client.chat.completions.create(
                model=MODEL,
                max_tokens=2000,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
            )

    response = await _with_backoff(_call)
    choice = response.choices[0]
    return {
        "custom_id": custom_id,
        "response": {
            "body": {
                "choices": [{"message": {"content": choice.message.content}}],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                },
            },
        },
    }


async def async_phase2_populate(
    nodes: list[dict],
    summaries: dict[str, str],
    schema: dict,
    client: AsyncOpenAI,
) -> tuple[list[dict], list[dict], dict[str, list[str]]]:
    """Run Phase 2 population with concurrent async requests.

    Returns (raw_results, populated_nodes, suggestions) — same shape as batch mode.
    """
    from tools.batch_tools import parse_phase2_results

    sem = asyncio.Semaphore(MAX_CONCURRENT)
    tasks = []
    for i, node in enumerate(nodes):
        cid = f"{i:04d}_{node['id']}"
        summary_text = summaries.get(cid, "")
        if not summary_text:
            continue
        tasks.append(_populate_one(client, sem, node, summary_text, schema, cid))

    print(f"[async] Phase 2: sending {len(tasks)} population requests "
          f"(concurrency={MAX_CONCURRENT}, chunk_size={CHUNK_SIZE}, delay={CHUNK_DELAY_P2}s)...")
    results = await _chunked_gather(tasks, CHUNK_DELAY_P2)

    populated, suggestions = parse_phase2_results(list(results))

    # Restore identity fields using _custom_id set by parse_phase2_results
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

    print(f"[async] Phase 2 complete: {len(populated)} nodes populated.")
    return list(results), populated, suggestions


# ---------------------------------------------------------------------------
# Cost estimation (standard pricing, no batch discount)
# ---------------------------------------------------------------------------


def estimate_async_cost(num_nodes: int) -> float:
    """Estimate total cost for Phase 1 + Phase 2 at standard (non-batch) pricing."""
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
