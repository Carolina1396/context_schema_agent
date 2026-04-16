"""
Schema management tools: loading, checkpointing, and finalizing.

All schema artifacts live in output/archive/ with run-number suffixes.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

_DEFAULT_ARCHIVE_DIR = Path(__file__).resolve().parent.parent.parent / "output" / "archive"
_ARCHIVE_DIR = _DEFAULT_ARCHIVE_DIR
_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

_current_run_number: int | None = None


def set_archive_dir(path: Path | str) -> None:
    """Override the archive directory (e.g. for node-type-specific runs)."""
    global _ARCHIVE_DIR
    _ARCHIVE_DIR = Path(path)
    _ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

MAX_VOCAB_SIZE = 20
NULL_LIKE_TERMS = frozenset({
    "not_applicable", "unknown", "none", "not_specified",
    "none_known", "unclassified", "other", "n/a", "na",
    "not_a_drug", "not_organism_specific",
})


def is_null_like(term: str) -> bool:
    """Check if a vocabulary term is a null-like placeholder."""
    if term is None:
        return True
    return term.lower().strip() in NULL_LIKE_TERMS


def _clean_vocabularies(schema: dict) -> list[str]:
    """Remove null-like terms and enforce 20-term cap on all vocabularies.

    Modifies schema in-place. Returns a list of warning strings.
    """
    warnings = []
    vocabs = schema.get("controlled_vocabularies", {})
    for vocab_name, terms in vocabs.items():
        original_len = len(terms)
        # Remove null-like terms
        cleaned = [t for t in terms if not is_null_like(t)]
        removed_nulls = sorted(set(terms) - set(cleaned))
        if removed_nulls:
            warnings.append(
                f"{vocab_name}: stripped null-like terms: {removed_nulls}"
            )
        # Enforce cap
        if len(cleaned) > MAX_VOCAB_SIZE:
            overflow = cleaned[MAX_VOCAB_SIZE:]
            warnings.append(
                f"{vocab_name}: truncated from {len(cleaned)} to {MAX_VOCAB_SIZE} "
                f"(dropped: {[t for t in overflow]})"
            )
            cleaned = cleaned[:MAX_VOCAB_SIZE]
        vocabs[vocab_name] = cleaned
    return warnings


def set_run_number(n: int) -> None:
    """Set the run number for the current pipeline execution."""
    global _current_run_number
    _current_run_number = n


def load_latest_schema() -> tuple[dict, int]:
    """Load the schema_final_N.json with the highest N from the archive.

    Returns
    -------
    (schema_dict, N) where N is the run number of the loaded schema.
    Raises FileNotFoundError if no schema exists in the archive.
    """
    existing = list(_ARCHIVE_DIR.glob("schema_final_*.json"))
    nums: list[tuple[int, Path]] = []
    for p in existing:
        stem = p.stem  # e.g. schema_final_3
        parts = stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            nums.append((int(parts[1]), p))

    if not nums:
        print("[schema] ERROR: No schema_final_N.json found in output/archive/")
        raise FileNotFoundError("No schema found in output/archive/")

    nums.sort(key=lambda x: x[0])
    latest_n, latest_path = nums[-1]
    schema = json.loads(latest_path.read_text(encoding="utf-8"))
    print(f"[schema] Loaded schema from {latest_path} (run {latest_n})")
    return schema, latest_n


def save_schema(schema: dict, version: str) -> dict:
    """Save a schema draft to disk as a versioned checkpoint."""
    _ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    warnings = _clean_vocabularies(schema)
    for w in warnings:
        print(f"[checkpoint] WARNING: {w}")
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"schema_checkpoint_v{version}_{ts}.json"
    path = _ARCHIVE_DIR / filename
    path.write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[checkpoint] Saved schema version {version} → {path}")
    result = {"saved": True, "path": str(path), "version": version}
    if warnings:
        result["warnings"] = warnings
    return result


def finalize_schema(schema: dict) -> dict:
    """Save the final schema as schema_final_N.json in the archive."""
    if _current_run_number is None:
        return {"error": "Run number not set. Call set_run_number() first."}
    _ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    warnings = _clean_vocabularies(schema)
    for w in warnings:
        print(f"[finalize] WARNING: {w}")
    path = _ARCHIVE_DIR / f"schema_final_{_current_run_number}.json"
    path.write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[finalize] Saved final schema → {path}")
    result = {"finalized": True, "path": str(path), "run_number": _current_run_number}
    if warnings:
        result["warnings"] = warnings
    return result


def write_summary(content: str) -> dict:
    """Write the refinement summary to output/archive/refinement_summary_N.md."""
    if _current_run_number is None:
        return {"error": "Run number not set. Call set_run_number() first."}
    _ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    path = _ARCHIVE_DIR / f"refinement_summary_{_current_run_number}.md"
    path.write_text(content, encoding="utf-8")
    print(f"[summary] Saved refinement summary → {path}")
    return {"saved": True, "path": str(path), "run_number": _current_run_number}


def write_nodes(nodes: list[dict]) -> dict:
    """Write populated node context objects to output/archive/nodes_N.json."""
    if _current_run_number is None:
        return {"error": "Run number not set. Call set_run_number() first."}
    _ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    path = _ARCHIVE_DIR / f"nodes_{_current_run_number}.json"
    path.write_text(
        json.dumps(nodes, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[nodes] Saved {len(nodes)} populated nodes → {path}")
    return {"saved": True, "path": str(path), "count": len(nodes)}


def cleanup_checkpoints() -> dict:
    """Remove schema checkpoint intermediates from the archive directory."""
    removed = []
    for p in _ARCHIVE_DIR.glob("schema_checkpoint_*.json"):
        p.unlink()
        removed.append(p.name)
        print(f"[cleanup] Removed checkpoint {p.name}")
    return {"removed": removed, "count": len(removed)}


# ---------------------------------------------------------------------------
# Idea B: Rejected suggestions — cross-iteration memory
# ---------------------------------------------------------------------------

_REJECTED_FILE = "rejected_suggestions.json"


def load_rejected_suggestions() -> dict[str, list[str]]:
    """Load the rejected-suggestions ledger from the archive.

    Returns a dict of field_name -> [rejected term, ...].
    """
    path = _ARCHIVE_DIR / _REJECTED_FILE
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        print(f"[rejected] Loaded {sum(len(v) for v in data.values())} rejected terms from {path}")
        return data
    return {}


def save_rejected_suggestions(rejected: dict[str, list[str]]) -> None:
    """Persist the rejected-suggestions ledger to the archive."""
    path = _ARCHIVE_DIR / _REJECTED_FILE
    path.write_text(json.dumps(rejected, indent=2, ensure_ascii=False), encoding="utf-8")
    total = sum(len(v) for v in rejected.values())
    print(f"[rejected] Saved {total} rejected terms → {path}")


def update_rejected_suggestions(
    suggestions: dict[str, list[str]],
    final_schema: dict,
    existing_rejected: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Determine which Phase 2 suggestions the agent rejected, merge into ledger.

    A suggestion is "rejected" if it was presented to Phase 3 but does NOT
    appear in the finalized schema's controlled vocabularies.
    """
    vocabs = final_schema.get("controlled_vocabularies", {})
    merged = {k: list(v) for k, v in existing_rejected.items()}  # deep copy

    newly_rejected = 0
    for field, terms in suggestions.items():
        accepted = set(vocabs.get(field, []))
        rejected_field = set(merged.get(field, []))
        for term in terms:
            if term not in accepted and term not in rejected_field:
                merged.setdefault(field, []).append(term)
                newly_rejected += 1

    print(f"[rejected] {newly_rejected} new rejected terms this iteration")
    return merged


def filter_suggestions(
    suggestions: dict[str, list[str]],
    rejected: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Remove previously-rejected terms from Phase 2 suggestions."""
    filtered: dict[str, list[str]] = {}
    removed_count = 0
    for field, terms in suggestions.items():
        rejected_set = set(rejected.get(field, []))
        kept = [t for t in terms if t not in rejected_set]
        removed_count += len(terms) - len(kept)
        if kept:
            filtered[field] = kept
    if removed_count:
        print(f"[rejected] Filtered out {removed_count} previously-rejected suggestions")
    return filtered


# ---------------------------------------------------------------------------
# Idea D: Vocabulary change history — cross-iteration awareness
# ---------------------------------------------------------------------------


def load_vocabulary_history(last_n: int = 3) -> str:
    """Load recent refinement summaries and build a compressed change history.

    Parses refinement_summary_N.md files to extract per-field term adds/removes,
    then formats a compact history string for the Phase 3 prompt.

    Parameters
    ----------
    last_n : int
        Number of most recent refinement summaries to include (default 3).

    Returns
    -------
    str
        A multi-line history block, or empty string if no history is available.
    """
    import re

    summaries = sorted(
        _ARCHIVE_DIR.glob("refinement_summary_*.md"),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
    )
    if not summaries:
        return ""

    # Take the last N
    summaries = summaries[-last_n:]

    # Parse each summary: extract field_name, terms added, terms removed
    # Expected format per field:
    #   - **field_name** — coverage: XX% | applicable coverage: YY%
    #     - Terms added: term1, term2, ...
    #     - Terms removed: term1, term2, ...
    field_re = re.compile(r"^\s*-\s+\*\*(\w+)\*\*")
    added_re = re.compile(r"Terms added:\s*(.+)", re.IGNORECASE)
    removed_re = re.compile(r"Terms removed:\s*(.+)", re.IGNORECASE)

    # field -> [(iteration_num, added_list, removed_list), ...]
    history: dict[str, list[tuple[int, list[str], list[str]]]] = {}

    for summary_path in summaries:
        iter_num = int(summary_path.stem.rsplit("_", 1)[-1])
        text = summary_path.read_text(encoding="utf-8")
        lines = text.split("\n")

        current_field = None
        added: list[str] = []
        removed: list[str] = []

        for line in lines:
            fm = field_re.match(line)
            if fm:
                # Save previous field
                if current_field:
                    history.setdefault(current_field, []).append(
                        (iter_num, added, removed)
                    )
                current_field = fm.group(1)
                added = []
                removed = []
                continue

            am = added_re.search(line)
            if am and current_field:
                raw = am.group(1).strip()
                if raw.lower() != "none":
                    added = [t.strip() for t in raw.split(",") if t.strip()]
                continue

            rm = removed_re.search(line)
            if rm and current_field:
                raw = rm.group(1).strip()
                if raw.lower() != "none":
                    removed = [t.strip() for t in raw.split(",") if t.strip()]
                continue

        # Save last field
        if current_field:
            history.setdefault(current_field, []).append(
                (iter_num, added, removed)
            )

    if not history:
        return ""

    # Build compressed output — only include fields that had changes
    out_lines = [
        f"## Vocabulary change history (last {len(summaries)} iterations)",
        "Fields with no changes in this window are omitted.\n",
    ]

    for field in sorted(history.keys()):
        entries = history[field]
        # Skip fields with zero changes across all iterations
        if all(not a and not r for _, a, r in entries):
            continue
        parts = []
        for iter_num, added, removed in entries:
            changes = []
            if added:
                changes.append("+" + ", +".join(added))
            if removed:
                changes.append("-" + ", -".join(removed))
            if changes:
                parts.append(f"iter {iter_num}: {'; '.join(changes)}")
            else:
                parts.append(f"iter {iter_num}: (no change)")
        out_lines.append(f"  - **{field}**: {' → '.join(parts)}")

    return "\n".join(out_lines)
