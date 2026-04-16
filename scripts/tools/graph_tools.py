"""
Graph inspection and node sampling tools.

Loads nodes.csv once on import and provides tools for the agent to
understand graph composition and sample diverse nodes.
"""

import csv
import random
from pathlib import Path
from collections import Counter

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_NODES_PATH = Path(__file__).resolve().parent.parent.parent / "db" / "nodes.csv"
_nodes: list[dict] = []
_nodes_by_id: dict[str, dict] = {}
_nodes_by_type: dict[str, list[dict]] = {}
_loaded = False


def _ensure_loaded() -> None:
    global _loaded
    if _loaded:
        return
    with open(_NODES_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Loading nodes", unit=" nodes"):
            _nodes.append(row)
            _nodes_by_id[row["id"]] = row
            label = row.get("label", "unknown")
            _nodes_by_type.setdefault(label, []).append(row)
    _loaded = True


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def get_type_distribution() -> dict:
    """Return a count of nodes grouped by their label (entity type)."""
    _ensure_loaded()
    counts = Counter(n["label"] for n in _nodes)
    sorted_counts = dict(counts.most_common())
    return {
        "total_nodes": len(_nodes),
        "num_types": len(sorted_counts),
        "type_counts": sorted_counts,
    }


def get_predicate_distribution() -> dict:
    """Analyse cross-reference prefixes to understand relationship types.

    Since the dataset contains only a node table (no explicit edges), we
    infer relationship connectivity from the *xrefs* column.  Each xref
    prefix (e.g. MESH, UMLS, SNOMED) represents a link to an external
    ontology, which is the closest proxy we have for edge predicates.
    """
    _ensure_loaded()
    prefix_counter: Counter[str] = Counter()
    nodes_with_xrefs = 0
    total_xrefs = 0

    for node in _nodes:
        xrefs = node.get("xrefs", "")
        if not xrefs:
            continue
        nodes_with_xrefs += 1
        for xref in xrefs.split("|"):
            xref = xref.strip()
            if not xref:
                continue
            total_xrefs += 1
            prefix = xref.split(":")[0] if ":" in xref else xref
            prefix_counter[prefix] += 1

    return {
        "note": (
            "No explicit edge table found. Distribution below reflects "
            "cross-reference (xref) prefixes, which indicate external "
            "ontology links per node."
        ),
        "nodes_with_xrefs": nodes_with_xrefs,
        "total_xref_links": total_xrefs,
        "xref_prefix_counts": dict(prefix_counter.most_common(30)),
    }


def sample_nodes(
    node_type: str | None = None,
    count: int = 10,
    strategy: str = "random",
) -> dict:
    """Sample nodes from the graph.

    Parameters
    ----------
    node_type : str or None
        If provided, only sample from this entity type.
    count : int
        Number of nodes to return (capped at 50).
    strategy : str
        One of "random", "high_degree", "low_degree".
        Degree is approximated by the number of xref entries.
    """
    _ensure_loaded()
    count = min(count, 50)

    pool = _nodes_by_type.get(node_type, []) if node_type else _nodes
    if not pool:
        return {"error": f"No nodes found for type '{node_type}'"}

    def _degree(n: dict) -> int:
        xrefs = n.get("xrefs", "")
        return len(xrefs.split("|")) if xrefs else 0

    if strategy == "random":
        sampled = random.sample(pool, min(count, len(pool)))
    elif strategy == "high_degree":
        sampled = sorted(pool, key=_degree, reverse=True)[:count]
    elif strategy == "low_degree":
        sampled = sorted(pool, key=_degree)[:count]
    else:
        return {"error": f"Unknown strategy '{strategy}'. Use random|high_degree|low_degree."}

    results = []
    for n in sampled:
        results.append({
            "id": n["id"],
            "name": n["name"],
            "label": n["label"],
            "xrefs": n.get("xrefs", ""),
            "synonyms": n.get("synonyms", ""),
            "alt_ids": n.get("alt_ids", ""),
            "subsets": n.get("subsets", ""),
            "xref_count": _degree(n),
        })

    return {
        "count": len(results),
        "strategy": strategy,
        "node_type_filter": node_type,
        "nodes": results,
    }


def get_node_by_id(node_id: str) -> dict:
    """Fetch a single node by its ID."""
    _ensure_loaded()
    node = _nodes_by_id.get(node_id)
    if node is None:
        return {"error": f"Node '{node_id}' not found."}
    xrefs = node.get("xrefs", "")
    return {
        "id": node["id"],
        "name": node["name"],
        "label": node["label"],
        "xrefs": xrefs,
        "synonyms": node.get("synonyms", ""),
        "alt_ids": node.get("alt_ids", ""),
        "subsets": node.get("subsets", ""),
        "xref_count": len(xrefs.split("|")) if xrefs else 0,
    }
