"""
Embed annotated nodes via OpenAI text-embedding-3-small, then project with PCA.

Works with both disease nodes and drug cell-context composite nodes.
Automatically detects node type and adjusts plot titles and filenames accordingly
(disease_embed_* vs drug_embed_*).

Workflow:
  1. Load all nodes_N.json from the archive directory (deduplicated by ID).
  2. Serialize each node's populated vocab fields into a structured text string.
  3. Call OpenAI embeddings API (text-embedding-3-small, 1536-dim).
  4. Cache embeddings to <archive>/embeddings.npy + embeddings_meta.json
     so re-running skips the API call.
  5. Run PCA (and optionally UMAP / t-SNE).
  6. Save plots to output directory.

Usage:
    python embed_disease_nodes.py                   # PCA
    python embed_disease_nodes.py --method umap
    python embed_disease_nodes.py --method all
    python embed_disease_nodes.py --reembed         # force new API call
    python embed_disease_nodes.py --color-by tissue_location
    python embed_disease_nodes.py --color-by biological_system
"""

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import seaborn as sns
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.decomposition import PCA

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ARCHIVE = _PROJECT_ROOT / "output" / "archive_disease"  # overridden by --archive-dir
_IMAGES = _PROJECT_ROOT / "images"
_IMAGES.mkdir(exist_ok=True)

load_dotenv(_PROJECT_ROOT / ".env")

EMBED_MODEL = "text-embedding-3-small"   # 1536 dims, $0.02/1M tokens
# Cache paths are set dynamically in main() after --archive-dir is resolved
EMBED_CACHE = _ARCHIVE / "embeddings.npy"
META_CACHE  = _ARCHIVE / "embeddings_meta.json"

# ── Palettes ─────────────────────────────────────────────────────────────────

# Default palette for disease_association field
DISEASE_PALETTE = {
    "cancer":           "#D62728",
    "neurodegenerative":"#1F77B4",
    "cardiovascular":   "#FF7F0E",
    "autoimmune":       "#2CA02C",
    "metabolic":        "#9467BD",
    "infectious":       "#8C564B",
    "rare_genetic":     "#E377C2",
    "psychiatric":      "#7F7F7F",
    "inflammatory":     "#17BECF",
    "developmental":    "#BCBD22",
    "hematological":    "#AEC7E8",
    "endocrine":        "#FFBB78",
    "renal":            "#98DF8A",
    "pulmonary":        "#FF9896",
    "hepatic":          "#C5B0D5",
    "musculoskeletal":  "#C49C94",
    "dermatological":   "#F7B6D2",
    "reproductive":     "#C7C7C7",
    "ophthalmological": "#DBDB8D",
    "aging":            "#9EDAE5",
    "other":            "#AAAAAA",
}


def build_palette(nodes: list[dict], field: str) -> dict[str, str]:
    """Return a color dict for any vocab field. Uses DISEASE_PALETTE for disease_association."""
    if field == "disease_association":
        return DISEASE_PALETTE
    terms = sorted({primary_category(n, field) for n in nodes})
    cmap = plt.colormaps["tab20"].resampled(max(len(terms), 1))
    return {term: mcolors.to_hex(cmap(i)) for i, term in enumerate(terms)}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all_nodes() -> list[dict]:
    """Stack all nodes_N.json, keeping the latest record per node ID."""
    files = sorted(
        glob.glob(str(_ARCHIVE / "nodes_*.json")),
        key=lambda f: int(re.search(r"(\d+)", Path(f).stem).group(1)),
    )
    if not files:
        print(f"ERROR: No nodes_*.json found in {_ARCHIVE}")
        sys.exit(1)
    seen: dict[str, dict] = {}
    for fp in files:
        for node in json.loads(Path(fp).read_text()):
            seen[node["id"]] = node
    nodes = list(seen.values())
    print(f"Loaded {len(nodes)} unique disease nodes from {len(files)} file(s)")
    return nodes


def filter_by_coverage(
    nodes: list[dict],
    min_fields: int,
    schema_fields: list[str] | None = None,
) -> list[dict]:
    """Keep only nodes with at least min_fields populated.

    Parameters
    ----------
    min_fields : minimum number of non-null fields required.
    schema_fields : field names to check. If None, uses all FIELD_LABELS keys.
    """
    check_fields = schema_fields or list(FIELD_LABELS.keys())
    filtered = [
        n for n in nodes
        if sum(1 for f in check_fields if n.get(f)) >= min_fields
    ]
    print(f"Coverage filter (>= {min_fields} fields): {len(filtered)}/{len(nodes)} nodes kept")
    return filtered


# ── Text serialisation ────────────────────────────────────────────────────────

FIELD_LABELS = {
    # Original 21 schema fields
    "organism":              "Organism",
    "tissue_location":       "Tissue",
    "cell_type":             "Cell type",
    "cellular_compartment":  "Compartment",
    "biological_system":     "System",
    "biological_scale":      "Scale",
    "biological_process":    "Process",
    "molecular_function":    "Molecular function",
    "pathway_category":      "Pathway",
    "mechanism_of_action":   "Mechanism",
    "disease_association":   "Disease type",
    "clinical_relevance":    "Clinical relevance",
    "phenotype_category":    "Phenotype",
    "chemical_classification":"Chemical class",
    "drug_class":            "Drug class",
    "regulatory_role":       "Regulatory role",
    "interaction_type":      "Interaction",
    "inheritance_pattern":   "Inheritance",
    "developmental_stage":   "Developmental stage",
    "taxonomic_domain":      "Taxonomy",
    "expression_context":    "Expression",
    # Experiment B: cell-type-specific drug context fields
    "cell_effect":           "Cell effect",
    "target_in_cell":        "Target in cell",
    "pathway_in_cell":       "Pathway in cell",
    "cell_vulnerability":    "Cell vulnerability",
    "therapeutic_role_in_cell": "Therapeutic role",
    # Experiment C: cell-type-characteristic fields
    "cell_function_engaged":     "Cell function",
    "cell_metabolic_context":    "Cell metabolism",
    "cell_stress_response":      "Cell stress",
    "receptor_class_in_cell":    "Receptor class",
    "drug_cell_interaction_mode": "Interaction mode",
}


# Fields that describe cell-type identity (the label itself, not the biology)
_CONTEXT_IDENTITY_FIELDS = frozenset({
    "organism",
    "tissue_location",
    "cell_type",
    "cellular_compartment",
    "biological_system",
})


def node_to_text(
    node: dict,
    exclude_fields: frozenset[str] | None = None,
    include_name: bool = True,
) -> str:
    """Serialize a populated node into a structured text string for embedding.

    Parameters
    ----------
    exclude_fields : set of field names to omit from the text.
        Pass _CONTEXT_IDENTITY_FIELDS to embed only drug-behavior fields.
    include_name : if False, omit the Drug/Disease name line entirely.
    """
    lines = []
    if include_name:
        lines.append(f"Drug: {node.get('name', node['id'])}")
    for field, label in FIELD_LABELS.items():
        if exclude_fields and field in exclude_fields:
            continue
        vals = node.get(field)
        if vals and isinstance(vals, list):
            lines.append(f"{label}: {', '.join(vals)}")
    return "\n".join(lines)


# ── Embeddings ────────────────────────────────────────────────────────────────

def get_embeddings(
    nodes: list[dict],
    client: OpenAI,
    force: bool = False,
    exclude_fields: frozenset[str] | None = None,
    include_name: bool = True,
) -> np.ndarray:
    """Return (N, 1536) embedding matrix. Loads from cache unless force=True."""
    parts = []
    if exclude_fields:
        parts.append("behavior_only")
    if not include_name:
        parts.append("noname")
    cache_key = "_".join(parts) if parts else "full"
    embed_cache = EMBED_CACHE.with_stem(f"{EMBED_CACHE.stem}_{cache_key}")
    meta_cache  = META_CACHE.with_stem(f"{META_CACHE.stem}_{cache_key}")

    if not force and embed_cache.exists() and meta_cache.exists():
        cached_meta = json.loads(meta_cache.read_text())
        current_ids = [n["id"] for n in nodes]
        if cached_meta.get("node_ids") == current_ids:
            print(f"Loading cached embeddings from {embed_cache.name}")
            return np.load(embed_cache)
        else:
            print("Node set changed — re-embedding...")

    texts = [
        node_to_text(n, exclude_fields=exclude_fields, include_name=include_name) or "unknown"
        for n in nodes
    ]

    # Show a sample text so user can verify the format
    print("\n── Sample embedding input ──")
    print(texts[0])
    print("───────────────────────────\n")

    # OpenAI allows up to 2048 texts per call; batch if needed
    all_embeddings = []
    batch_size = 500
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        print(f"Embedding batch {start // batch_size + 1} ({len(batch)} texts)...")
        response = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vecs = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
        all_embeddings.extend(vecs)
        total_tokens = response.usage.total_tokens
        cost = total_tokens * 0.02 / 1_000_000
        print(f"  tokens: {total_tokens:,}  cost: ${cost:.5f}")

    matrix = np.array(all_embeddings, dtype=np.float32)
    np.save(embed_cache, matrix)
    meta_cache.write_text(json.dumps({
        "node_ids":   [n["id"] for n in nodes],
        "node_names": [n.get("name", n["id"]) for n in nodes],
        "model":      EMBED_MODEL,
        "dims":       matrix.shape[1],
        "cache_key":  cache_key,
    }, indent=2))
    print(f"Saved embeddings → {embed_cache.name}  {matrix.shape}")
    return matrix


# ── Mean subtraction ─────────────────────────────────────────────────────────

def subtract_drug_mean(matrix: np.ndarray, nodes: list[dict]) -> np.ndarray:
    """Remove per-drug mean embedding so only cell-type variation remains.

    For each drug (identified by base_id or the prefix before '::' in the id),
    compute the mean of its cell-type variant embeddings and subtract it.
    Nodes that have no variants (singletons) are left unchanged.
    """
    residuals = matrix.copy()
    # Group row indices by base drug id
    from collections import defaultdict
    groups: dict[str, list[int]] = defaultdict(list)
    for i, node in enumerate(nodes):
        base = node.get("base_id") or node["id"].rsplit("::", 1)[0]
        groups[base].append(i)

    n_subtracted = 0
    for base_id, indices in groups.items():
        if len(indices) < 2:
            continue  # singleton — nothing to subtract
        mean_vec = matrix[indices].mean(axis=0)
        for idx in indices:
            residuals[idx] = matrix[idx] - mean_vec
        n_subtracted += 1

    print(f"[mean-subtract] Removed drug-identity component from {n_subtracted} drugs "
          f"({len(matrix)} total embeddings)")
    return residuals


# ── Dimensionality reduction ──────────────────────────────────────────────────

def run_pca(matrix: np.ndarray, n: int = 2):
    pca = PCA(n_components=n, random_state=42)
    return pca.fit_transform(matrix), pca


def run_umap(matrix: np.ndarray):
    try:
        from umap import UMAP
    except ImportError:
        raise ImportError("umap-learn not installed. Run: conda install -c conda-forge umap-learn")
    r = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    return r.fit_transform(matrix), None


def run_tsne(matrix: np.ndarray):
    from sklearn.manifold import TSNE
    r = TSNE(n_components=2, random_state=42, perplexity=min(30, len(matrix) - 1))
    return r.fit_transform(matrix), None


# ── Plotting ──────────────────────────────────────────────────────────────────

def primary_category(node: dict, field: str) -> str:
    vals = node.get(field)
    if not vals:
        return "other"
    if isinstance(vals, str):
        return vals
    if isinstance(vals, list):
        return vals[0]
    return "other"


def plot_embedding(
    coords: np.ndarray,
    nodes: list[dict],
    method: str,
    color_by: str = "disease_association",
    palette: dict = None,
    reducer=None,
    ax: plt.Axes = None,
) -> None:
    if palette is None:
        palette = DISEASE_PALETTE
    categories = [primary_category(n, color_by) for n in nodes]
    colors     = [palette.get(c, "#AAAAAA") for c in categories]
    names      = [n.get("name", n["id"]) for n in nodes]

    ax.scatter(coords[:, 0], coords[:, 1],
               c=colors, s=60, alpha=0.82, linewidths=0.3, edgecolors="white")

    # Label the 5 most extreme nodes in PC1 direction
    order = np.argsort(coords[:, 0])
    for idx in list(order[:3]) + list(order[-3:]):
        ax.annotate(names[idx], (coords[idx, 0], coords[idx, 1]),
                    fontsize=6.5, alpha=0.75,
                    xytext=(4, 0), textcoords="offset points")

    if method == "pca" and reducer is not None:
        ax.set_xlabel(f"PC1 ({reducer.explained_variance_ratio_[0]:.1%} var)")
        ax.set_ylabel(f"PC2 ({reducer.explained_variance_ratio_[1]:.1%} var)")
    else:
        ax.set_xlabel(f"{method.upper()} 1")
        ax.set_ylabel(f"{method.upper()} 2")

    is_drug = any(n.get("cell_type_context") for n in nodes)
    embed_label = "Drug" if is_drug else "Disease"
    ax.set_title(f"{embed_label} embeddings — {method.upper()}  (n={len(nodes)}, {EMBED_MODEL})")

    seen = sorted(set(categories))
    patches = [mpatches.Patch(color=palette.get(c, "#AAAAAA"), label=c) for c in seen]
    ax.legend(handles=patches, loc="upper left", bbox_to_anchor=(1.01, 1),
              fontsize=7, title=color_by, title_fontsize=8, borderaxespad=0)


def plot_pca_variance(pca: PCA, ax: plt.Axes, n_show: int = 20) -> None:
    """Scree plot of explained variance."""
    ratios = pca.explained_variance_ratio_[:n_show] * 100
    ax.bar(range(1, len(ratios) + 1), ratios, color="#1F77B4", alpha=0.8)
    ax.plot(range(1, len(ratios) + 1), np.cumsum(ratios), "o-", color="#D62728",
            markersize=4, label="Cumulative")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance (%)")
    ax.set_title("Scree plot")
    ax.legend(fontsize=8)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["pca", "umap", "tsne", "all"], default="pca")
    parser.add_argument("--reembed", action="store_true",
                        help="Force new API call even if cache exists")
    parser.add_argument("--color-by", dest="color_by", default="disease_association",
                        help="Vocabulary field to use for point colors (default: disease_association)")
    parser.add_argument("--archive-dir", dest="archive_dir", default=None,
                        help="Path to archive directory (default: output/archive_disease). "
                             "Use output/archive_cell_context for drug cell-context runs.")
    parser.add_argument("--subtract-drug-mean", dest="subtract_drug_mean",
                        action="store_true",
                        help="Remove per-drug mean embedding before projection, "
                             "revealing cell-type-specific variation.")
    parser.add_argument("--behavior-only", dest="behavior_only",
                        action="store_true",
                        help="Exclude cell-type identity fields (cell_type, tissue_location, "
                             "biological_system, cellular_compartment, organism) from the "
                             "embedding text, keeping only drug-behavior fields.")
    parser.add_argument("--no-name", dest="no_name", action="store_true",
                        help="Exclude the drug/disease name from the embedding text.")
    parser.add_argument("--output-dir", dest="output_dir", default=None,
                        help="Directory to save plot images (default: images/). "
                             "Use the archive dir to keep results self-contained.")
    parser.add_argument("--min-fields", dest="min_fields", type=int, default=None,
                        help="Only embed nodes with at least this many fields populated. "
                             "Use to filter out sparse nodes before UMAP.")
    args = parser.parse_args()

    global _ARCHIVE, EMBED_CACHE, META_CACHE, _IMAGES
    if args.archive_dir:
        _ARCHIVE = Path(args.archive_dir)
        if not _ARCHIVE.is_absolute():
            _ARCHIVE = _PROJECT_ROOT / _ARCHIVE
        print(f"Archive: {_ARCHIVE}")
    EMBED_CACHE = _ARCHIVE / "embeddings.npy"
    META_CACHE  = _ARCHIVE / "embeddings_meta.json"

    if args.output_dir:
        _IMAGES = Path(args.output_dir)
        if not _IMAGES.is_absolute():
            _IMAGES = _PROJECT_ROOT / _IMAGES
        _IMAGES.mkdir(parents=True, exist_ok=True)
        print(f"Output dir: {_IMAGES}")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set in .env")
        sys.exit(1)
    client = OpenAI(api_key=api_key)

    exclude_fields = _CONTEXT_IDENTITY_FIELDS if args.behavior_only else None

    include_name = not args.no_name

    nodes = load_all_nodes()
    if args.min_fields is not None:
        nodes = filter_by_coverage(nodes, args.min_fields)
        if len(nodes) < 10:
            print(f"ERROR: Only {len(nodes)} nodes passed the coverage filter. Lower --min-fields.")
            sys.exit(1)

    embeddings = get_embeddings(nodes, client, force=args.reembed,
                                exclude_fields=exclude_fields, include_name=include_name)
    print(f"Embedding matrix: {embeddings.shape}")
    print(f"Coloring by: {args.color_by}")
    if args.behavior_only:
        print(f"Behavior-only mode: excluded {sorted(_CONTEXT_IDENTITY_FIELDS)}")
    if args.no_name:
        print("Name excluded from embedding text.")

    if args.subtract_drug_mean:
        embeddings = subtract_drug_mean(embeddings, nodes)

    is_drug = any(n.get("cell_type_context") for n in nodes)
    file_prefix = "drug_embed" if is_drug else "disease_embed"

    suffix = f"_{args.color_by}"
    if args.behavior_only:
        suffix += "_behavior"
    if args.no_name:
        suffix += "_noname"
    if args.min_fields is not None:
        suffix += f"_min{args.min_fields}"
    if args.subtract_drug_mean:
        suffix += "_meansub"

    palette = build_palette(nodes, args.color_by)
    methods = ["pca", "umap", "tsne"] if args.method == "all" else [args.method]

    for method in methods:
        print(f"\nRunning {method.upper()}...")
        try:
            if method == "pca":
                n_components = min(20, embeddings.shape[0], embeddings.shape[1])
                coords, reducer = run_pca(embeddings, n=n_components)
                sns.set_theme(style="whitegrid")
                fig, (ax_scatter, ax_scree) = plt.subplots(
                    1, 2, figsize=(18, 7),
                    gridspec_kw={"width_ratios": [2.5, 1]},
                )
                plot_embedding(coords[:, :2], nodes, method, args.color_by, palette, reducer, ax_scatter)
                plot_pca_variance(reducer, ax_scree)
            else:
                run_fn = run_umap if method == "umap" else run_tsne
                coords, reducer = run_fn(embeddings)
                sns.set_theme(style="whitegrid")
                fig, ax_scatter = plt.subplots(figsize=(11, 8))
                plot_embedding(coords, nodes, method, args.color_by, palette, reducer, ax_scatter)

            plt.tight_layout()
            out = _IMAGES / f"{file_prefix}_{method}{suffix}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            print(f"Saved → {out}")
            plt.close(fig)
        except ImportError as e:
            print(f"  Skipping {method.upper()}: {e}")


if __name__ == "__main__":
    main()
