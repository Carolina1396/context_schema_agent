"""
Disease-node context-vector clustering.

Loads all nodes_N.json from output/archive_disease/, stacks them into a
single binary feature matrix (field::term columns), and projects into 2D
using PCA.  A second panel shows the top vocabulary features (loadings)
that drive each principal component.

Optionally falls back from UMAP → t-SNE → PCA depending on what is installed.

Usage:
    python plot_disease_pca.py                  # PCA (default)
    python plot_disease_pca.py --method umap
    python plot_disease_pca.py --method tsne
    python plot_disease_pca.py --method all     # side-by-side comparison
"""

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.decomposition import PCA

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ARCHIVE = _PROJECT_ROOT / "output" / "archive_disease"
_IMAGES = _PROJECT_ROOT / "images"
_IMAGES.mkdir(exist_ok=True)

# ── Disease-association color palette ────────────────────────────────────────

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


# ── Helpers ──────────────────────────────────────────────────────────────────

def _latest(pattern: str) -> Path | None:
    files = glob.glob(str(_ARCHIVE / pattern))
    if not files:
        return None
    return Path(max(files, key=lambda f: int(re.search(r"(\d+)", f).group(1))))


def load_schema() -> dict:
    path = _latest("schema_final_*.json")
    if not path:
        raise FileNotFoundError(f"No schema_final_*.json in {_ARCHIVE}")
    print(f"Schema:  {path.name}")
    return json.loads(path.read_text())


def load_all_nodes() -> list[dict]:
    """Stack all nodes_N.json files, deduplicating by node ID (keep latest)."""
    files = sorted(
        glob.glob(str(_ARCHIVE / "nodes_*.json")),
        key=lambda f: int(re.search(r"(\d+)", f.split("nodes_")[-1]).group(1)),
    )
    if not files:
        raise FileNotFoundError(f"No nodes_*.json in {_ARCHIVE}")
    seen: dict[str, dict] = {}
    for fp in files:
        for node in json.loads(Path(fp).read_text()):
            seen[node["id"]] = node   # later iteration overwrites earlier
    nodes = list(seen.values())
    print(f"Nodes:   {len(nodes)} unique disease nodes across {len(files)} iteration(s)")
    return nodes


def build_feature_matrix(nodes: list[dict], schema: dict) -> tuple[np.ndarray, list[str]]:
    """Binary (node × feature) matrix where features are 'field::term' pairs."""
    controlled_fields = [
        (f["name"], schema["controlled_vocabularies"][f["controlled_vocabulary"]])
        for f in schema["fields"]
        if f["field_type"] == "controlled"
    ]
    columns = [f"{fn}::{t}" for fn, terms in controlled_fields for t in terms]
    matrix = np.zeros((len(nodes), len(columns)), dtype=np.float32)
    for i, node in enumerate(nodes):
        col = 0
        for fn, terms in controlled_fields:
            vals = node.get(fn) or []
            for j, term in enumerate(terms):
                if term in vals:
                    matrix[i, col + j] = 1.0
            col += len(terms)
    return matrix, columns


def primary_disease_category(node: dict) -> str:
    """Return the first disease_association label, or 'other'."""
    vals = node.get("disease_association")
    if vals and isinstance(vals, list):
        return vals[0]
    return "other"


def color_for(cat: str) -> str:
    return DISEASE_PALETTE.get(cat, DISEASE_PALETTE["other"])


# ── Dimensionality reduction ──────────────────────────────────────────────────

def run_pca(matrix: np.ndarray, n: int = 2):
    pca = PCA(n_components=n, random_state=42)
    coords = pca.fit_transform(matrix)
    return coords, pca


def run_umap(matrix: np.ndarray):
    try:
        from umap import UMAP
    except ImportError:
        raise ImportError("umap-learn not installed. Run: conda install -c conda-forge umap-learn")
    reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    return reducer.fit_transform(matrix), None


def run_tsne(matrix: np.ndarray):
    from sklearn.manifold import TSNE
    reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(matrix) - 1))
    return reducer.fit_transform(matrix), None


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_projection(
    coords: np.ndarray,
    nodes: list[dict],
    pca_obj,
    method: str,
    columns: list[str],
    ax_main: plt.Axes,
    ax_load: plt.Axes | None,
) -> None:
    categories = [primary_disease_category(n) for n in nodes]
    colors = [color_for(c) for c in categories]
    names = [n.get("name", n["id"]) for n in nodes]

    # ── scatter ──
    ax_main.scatter(
        coords[:, 0], coords[:, 1],
        c=colors, s=55, alpha=0.80, linewidths=0.3, edgecolors="white",
    )

    # Annotate a few extreme points
    df_tmp = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "name": names})
    for _, row in df_tmp.nlargest(3, "x").iterrows():
        ax_main.annotate(row["name"], (row["x"], row["y"]),
                         fontsize=6, alpha=0.7, ha="left",
                         xytext=(3, 0), textcoords="offset points")
    for _, row in df_tmp.nsmallest(3, "x").iterrows():
        ax_main.annotate(row["name"], (row["x"], row["y"]),
                         fontsize=6, alpha=0.7, ha="right",
                         xytext=(-3, 0), textcoords="offset points")

    if method == "pca" and pca_obj is not None:
        ax_main.set_xlabel(f"PC1 ({pca_obj.explained_variance_ratio_[0]:.1%} var)")
        ax_main.set_ylabel(f"PC2 ({pca_obj.explained_variance_ratio_[1]:.1%} var)")
    else:
        ax_main.set_xlabel(f"{method.upper()} 1")
        ax_main.set_ylabel(f"{method.upper()} 2")
    ax_main.set_title(f"Disease nodes — {method.upper()} of context vectors (n={len(nodes)})")

    # Legend
    seen_cats = sorted(set(categories))
    patches = [mpatches.Patch(color=color_for(c), label=c) for c in seen_cats]
    ax_main.legend(handles=patches, loc="upper left", bbox_to_anchor=(1.01, 1),
                   fontsize=7, title="disease_association", title_fontsize=8,
                   borderaxespad=0)

    # ── loadings panel (PCA only) ──
    if ax_load is not None and method == "pca" and pca_obj is not None:
        n_show = 15
        loadings = pca_obj.components_  # shape (2, n_features)
        # Top features by combined absolute loading across PC1 + PC2
        importance = np.abs(loadings[0]) + np.abs(loadings[1])
        top_idx = np.argsort(importance)[-n_show:][::-1]
        top_features = [columns[i].replace("::", "\n", 1) for i in top_idx]
        pc1_vals = loadings[0][top_idx]
        pc2_vals = loadings[1][top_idx]

        y = np.arange(n_show)
        ax_load.barh(y + 0.2, pc1_vals, height=0.35, label="PC1", color="#1F77B4", alpha=0.8)
        ax_load.barh(y - 0.2, pc2_vals, height=0.35, label="PC2", color="#FF7F0E", alpha=0.8)
        ax_load.set_yticks(y)
        ax_load.set_yticklabels(top_features, fontsize=7)
        ax_load.axvline(0, color="black", linewidth=0.5)
        ax_load.set_xlabel("Loading")
        ax_load.set_title(f"Top {n_show} vocabulary features")
        ax_load.legend(fontsize=8)


def make_figure(method: str, nodes: list[dict], matrix: np.ndarray,
                schema: dict, columns: list[str]) -> plt.Figure:
    if method == "pca":
        coords, pca_obj = run_pca(matrix)
        fig, (ax_main, ax_load) = plt.subplots(
            1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [2, 1]}
        )
    elif method == "umap":
        coords, pca_obj = run_umap(matrix)
        fig, ax_main = plt.subplots(figsize=(10, 8))
        ax_load = None
    elif method == "tsne":
        coords, pca_obj = run_tsne(matrix)
        fig, ax_main = plt.subplots(figsize=(10, 8))
        ax_load = None
    else:
        raise ValueError(f"Unknown method: {method}")

    sns.set_theme(style="whitegrid")
    plot_projection(coords, nodes, pca_obj, method, columns, ax_main, ax_load)
    plt.tight_layout()
    return fig


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", choices=["pca", "umap", "tsne", "all"], default="pca",
        help="Dimensionality reduction method (default: pca)",
    )
    args = parser.parse_args()

    schema = load_schema()
    nodes = load_all_nodes()
    matrix, columns = build_feature_matrix(nodes, schema)
    print(f"Feature matrix: {matrix.shape[0]} nodes × {matrix.shape[1]} features")
    print(f"Non-zero: {matrix.sum():.0f} ({matrix.mean()*100:.1f}% density)")

    methods = ["pca", "umap", "tsne"] if args.method == "all" else [args.method]

    for method in methods:
        print(f"\nRunning {method.upper()}...")
        try:
            fig = make_figure(method, nodes, matrix, schema, columns)
            out = _IMAGES / f"disease_pca_{method}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            print(f"Saved → {out}")
            plt.close(fig)
        except ImportError as e:
            print(f"  Skipping {method}: {e}")


if __name__ == "__main__":
    main()
