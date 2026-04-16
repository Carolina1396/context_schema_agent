"""
Grouped barplot: total terms added vs removed per iteration.

Parses refinement_summary_N.md files from output/archive/ and plots
a grouped bar chart.

X-axis = iteration (N), grouped by added/removed.
Y-axis = total count of vocabulary terms changed.
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from color_scheme import ENTITY_TYPE_PALETTE

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent
_ARCHIVE_DIR = _ROOT / "output" / "archive"
_IMAGES_DIR = _ROOT / "images"
_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Colors from palette: purple for added, orange for removed
COLOR_ADDED = ENTITY_TYPE_PALETTE["AnatomicalEntity"]        # deep purple
COLOR_REMOVED = ENTITY_TYPE_PALETTE["OrganismTaxon"]         # burnt orange

# ---------------------------------------------------------------------------
# Parse refinement summaries
# ---------------------------------------------------------------------------

_ADDED_RE = re.compile(r"Count of terms added:\s*(\d+)")
_REMOVED_RE = re.compile(r"Count of terms removed:\s*(\d+)")


def _parse_summary(path: Path) -> dict[str, int]:
    """Parse a refinement summary into {added: N, removed: N} totals."""
    total_added = 0
    total_removed = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        m = _ADDED_RE.search(line)
        if m:
            total_added += int(m.group(1))
            continue
        m = _REMOVED_RE.search(line)
        if m:
            total_removed += int(m.group(1))
    return {"added": total_added, "removed": total_removed}


def _load_summaries() -> list[tuple[int, dict[str, int]]]:
    """Load all refinement summaries, return sorted by run number."""
    summaries = []
    for p in _ARCHIVE_DIR.glob("refinement_summary_*.md"):
        stem = p.stem
        parts = stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            run_num = int(parts[1])
            parsed = _parse_summary(p)
            summaries.append((run_num, parsed))
    summaries.sort(key=lambda x: x[0])
    return summaries


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def main():
    global _ARCHIVE_DIR, _IMAGES_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-dir", dest="archive_dir", default=None,
                        help="Archive directory to read summaries from (default: output/archive/)")
    parser.add_argument("--output-dir", dest="output_dir", default=None,
                        help="Directory to save the plot (default: images/)")
    args = parser.parse_args()

    if args.archive_dir:
        _ARCHIVE_DIR = Path(args.archive_dir)
        if not _ARCHIVE_DIR.is_absolute():
            _ARCHIVE_DIR = _ROOT / _ARCHIVE_DIR
    if args.output_dir:
        _IMAGES_DIR = Path(args.output_dir)
        if not _IMAGES_DIR.is_absolute():
            _IMAGES_DIR = _ROOT / _IMAGES_DIR
        _IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    summaries = _load_summaries()
    if not summaries:
        print(f"No refinement_summary_N.md files found in {_ARCHIVE_DIR}.")
        return

    run_numbers = [r for r, _ in summaries]
    added = np.array([d["added"] for _, d in summaries])
    removed = np.array([d["removed"] for _, d in summaries])

    x = np.arange(len(run_numbers))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(max(5, len(run_numbers) * 1.8), 5))

    ax.bar(
        x - bar_width / 2, added, bar_width,
        label="Added",
        color=COLOR_ADDED,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.bar(
        x + bar_width / 2, removed, bar_width,
        label="Removed",
        color=COLOR_REMOVED,
        edgecolor="white",
        linewidth=0.5,
    )

    # Add count labels on top of bars
    for i in range(len(run_numbers)):
        if added[i] > 0:
            ax.text(x[i] - bar_width / 2, added[i] + 0.5, str(added[i]),
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
        if removed[i] > 0:
            ax.text(x[i] + bar_width / 2, removed[i] + 0.5, str(removed[i]),
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in run_numbers])
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Count of Terms")
    ax.set_title("Vocabulary Terms Added / Removed per Iteration")
    ax.legend(frameon=False)

    fig.tight_layout()
    out_path = _IMAGES_DIR / "term_changes_by_iteration.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
