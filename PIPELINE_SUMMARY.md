# Knowledge Graph Schema Discovery Agent — Pipeline Summary

## Problem

We have a biomedical knowledge graph with **250,000 nodes** spanning 9 entity
types (Disease, ChemicalSubstance, MacromolecularMachine, GeneFamily, Pathway,
BiologicalProcessOrActivity, PhenotypicFeature, AnatomicalEntity, OrganismTaxon).
Each node has an ID, name, and label — but no structured biological context.

We need to annotate every node with rich, multi-dimensional biological context
using a controlled-vocabulary schema, so that nodes can be compared, embedded,
and clustered by biological meaning rather than just graph topology.

**The challenge:** we don't know in advance what vocabulary terms will adequately
cover 250k diverse biological entities. A hand-curated vocabulary would be
incomplete and biased.

---

## Approach: Agentic Schema Refinement

Instead of manually curating vocabularies, we built an **agentic pipeline** that
iteratively discovers and refines controlled vocabularies by attempting to
annotate real nodes and learning from what doesn't fit.

### Core loop (per iteration)

```
┌─────────────────────────────────────────────────────────────┐
│  1. Load latest schema (schema_final_K.json)                │
│  2. Sample 100 diverse nodes from the knowledge graph       │
│  3. Phase 1 — Summarize: LLM produces a biological summary │
│     of each entity (~1,000 tokens per node)                 │
│  4. Phase 2 — Populate: LLM maps each summary to schema    │
│     fields using controlled vocabularies (multi-label)      │
│  5. Phase 3 — Refine: Agent reviews coverage statistics,    │
│     adds/removes/merges vocabulary terms (max 20 per field) │
│  6. Output: schema_final_(K+1).json, nodes_(K+1).json,     │
│     refinement_summary_(K+1).md                             │
└─────────────────────────────────────────────────────────────┘
         ↑                                          │
         └──────────── next iteration ──────────────┘
```

Each iteration's refined schema feeds into the next. Over multiple iterations
(typically 5–12), vocabularies converge: early iterations expand rapidly to
cover gaps, later iterations consolidate and prune.

### Key constraints
- **Schema fields are fixed** (21 biological context fields) — only vocabulary
  terms are modified
- **Max 20 terms** per controlled vocabulary — forces merging and prioritization
- **No placeholder values** ("unknown", "N/A", etc.) — if nothing fits, the
  value is `null`
- All LLM calls use **gpt-4o-mini** via OpenAI async API
- Each field produces a **list** of labels (multi-label classification)

---

## Experiments Run

We ran four experiments, each targeting different aspects of the knowledge graph:

### Experiment A — General schema (all entity types)

| Detail | Value |
|---|---|
| **Archive** | `output/archive/` |
| **Schema** | 21 fields + 2 identity fields |
| **Iterations** | 7 (schema versions 2–8) |
| **Nodes per iteration** | 100 (all 9 entity types) |
| **Purpose** | Establish baseline schema covering the full diversity of the knowledge graph |

The general-purpose schema covers everything from organisms and tissues to
molecular functions and disease associations. Vocabularies matured over 7
iterations, expanding from ~10 terms per field to 10–20. This schema was the
starting point for all subsequent experiments.

### Experiment B — Disease-focused refinement

| Detail | Value |
|---|---|
| **Archive** | `output/archive_disease/` |
| **Schema** | Same 21 fields, disease-tuned vocabularies |
| **Iterations** | 12 (schema versions 1–12) |
| **Nodes per iteration** | 100 Disease nodes only |
| **Purpose** | Specialize vocabularies for disease annotation |

By running 12 iterations on Disease nodes exclusively, vocabularies shifted
toward clinically relevant terms. Notable changes:
- `mechanism_of_action` gained disease-specific terms: `loss_of_function`,
  `gain_of_function`, `toxic_accumulation`, `immune_evasion`
- `pathway_category` shifted from generic to named pathways: `PI3K_AKT`,
  `MAPK_ERK`, `Wnt_beta_catenin`, `NF_kB`, `JAK_STAT`, `mTOR`
- `disease_association` expanded to 20 organ-system-level categories
- `phenotype_category` shifted to clinical symptoms: `pain`, `seizure`,
  `organ_failure`, `cognitive_impairment`

Embeddings of annotated disease nodes were generated and visualized via PCA and
UMAP, colored by `disease_association`, `biological_system`, and
`tissue_location`.

### Experiment C — Cell-type-specific drug behavior

| Detail | Value |
|---|---|
| **Archive** | `output/archive_expC/` |
| **Schema** | 5 custom behavior fields (no identity fields except id/name) |
| **Iterations** | 9 (schema versions 1–9) |
| **Nodes per iteration** | 100 drugs x 11 cell types = ~1,100 composite nodes |
| **Purpose** | Capture how the same drug behaves differently in different cell types |

This experiment tested a fundamentally different approach. Instead of the 21
general-purpose fields, we designed **5 cell-behavior fields** specifically to
encode what is biologically distinctive about each cell type:

| Field | Example terms |
|---|---|
| `cell_function_engaged` | contractility, detoxification, immune_surveillance |
| `cell_metabolic_context` | glycolysis, lipogenesis, oxidative_phosphorylation |
| `cell_stress_response` | excitotoxicity, oxidative_stress, ischemia_reperfusion |
| `receptor_class_in_cell` | adrenergic, nuclear_hormone, ion_channel_gated |
| `drug_cell_interaction_mode` | direct_target, metabolized_by_cell, bystander_effect |

Each drug was expanded into 11 composite nodes (one per cell type: neuron,
hepatocyte, cardiomyocyte, epithelial, T_cell, B_cell, macrophage, endothelial,
smooth_muscle, NK_cell, dendritic_cell). LLM prompts were conditioned on the
specific cell type context.

**Key result:** UMAP embeddings of behavior-only features (with per-drug mean
subtraction to remove drug identity) produced **clear cell-type clusters** —
hepatocytes, neurons, cardiomyocytes, and immune cell subtypes separated
cleanly. This validates that the 5 behavioral fields encode enough cell-type
signal for downstream drug repositioning without explicit label leakage.

Schema evolution: early iterations added 13–26 terms/iteration; by iteration 9,
the agent was pruning (16 removed) as vocabularies hit the 20-term cap.

### Experiment D — General schema (mixed entity types), stricter refinement

| Detail | Value |
|---|---|
| **Archive** | `output/archive_expD/` |
| **Schema** | Same 21 fields as Experiment A |
| **Iterations** | 3 (schema versions 1–3) |
| **Nodes per iteration** | 100 (all entity types) |
| **Purpose** | Re-run general schema with stricter vocabulary control |

This experiment tracked rejected vocabulary suggestions in
`rejected_suggestions.json`, capturing terms the agent considered but chose not
to add (e.g., overly specific drug classes like "sclerosant", "fluorescent_probe",
"endocrine_disruptor"). Coverage by iteration 3: `organism` 90%, `cell_type`
94%, `chemical_classification` 92%, while less applicable fields like
`inheritance_pattern` and `taxonomic_domain` remained at 0% for mixed-type
samples.

---

## Pipeline Architecture

```
scripts/
  schema_agent.py          Main pipeline: orchestrates all 3 phases
  tools/
    graph_tools.py          Node sampling, type/predicate distributions
    async_tools.py          Async OpenAI API calls (Phase 1 & 2)
    batch_tools.py          OpenAI Batch API alternative (Phase 1 & 2)
    schema_tools.py         Schema I/O, checkpointing, finalization
    cell_context_tools.py   Composite node expansion + cell-conditioned prompts
  plot_pca.py               PCA/UMAP embedding visualization
  plot_node_types.py        Entity type distribution per iteration
  color_scheme.py           Colorblind-friendly palette
```

### Data flow

```
db/nodes.csv (250k nodes)
    │
    ▼
Phase 1: LLM summarization (gpt-4o-mini, ~1000 tokens/node)
    │
    ▼
Phase 2: Schema population (gpt-4o-mini, multi-label classification)
    │
    ▼
Phase 3: Agent refinement loop (analyze coverage → adjust vocabularies)
    │
    ▼
output/archive*/schema_final_N.json   — refined schema
output/archive*/nodes_N.json          — annotated nodes
output/archive*/refinement_summary_N.md — per-field changelog
```

---

## Schema Structure

Each schema contains:
- **`fields`**: array of field definitions (name, description, type,
  controlled_vocabulary reference, applies_to_types)
- **`controlled_vocabularies`**: dict of vocabulary name → term list (8–20
  terms, max 20)
- **`type_specific_fields`**: dict mapping entity type → relevant field names
- **`notes`**: agent observations

Each annotated node is a JSON object where every controlled-vocabulary field is
either a **list of matching terms** or **null**:

```json
{
  "id": "DOID:0001816",
  "name": "angiosarcoma",
  "organism": ["Homo sapiens"],
  "tissue_location": ["blood", "soft_tissue"],
  "biological_process": ["angiogenesis", "cell_proliferation"],
  "mechanism_of_action": ["aberrant_signaling"],
  "disease_association": ["cancer"],
  "inheritance_pattern": null
}
```

---

## Visualizations

| Plot | Location | Description |
|---|---|---|
| Term changes by iteration | `images/term_changes_by_iteration.png`, `output/archive_expC/term_changes_by_iteration.png` | Grouped barplot showing vocabulary terms added/removed per iteration |
| Node types by iteration | `images/node_types_by_iteration.png` | Stacked barplot of entity type distribution per sample |
| Disease PCA | `images/disease_embed_pca_disease_association.png` | PCA of disease nodes colored by disease category |
| Disease UMAP | `images/disease_embed_umap_disease_association.png` | UMAP of disease nodes colored by disease category |
| Cell-context UMAP | `output/archive_expC/disease_embed_umap_cell_type_context_behavior_noname_min3_meansub.png` | UMAP of drug composite nodes — behavior-only features, mean-subtracted — showing cell-type clusters |

---

## Key Findings

1. **Iterative refinement works.** Vocabularies that start with 10 generic terms
   converge to 15–20 domain-specific terms after 5–12 iterations, with early
   rapid expansion followed by consolidation.

2. **Domain-specific schemas outperform general ones.** The disease-focused
   schema (Exp B) produced much more clinically relevant vocabularies than the
   general schema — named pathways instead of generic categories, clinical
   symptoms instead of abstract phenotype classes.

3. **Cell-type behavior is encodable without labels.** The 5-field behavior
   schema (Exp C) captures enough cell-type-specific biology to reconstruct cell
   identity from drug behavior alone, validated by UMAP clustering.

4. **The 20-term cap forces meaningful curation.** Rather than unbounded
   vocabulary growth, the cap forces the agent to merge redundant terms and drop
   low-value ones, producing tighter vocabularies.

5. **Coverage varies by field applicability.** Universal fields like `organism`
   and `biological_system` reach 90%+ coverage; type-specific fields like
   `inheritance_pattern` are 0% for non-Disease samples (correctly so).

---

## Next Steps

- **Scale to 250k nodes**: Use OpenAI Batch API (50% off pricing) to annotate
  all nodes using the finalized schemas
- **Drug-disease repositioning**: Embed both disease nodes (Exp B) and
  cell-context drug nodes (Exp C) into a shared vector space, then compute
  cell-type-filtered similarity for repositioning candidates
- **Cross-experiment schema merging**: Combine the specialized disease and
  cell-context schemas into a unified annotation framework
