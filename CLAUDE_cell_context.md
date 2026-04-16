# Cell-Type Context Expansion for Drug-Disease Repositioning

## Motivation

The standard pipeline annotates each drug node with a single flat context object.
This misses a fundamental biological reality: **the same drug can have completely
different mechanisms depending on the cell type it acts in**.

- Tamoxifen blocks estrogen receptors in breast cancer cells, acts as an agonist in bone cells
- Metformin affects metabolism in hepatocytes differently than in skeletal muscle
- Imatinib inhibits BCR-ABL in leukemia cells, but also acts on c-KIT in mast cells

To enable **context-specific drug-disease repositioning**, each drug must be
represented as a set of cell-type-specific context objects rather than a single
generic annotation.

---

## Concept: (drug, cell_type) composite nodes

A single drug node is expanded into multiple composite nodes — one per relevant
cell type:

```
imatinib  →  imatinib::T_cell
             imatinib::hepatocyte
             imatinib::endothelial
             imatinib::macrophage
             ...
```

Each composite node is independently summarized (Phase 1) and annotated
(Phase 2) with prompts conditioned on the specific cell type. The resulting
annotation captures what the drug does **in that cell**, not an average across
all tissues.

### Composite node structure

```json
{
  "id": "DRUGBANK:DB00001::T_cell",
  "base_id": "DRUGBANK:DB00001",
  "name": "imatinib",
  "label": "ChemicalSubstance",
  "cell_type_context": "T_cell",
  "cell_type": ["T_cell"],
  "mechanism_of_action": ["kinase_inhibition"],
  "pathway_category": ["signaling"],
  "biological_process": ["apoptosis", "cell_cycle"],
  "tissue_location": ["blood", "bone_marrow"]
}
```

---

## Default cell types

Defined in `scripts/tools/cell_context_tools.py` as `DRUG_CELL_TYPES`:

| Cell type | Rationale |
|---|---|
| `neuron` | CNS drugs, neurotoxicity |
| `hepatocyte` | Metabolic drugs, liver metabolism |
| `cardiomyocyte` | Cardiac drugs, cardiotoxicity |
| `epithelial` | Many drugs act on epithelial barriers |
| `T_cell` | Immunology, checkpoint inhibitors |
| `B_cell` | Autoimmune, lymphoma drugs |
| `macrophage` | Anti-inflammatory drugs |
| `endothelial` | Angiogenesis inhibitors, cardiovascular |
| `smooth_muscle` | Vasodilators, antihypertensives |
| `NK_cell` | Cancer immunotherapy |
| `dendritic_cell` | Vaccines, immunomodulators |

---

## How to run

```bash
# Cell-context mode: 100 drugs × 11 cell types = 1,100 composite nodes per iteration
python schema_agent.py --mode async --iterations 5 --cell-context
```

- Implies `--node-type ChemicalSubstance` (no need to specify separately)
- Outputs go to `output/archive_cell_context/`
- Schema and nodes files follow the same naming convention: `schema_final_N.json`, `nodes_N.json`

---

## Pipeline changes

### Phase 1 — Cell-type-conditional summarization

The standard prompt:
```
Provide a concise biological/chemical summary of "imatinib" (entity type: ChemicalSubstance).
```

Is replaced with:
```
Provide a concise biological/chemical summary of "imatinib" (entity type: ChemicalSubstance)
specifically in the context of T_cell cells.

Focus ONLY on what is known about this drug's activity in this cell type...
```

### Phase 2 — Cell-type-conditional population

The population prompt is similarly conditioned:
```
Drug: "imatinib" (ID: DRUGBANK:DB00001) specifically in the context of T_cell cells.
Return a JSON object reflecting the drug's activity SPECIFICALLY in T_cell cells.
```

### Phase 3 — Schema refinement

Phase 3 runs unchanged — the agent reviews coverage statistics and adjusts
controlled vocabularies. Cell-context runs may surface new terms relevant to
cell-type-specific drug biology (e.g. more granular `mechanism_of_action` terms).

---

## Files

| File | Role |
|---|---|
| `scripts/tools/cell_context_tools.py` | Node expansion + conditional prompts |
| `scripts/tools/async_tools.py` | Uses cell-context prompts when `cell_type_context` present |
| `scripts/schema_agent.py` | `--cell-context` CLI flag, expansion step in `run_pipeline()` |
| `output/archive_cell_context/` | All outputs from cell-context runs |

---

## Drug-disease repositioning pipeline

Once both disease nodes (from `output/archive_disease/`) and cell-context drug
nodes (from `output/archive_cell_context/`) are annotated:

### Step 1 — Embed both sets

```bash
# Disease nodes (already done)
python embed_disease_nodes.py --method umap --color-by disease_association

# Drug nodes (future: embed_drug_nodes.py)
python embed_drug_nodes.py --method umap --color-by cell_type_context
```

### Step 2 — Compute similarity with cell-type filter

```python
# For each disease node, find drugs annotated in the same cell type
# that share vocabulary terms in bridging fields:
#   mechanism_of_action, pathway_category, biological_process

for disease in disease_nodes:
    cell_type = disease["cell_type"][0]
    candidate_drugs = [d for d in drug_nodes if d["cell_type_context"] == cell_type]
    scores = cosine_similarity(disease_embedding, candidate_drug_embeddings)
    # → ranked repositioning candidates in the same cellular context
```

### Step 3 — Filter and validate

- Remove known approved drug-disease pairs (not repositioning candidates)
- Rank by shared vocabulary terms in bridging fields
- Validate top candidates against known repositioning successes

---

## Cost estimate (cell-context run)

| Component | Nodes | Cost estimate |
|---|---|---|
| Phase 1 (async, 1,100 nodes/iter) | 100 drugs × 11 cell types | ~$0.04/iter |
| Phase 2 (async, 1,100 nodes/iter) | 100 drugs × 11 cell types | ~$0.20/iter |
| Phase 3 (agent loop) | synchronous | ~$0.04/iter |
| **5 iterations total** | | **~$1.40** |

Cell-context runs cost ~11× more per iteration than standard runs (more nodes),
but produce far richer drug representations.
