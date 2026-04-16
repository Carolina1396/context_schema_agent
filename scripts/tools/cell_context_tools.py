"""
Cell-type context expansion for drug (ChemicalSubstance) nodes.

Transforms a flat drug node into multiple (drug, cell_type) composite nodes —
one per relevant cellular context. This enables cell-type-specific drug
annotations for drug-disease repositioning.

Each composite node carries:
    id               : "{base_id}::{cell_type}"   (e.g. "DRUGBANK:DB00001::T_cell")
    base_id          : original drug id
    cell_type_context: the specific cell type being annotated
    name, label, ...  : all other fields carried from the base node

Usage
-----
    from tools.cell_context_tools import expand_drug_nodes, DRUG_CELL_TYPES

    drug_nodes = select_diverse_nodes(node_type="ChemicalSubstance")
    composite_nodes = expand_drug_nodes(drug_nodes)
    # → len(composite_nodes) == len(drug_nodes) * len(DRUG_CELL_TYPES)
"""

import json

# ---------------------------------------------------------------------------
# Default cell types for drug expansion
# (subset of the schema's cell_type vocabulary most relevant to pharmacology)
# ---------------------------------------------------------------------------

# Fields relevant to ChemicalSubstance annotation — sent in Phase 2 prompts.
# Excludes fields that are not meaningful for drugs (inheritance_pattern,
# taxonomic_domain, phenotype_category, biological_scale, developmental_stage,
# expression_context) to reduce Phase 2 prompt size by ~40%.
DRUG_RELEVANT_FIELDS: frozenset[str] = frozenset({
    # Original schema fields relevant to drugs (Experiment A)
    "organism",
    "tissue_location",
    "cell_type",
    "cellular_compartment",
    "biological_system",
    "biological_process",
    "molecular_function",
    "pathway_category",
    "mechanism_of_action",
    "disease_association",
    "clinical_relevance",
    "chemical_classification",
    "drug_class",
    "regulatory_role",
    "interaction_type",
    "expression_context",
    # Experiment B: cell-type-specific drug context fields
    "cell_effect",
    "target_in_cell",
    "pathway_in_cell",
    "cell_vulnerability",
    "therapeutic_role_in_cell",
    # Experiment C: cell-type-characteristic fields
    "cell_function_engaged",
    "cell_metabolic_context",
    "cell_stress_response",
    "receptor_class_in_cell",
    "drug_cell_interaction_mode",
})

DRUG_CELL_TYPES: list[str] = [
    "neuron",
    "hepatocyte",
    "cardiomyocyte",
    "epithelial",
    "myocyte",
    "immune_cell",
]


# ---------------------------------------------------------------------------
# Node expansion
# ---------------------------------------------------------------------------


def expand_drug_to_cell_contexts(
    drug_node: dict,
    cell_types: list[str] | None = None,
) -> list[dict]:
    """Expand a single drug node into (drug, cell_type) composite nodes.

    Parameters
    ----------
    drug_node : dict
        A ChemicalSubstance node with at least 'id', 'name', 'label'.
    cell_types : list[str] or None
        Cell types to expand into. Defaults to DRUG_CELL_TYPES.

    Returns
    -------
    List of composite node dicts, one per cell type.
    """
    if cell_types is None:
        cell_types = DRUG_CELL_TYPES

    composites = []
    for ct in cell_types:
        composite = {
            **drug_node,
            "id": f"{drug_node['id']}::{ct}",
            "base_id": drug_node["id"],
            "cell_type_context": ct,
        }
        composites.append(composite)
    return composites


def expand_drug_nodes(
    drug_nodes: list[dict],
    cell_types: list[str] | None = None,
) -> list[dict]:
    """Expand a list of drug nodes into composite (drug, cell_type) nodes.

    Parameters
    ----------
    drug_nodes : list[dict]
        ChemicalSubstance nodes.
    cell_types : list[str] or None
        Cell types to expand into. Defaults to DRUG_CELL_TYPES.

    Returns
    -------
    Flat list of composite nodes:
        len(drug_nodes) × len(cell_types) entries.
    """
    composite_nodes = []
    for drug in drug_nodes:
        composite_nodes.extend(expand_drug_to_cell_contexts(drug, cell_types))
    return composite_nodes


# ---------------------------------------------------------------------------
# Cell-type-conditional prompts
# ---------------------------------------------------------------------------

_CELL_CONTEXT_SUMMARIZE_TEMPLATE = (
    'Provide a concise biological/chemical summary of "{entity_name}" '
    "(entity type: {entity_type}) specifically in the context of {cell_type} cells.\n\n"
    "Focus ONLY on what is known about this drug's activity in this cell type. Include:\n"
    "- Primary mechanism of action in {cell_type} cells\n"
    "- Molecular targets expressed or relevant in {cell_type} cells\n"
    "- Relevant signaling pathways active in {cell_type} cells that this drug modulates\n"
    "- Biological processes affected in {cell_type} cells\n"
    "- Tissue/organ context where {cell_type} cells are the relevant site of action\n"
    "- Clinical relevance in diseases involving {cell_type} cells\n"
    "- Known or proposed drug effects specific to {cell_type} cell biology\n\n"
    "If this drug has no known or plausible activity in {cell_type} cells, state that clearly.\n"
    "Be factual and specific to the cell type context."
)


def build_cell_context_summarize_prompt(node: dict) -> str:
    """Build a cell-type-conditional Phase 1 summarization prompt.

    Parameters
    ----------
    node : dict
        A composite node with 'name', 'label', and 'cell_type_context'.
    """
    cell_type = node.get("cell_type_context", "")
    return _CELL_CONTEXT_SUMMARIZE_TEMPLATE.format(
        entity_name=node["name"],
        entity_type=node.get("label", "ChemicalSubstance"),
        cell_type=cell_type,
    )


def build_cell_context_populate_prompt(node: dict, summary_text: str, schema: dict) -> str:
    """Build a cell-type-conditional Phase 2 population prompt.

    Parameters
    ----------
    node : dict
        A composite node with 'name', 'base_id', and 'cell_type_context'.
    summary_text : str
        The cell-type-specific summary from Phase 1.
    schema : dict
        The current schema with fields and controlled_vocabularies.
    """
    cell_type = node.get("cell_type_context", "")
    fields = schema.get("fields", [])
    vocabs = schema.get("controlled_vocabularies", {})

    field_specs = []
    for f in fields:
        if f.get("field_type") != "controlled":
            continue
        if f["name"] not in DRUG_RELEVANT_FIELDS:
            continue
        vocab_name = f.get("controlled_vocabulary", "")
        terms = vocabs.get(vocab_name, [])
        field_specs.append(
            f'- {f["name"]}: {f.get("description", "")}  '
            f"Terms: {json.dumps(terms)}"
        )
    fields_block = "\n".join(field_specs)

    base_id = node.get("base_id", node["id"])

    return (
        f"You are mapping biological context to schema fields.\n\n"
        f'Drug: "{node["name"]}" (ID: {base_id}) '
        f"specifically in the context of {cell_type} cells.\n\n"
        f"Cell-type-specific summary:\n{summary_text}\n\n"
        f"Schema fields and their allowed controlled-vocabulary terms:\n{fields_block}\n\n"
        f"Return a JSON object reflecting the drug's activity SPECIFICALLY in {cell_type} cells.\n"
        f"- Each key is a field name (use EXACTLY the field names listed above)\n"
        f"- Each value is a LIST of matching vocabulary terms\n"
        f"- Use null if the field does not apply or cannot be determined for this cell type\n"
        f'- Do NOT use placeholders like "not_applicable", "unknown", "none" — use null\n'
        f"- Only use terms from the provided vocabulary lists\n"
        f'- If a concept fits but no term matches, add a "suggested_additions" key\n\n'
        f"Return valid JSON only."
    )
