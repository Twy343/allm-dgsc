# Datasets

Status: empty placeholders. No benchmark has been downloaded and no cases have
been generated. The proposal begins with small controlled case studies.

## Planned Contents

- `semantic_intervention/`: prompts with an identifiable intermediate value,
  counterfactual alternatives, and expected downstream consequences.
- `positional_intervention/`: prompts with identifiable token anchors,
  neighborhood relationships, and position-sensitive constraints.
- `dynamic_revision/`: paired initial/updated contexts, required reasoning edits,
  unaffected content, and expected conclusions.

Use a structured format such as JSONL when cases are implemented. Record stable
case IDs, prompts, expected behavior, variant-generation seeds, and provenance.
Token IDs and intervention positions depend on the chosen tokenizer and sampled
trajectory; record them with run metadata after verifying alignment.

Keep expected behavior separate from observed outputs. Runtime trajectories,
logits, and metrics belong in ignored `outputs/`, not in these source datasets.
Track small reproducible cases here; decide storage separately before adding
large external data.
