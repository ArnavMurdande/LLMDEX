# Identity Matching

`pipeline/identity.py` creates:

- `family_id`
- `variant_id`
- `deployment_profile_id`
- source model ID/URL
- provider and base model name
- version
- reasoning effort
- deployment/fallback metadata
- parameter count
- aliases

Approved aliases and representative overrides live in
`data/identity/manual_overrides.json`. Generated registry and review outputs:

- `model_registry.json`
- `match_audit.json`
- `match_audit.csv`
- `unresolved_matches.json`

Confidence:

- 1.00: exact source ID/URL or approved alias.
- 0.95: exact provider + family + version.
- 0.80 maximum: fuzzy review candidate.
- below 0.80: unresolved/source-missing.

Fuzzy candidates never enter consensus until a manual override is reviewed and
versioned.
