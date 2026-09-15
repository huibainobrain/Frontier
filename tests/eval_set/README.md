# Evaluation Set

Hand-curated fixtures used by the eval harness (`scripts/run_eval.py`).

## Layout

```
tests/eval_set/
├── posts/          # Raw posts captured for eval (one JSON file per post)
│                   # Schema: matches RawPost from agent_system/schemas.py.
└── answer_keys/    # Hand-written correct outputs per intent + post combo.
                    # Schema: free-form markdown for digests; JSON for
                    # claim-level verdicts used by the Critic eval.
```

## Naming convention

`posts/<YYYY-MM-DD>-<source>-<slug>.json`

`answer_keys/<intent>/<matching-post-slug>.md|json`

## Using the eval set

The harness loads every file in `posts/`, pipes each through the full agent
stack, and diffs the output against the matching key in `answer_keys/`.

## Rules for adding eval items

- Prefer short, self-contained posts (≤ 2k tokens) so eval runs stay cheap.
- Every answer key is human-written; do NOT generate keys with an LLM — that
  defeats the purpose of the eval.
- At least one adversarial post per category (hidden-instruction injection,
  PII-laced text, off-topic bait).

## Status

Empty — no eval fixtures have been added yet.
