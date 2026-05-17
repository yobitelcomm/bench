# bench schema

Emit JSON Schema for the canonical envelope, the LLM plugin's benchmark spec, or the local mirror index produced by `bench publish --to local`. Useful when consuming envelopes from a non-Python service (Go, Rust, TypeScript verifier) that needs a stable contract.

## Synopsis

```bash
bench schema [--target envelope|benchmark-spec|mirror-index] [--out PATH] [--version]
```

## Example: dump the envelope schema

```bash
bench schema --target envelope --out envelope.schema.json
```

Then verify the version the schema corresponds to:

```bash
bench schema --version
```

Expected output:

```
v1
```

A truncated peek at the envelope schema:

```json
{
  "$defs": { "Dataset": { ... }, "Engine": { ... } },
  "additionalProperties": false,
  "properties": {
    "schema_version": { "const": "v1", "type": "string" },
    "suite_id": { "type": "string" },
    "model": { "$ref": "#/$defs/Model" },
    "engine": { "$ref": "#/$defs/Engine" },
    "metrics": { "type": "object" },
    "signature": { "anyOf": [ { "$ref": "#/$defs/Signature" }, { "type": "null" } ] }
  },
  "required": ["schema_version", "suite_id", "model", "engine", "metrics"],
  "type": "object"
}
```

## Targets

| `--target` | Source | Notes |
|---|---|---|
| `envelope` (default) | `Envelope.model_json_schema()` | Always available. |
| `benchmark-spec` | `inferencebench_llm.schemas.BenchmarkSpec` | Requires the `inferencebench-llm` plugin to be installed; exits `2` with an install hint otherwise. |
| `mirror-index` | hand-written for `inferencebench.mirror.v1` | Matches the `index.json` payload written by `bench publish --to local`. |

## Flags

| Flag | Default | Description |
|---|---|---|
| `--target` | `envelope` | Which schema to emit. |
| `--out` | stdout | Write JSON to a path instead of stdout (parent dirs are created). |
| `--version` | off | Print the envelope schema version (e.g. `v1`) and exit. Ignores `--target`. |

When `--out` is unset the JSON goes to stdout via `print()` (no Rich highlighting), so it's safe to pipe into `jq` or a schema validator.

## See also

- [Envelope schema reference](../reference/envelope-schema.md)
- [The signed envelope](../concepts/envelope.md)
