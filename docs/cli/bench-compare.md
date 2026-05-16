# bench compare

Compare two or more benchmark runs. Default output is a Pareto frontier across throughput, latency, cost, and energy.

```bash
bench compare <run-id-or-envelope-path>... [--report pareto|table|json]
```

## Example

```bash
bench compare \
  ~/.cache/inferencebench/runs/01J7Q5C6.../envelope.json \
  ~/.cache/inferencebench/runs/01J7Q6XY.../envelope.json \
  --report pareto
```

Expected output:

```
Pareto frontier: 2 runs, 8 points
  *  Llama-4-Maverick fp8     ttft_p50=142.0ms  tput=1842 tok/s  $0.18/Mtok
     Llama-4-Maverick fp16    ttft_p50=168.4ms  tput=1340 tok/s  $0.31/Mtok
Pareto-dominant (1):
  - fp8 dominates fp16 on (latency, throughput, cost)
```

## Arguments

| Argument | Required | Description |
|---|---|---|
| `run-ids` | yes (one or more) | Run IDs, envelope paths, or `hf://datasets/...` URIs. |

## Options

| Option | Default | Description |
|---|---|---|
| `--report` | `pareto` | Report format: `pareto`, `table`, `json`. |

## Report formats

| Format | What you get |
|---|---|
| `pareto` | A frontier across throughput, latency, cost, and energy. Marks dominated runs. |
| `table` | A side-by-side metrics table. |
| `json` | The canonical JSON, suitable for piping into another tool. |

## Phase 1 status

`bench compare` is a stub in v0.0.0. The full Pareto renderer wires in during the v0.1 release.

## See also

- [Pareto frontiers](../concepts/pareto.md)
- [bench run](bench-run.md)
