# inferencebench-code

Code-generation plugin for the InferenceBench Suite.

HumanEval-style execution-based benchmarks: the plugin sends a function-signature
prompt to the model, extracts the Python code from its response, executes it
against bundled unit tests in a subprocess, and reports `pass_at_1`.

Suite ID: `code.generation`

Bundled benchmarks:

- `code.generation.humaneval-mini` — 5 stdlib-only Python tasks, `pass_at_1`
  scoring with a 5-second per-task wall-clock timeout.

## SAFETY WARNING — read before running

**This plugin executes model-generated code.** Every run prints a yellow banner
reminding you of that. The execution layer is *best-effort* defence-in-depth,
not a real sandbox:

- Each task's solution + tests are written to a temp file and invoked with
  `python -I` (isolated mode) under a `subprocess.run(timeout=...)` wall clock.
- A cheap substring pre-scan refuses any solution that imports `subprocess`,
  `os.system`, `socket`, `urllib`, `multiprocessing`, or `ctypes`.
- The bundled fixtures are stdlib-only, no I/O, no network.

This is **deliberately not airtight**. Phase 2 adds real isolation (firejail /
nsjail / container-per-task). Until then: only run code-generation benchmarks
against models you trust, on machines you can afford to throw away, and never
against the bundled fixtures replaced with untrusted input.

## Metrics

The envelope's `metrics` block includes:

| Metric             | Direction       | Meaning                                   |
| ------------------ | --------------- | ----------------------------------------- |
| `pass_at_1`        | higher is better | mean of per-task passed booleans          |
| `pass_at_1_p05/50/95` | higher is better | bootstrap quantiles of per-sample scores |
| `timeout_rate`     | lower is better  | fraction of tasks that hit the wall clock |
| `ttft_p50_ms`      | -               | model time-to-first-token, median         |
| `total_p50_ms`     | -               | model total request time, median          |
| `tokens_out_total` | -               | total generated tokens across the run     |
| `ok_rate`          | -               | fraction of model calls that succeeded    |
| `n_samples`        | -               | fixture row count                         |
