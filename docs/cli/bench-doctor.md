# bench doctor

Diagnose hardware health before benchmarking. Refuses (exit `1`) when the system is in a state that would produce unreliable numbers — thermal throttling, ECC errors, driver drift, persistence disabled, etc.

## Synopsis

```bash
bench doctor [--strict]
```

## Example: healthy 8×H100 node

```bash
bench doctor
```

Expected output:

```
                    Hardware diagnostic
 Check                Status  Detail
 NVML available       PASS    8 GPUs visible
 Driver version       PASS    560.35.03
 ECC enabled          PASS    enabled on all GPUs
 Persistence mode     PASS    enabled
 Thermal headroom     PASS    all GPUs < 75 degC
 Clock state          PASS    no throttling flags
OK — all checks passed.
```

Refusal example:

```
 Thermal headroom     FAIL    GPU 4 at 87 degC, throttling
 Clock state          FAIL    SW_THERMAL_SLOWDOWN active on GPU 4
REFUSED — 2 FAIL. Resolve hardware issues before benchmarking.
```

Exit code is `0` if all checks pass, `1` otherwise. On a CPU-only host, all checks are skipped and the command exits `0` with `No checks ran (no NVIDIA GPUs detected).`

## Flags

| Flag | Default | Description |
|---|---|---|
| `--strict` | off | Treat `WARN` as failure. Default only fails on `FAIL`. |
| `--show-slo` | off | Append a table showing the detected hardware class + resolved `llm.standard` SLO thresholds for this host. |

## Inspecting hardware-aware SLO thresholds

`bench` ships hardware-aware SLO templates: the base numbers in `llm.standard` are anchored to an H100 (1.0x multiplier), and every other class scales them up or down. Use `--show-slo` to see what the active host's resolved thresholds are — useful when diagnosing "why did my benchmark fail its SLO":

```bash
bench doctor --show-slo
```

Example output on an RTX 4090 host (1.8x multiplier):

```
                SLO template (llm.standard)
 Field                Value
 Hardware class       rtx-4090
 Description          NVIDIA RTX 4090 (consumer)
 ttft multiplier      1.8x
 tpot multiplier      1.8x
 total multiplier     1.8x
 Resolved thresholds  ttft<360ms, tpot<90ms, total<5400ms
```

## Checks

| Check | Verifies |
|---|---|
| NVML available | NVML is loadable and at least one GPU is visible. |
| Driver version | Driver is at or above the supported floor. |
| ECC enabled | ECC is on for every GPU we will measure. |
| Persistence mode | `nvidia-persistenced` is running (clock-ramp noise off). |
| Thermal headroom | All GPUs are below the throttle temperature. |
| Clock state | No throttling flags (`SW_THERMAL_SLOWDOWN`, `HW_SLOWDOWN`, etc.). |

## See also

- [Reproducibility](../concepts/reproducibility.md)
- [Hardware fingerprinting](../concepts/fingerprinting.md)
