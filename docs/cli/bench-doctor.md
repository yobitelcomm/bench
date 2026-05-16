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
