# bench doctor

Diagnose hardware health before benchmarking. Refuses to continue if the system is in a state that would produce unreliable numbers.

```bash
bench doctor [--strict]
```

## Example

```bash
bench doctor
```

Expected output (healthy node):

```
Hardware diagnostic
Check                Status   Detail
NVML available       PASS     12 GPUs visible
Driver version       PASS     560.35.03
ECC enabled          PASS     enabled on all GPUs
Persistence mode     PASS     enabled
Thermal headroom     PASS     all GPUs < 75 degC
Clock state          PASS     no throttling flags
OK — all checks passed.
```

Expected output (refusal):

```
Hardware diagnostic
Check                Status   Detail
NVML available       PASS     12 GPUs visible
Thermal headroom     FAIL     GPU 4 at 87 degC, throttling
Clock state          FAIL     SW_THERMAL_SLOWDOWN active on GPU 4
REFUSED — 2 FAIL. Resolve hardware issues before benchmarking.
```

Exit code is `0` if all checks pass, `1` otherwise.

## Options

| Option | Default | Description |
|---|---|---|
| `--strict` | off | Treat `WARN` as failure. By default, `bench doctor` only fails on `FAIL`. |

## Refusal modes

`bench doctor` refuses (exit 1) when any of the following are true:

- Thermal throttling flags are set
- ECC errors are present (correctable or uncorrectable)
- Driver version is below the minimum supported
- Persistence mode is disabled (clock ramp-up adds noise)
- GPU clock state is not at expected base/boost levels

With `--strict`, the diagnostic also refuses on warnings such as elevated idle temperatures or a non-stable BIOS configuration.

## What the diagnostic checks

| Check | What it verifies |
|---|---|
| NVML available | The NVIDIA Management Library is loadable and at least one GPU is visible. |
| Driver version | Driver is at or above the supported floor. |
| ECC enabled | ECC is on for every GPU we will measure. |
| Persistence mode | `nvidia-persistenced` is running. |
| Thermal headroom | All GPUs are below the throttle temperature. |
| Clock state | No throttling flags (`SW_THERMAL_SLOWDOWN`, `HW_SLOWDOWN`, etc.). |

## When the diagnostic skips checks

On a CPU-only node, `bench doctor` exits 0 without running any check and prints `No checks ran (no NVIDIA GPUs detected).` Plugins that require a GPU will themselves refuse to run.

## See also

- [Reproducibility](../concepts/reproducibility.md)
- [Hardware fingerprinting](../concepts/fingerprinting.md)
