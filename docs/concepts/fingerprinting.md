# Hardware fingerprinting

The hardware fingerprint identifies the machine that produced a benchmark result. It is composed of stable identifiers from the BIOS, the GPUs, the CPU, the memory, the BIOS settings, the driver, CUDA, and NCCL. The composition is SHA-256'd into a single 64-character hex digest for fast equality checks.

## Why this matters

Two H100 nodes can produce very different numbers depending on PCIe topology, NUMA layout, BIOS settings, and driver version. Without a fingerprint, "ran on H100" tells you almost nothing. With a fingerprint, you can prove two results came from the same configuration, or you can show that a difference is explained by a configuration change.

## Composition

The fingerprint body is a canonical JSON object with the following fields:

```python
fingerprint_data = {
    "dmi_uuid": dmi_uuid,
    "gpus": sorted([{
        "model": gpu.model,
        "pci_id": gpu.pci_id,
        "serial": gpu.serial,
        "vbios": gpu.vbios,
    } for gpu in gpus], key=lambda g: g["pci_id"]),
    "cpu": {"model": cpu.model, "microcode": cpu.microcode},
    "memory": {"channels": memory.channels, "speed_mts": memory.speed_mts, "ecc": memory.ecc},
    "bios": {"version": bios.version, "resizable_bar": bios.resizable_bar, "above_4g": bios.above_4g},
    "numa": numa_topology_canonical(),
    "driver": gpu_driver_version,
    "cuda": cuda_version,
    "nccl": nccl_version,
}
canonical = json.dumps(fingerprint_data, sort_keys=True, separators=(",", ":"))
fingerprint_sha256 = hashlib.sha256(canonical.encode()).hexdigest()
```

## Fields

| Field | Source | Why it is in the fingerprint |
|---|---|---|
| `dmi_uuid` | `/sys/class/dmi/id/product_uuid` | Stable per-chassis identifier. |
| `gpus[].model` | NVML `nvmlDeviceGetName` | Distinguishes H100 SXM5 80GB from H100 PCIe 80GB. |
| `gpus[].pci_id` | `lspci` / NVML | Topology matters for NVLink/PCIe paths. |
| `gpus[].serial` | NVML `nvmlDeviceGetSerial` | Detects board swaps. |
| `gpus[].vbios` | NVML `nvmlDeviceGetVbiosVersion` | VBIOS revisions change clocks. |
| `cpu.model`, `cpu.microcode` | `/proc/cpuinfo`, `microcode_ctl` | Microcode patches change cache/timing behavior. |
| `memory.channels`, `speed_mts`, `ecc` | DMI / `dmidecode` | Bandwidth-bound workloads care. |
| `bios.version` | DMI | Vendor BIOS updates retune memory and PCIe. |
| `bios.resizable_bar`, `bios.above_4g` | DMI | Both affect host-to-device transfer. |
| `numa` | `/sys/devices/system/node/` | NUMA layout changes effective bandwidth. |
| `driver` | NVML | The NVIDIA kernel driver. |
| `cuda` | `nvcc --version` / runtime | The CUDA toolkit runtime. |
| `nccl` | NCCL library | Collective communication library. |

## Worked example

```bash
bench doctor --strict
```

The diagnostic prints the same fields the harness will hash. After a successful `bench run`, you can find the fingerprint inside the envelope:

```bash
jq .hardware_fingerprint ~/.cache/inferencebench/runs/latest/envelope.json
```

Expected output (truncated):

```json
{
  "fingerprint_sha256": "8b1a9c2f...",
  "dmi_uuid": "d3b07384-d9a8-4e0d-bf63-1c2e3f4a5b6c",
  "gpus": [
    {
      "model": "H100-SXM5-80GB",
      "pci_id": "0000:1b:00.0",
      "serial": "1234567890",
      "vbios": "96.00.5E.00.01"
    }
  ],
  "cpu": {"model": "AMD EPYC 9354", "microcode": "0xa101144"},
  "memory": {"channels": 12, "speed_mts": 4800, "ecc": true},
  "bios": {"version": "1.3.0", "resizable_bar": true, "above_4g": true},
  "driver": "560.35.03",
  "cuda": "12.6",
  "nccl": "2.22.3"
}
```

## When fingerprints differ

Two envelopes with different `fingerprint_sha256` values are not directly comparable. `bench compare` will surface the diff and refuse to draw a Pareto frontier across mismatched fingerprints unless you pass `--allow-fingerprint-mismatch` (Phase 2).

## See also

- [Reproducibility](reproducibility.md)
- [The signed envelope](envelope.md)
- [bench doctor](../cli/bench-doctor.md)
