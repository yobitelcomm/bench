# Recipe: audit a published corpus

Before you base a procurement decision, a regression gate, or a published claim on someone else's envelope, audit it. `bench fetch` pulls a remote envelope into the local cache; `bench audit` confirms the content hash, the signature, and that the hardware fingerprint isn't a test placeholder.

## 1. Pull the corpus

```bash
bench fetch hf://datasets/yobitel-bench-results/llama-3.1-8b__chatbot-short__abcdef123456
bench fetch hf://datasets/yobitel-bench-results/qwen-2.5-7b__chatbot-short__deadbeef0001
```

Both envelopes land in `~/.cache/inferencebench/fetched/` by default. Use [`bench cache list`](../cli/bench-cache.md) to inspect.

## 2. Audit the cache

```bash
bench audit "$(bench cache path)"
```

Expected output on a clean corpus:

```
                                    Audit of /home/abishek/.cache/inferencebench/fetched
 status  envelope                       model                              method     content_hash  reason
   ✓     8d7ef1b17fb7.json              Qwen/Qwen2.5-7B-Instruct           dev-key    8d7ef1b17fb7
   ✓     60be8efd6d21.json              meta-llama/Llama-3.1-8B-Instruct   dev-key    60be8efd6d21
2 / 2 envelopes verified (0 failed)
```

Exit code is `0` — every envelope's content hash matches, every signature verifies against the bundled dev key (or the Sigstore root for keyless envelopes), and no hardware fingerprint is the placeholder `0000…`.

## 3. The 0/N failure case

If the publisher slipped — wrong key, tampered payload, or a test fixture posing as a real run — `bench audit` flags it:

```
                                    Audit of /home/abishek/.cache/inferencebench/fetched
 status  envelope                       model                              method     content_hash  reason
   ✗     8d7ef1b17fb7.json              Qwen/Qwen2.5-7B-Instruct           dev-key    8d7ef1b17fb7  signature does not match content_hash (tampered or wrong key)
   ✗     60be8efd6d21.json              meta-llama/Llama-3.1-8B-Instruct   dev-key    60be8efd6d21  placeholder hardware_fingerprint
0 / 2 envelopes verified (2 failed)
```

`bench audit` exits `1` in strict mode (the default). Stop the downstream pipeline here; don't publish a leaderboard or wire the envelope into CI until the publisher re-issues a clean one. Send them the audit report — the table identifies the exact failure reason per envelope without you having to dig.

## 4. Wire it into a publishing pipeline

```bash
bench audit ./drop-zone --report json | jq '.n_ok == .n_total'
```

`bench audit --report json` emits an `inferencebench.audit.v1` payload with per-envelope rows plus aggregate counts, suitable for gating an upload script or a GitHub Actions step.

## Where to go next

- [bench audit reference](../cli/bench-audit.md) — full flag table and exit codes
- [bench verify](../cli/bench-verify.md) — single-envelope variant
- [The signed envelope](../concepts/envelope.md)
