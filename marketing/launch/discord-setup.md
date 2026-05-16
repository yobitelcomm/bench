# Discord setup notes — InferenceBench

Runbook for standing up the InferenceBench Discord. Not a published page.

Server name: `InferenceBench`; vanity URL `inferencebench` if available.

## Channels

Under a single category `InferenceBench`:

- `#welcome` — read-only, rules + invite
- `#general` — general chat
- `#cli` — `bench` CLI, install, plugin discovery
- `#studio` — Phase 2 placeholder (locked)
- `#plugin-development` — plugin authors, drivers, harness internals
- `#methodology-disputes` — public methodology arguments
- `#voice-and-multimodal` — Phase 2 placeholder (locked)
- `#robotics-and-chip` — Phase 3+ placeholder (locked)
- `#show-and-tell` — users post their signed envelopes
- `#show-and-tell-bots` — HF publish webhook, read-only
- `#ci` — GitHub Actions webhook, read-only

## Roles

- `@core` — maintainers with repo merge rights
- `@plugin-author` — shipped a plugin under `inferencebench.plugins`
- `@verified-vendor` — vendors posting under their real org, no astroturf
- `@community` — default on join

Role colors muted. `@verified-vendor` is a label, not a sales channel.

## Rules to pin in `#welcome`

1. No vendor astroturfing. Vendors get `@verified-vendor` first; no employer numbers under a personal handle.
2. Methodology disputes go through the `methodology-issue.md` GitHub template; the issue is the binding artifact.
3. Benchmark numbers must include the `bench verify` URL or envelope link.
4. Repo CoC applies. Reports: `conduct@yobitel.com`.

## Why `#methodology-disputes` exists

We expect Berkeley-RDI-style reward-hacking critiques of any benchmark this project ships. The channel airs them in public, linked to the GitHub issue — that becomes the audit trail.

## Webhooks

- **GitHub** → `#ci`: repo `Settings → Integrations → Webhooks`. Subscribe to push, pull_request, workflow_run, release.
- **HF publish** → `#show-and-tell-bots`: `integrations/hf-publisher` POSTs on successful `bench publish --to hf`. URL in `~/.config/inferencebench/integrations.toml` or a GH Actions secret.

## Invite link

Use Discord's vanity URL if eligible, otherwise a permanent invite (`Never expire` / `No max uses`). Rotate on leak.

TODO: human to fill in — paste the invite URL into the repo README and `docs/index.md` after provisioning.
