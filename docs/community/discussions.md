# GitHub Discussions

[GitHub Discussions](https://github.com/yobitelcomm/bench/discussions) is the home for open-ended conversation — questions, ideas, show-and-tell, results from running `bench` against your own hardware. Issues are reserved for actionable bug reports, feature requests, and benchmark proposals.

GitHub does not let us check the Discussions configuration into the repo, so the categories below are a recommendation for whoever administers the repo to create.

## Suggested categories

| Category | Format | Purpose |
|---|---|---|
| **Announcements** | Announcement | Release notes and roadmap updates from maintainers. Read-only for everyone else. |
| **Q & A** | Q & A | "How do I…" questions about `bench` usage. Mark the helpful answer once resolved. |
| **Ideas** | Open-ended | Half-baked thoughts before they harden into a feature request. Easier to triage than a low-effort feature issue. |
| **Show and tell** | Open-ended | Share your envelopes, your leaderboards, your hardware comparisons. Encouraged: links to a published Hugging Face dataset repo of envelopes. |
| **Methodology** | Open-ended | Discuss whether a benchmark is biased, contaminated, or game-able. Promote to a `[benchmark]` issue if a concrete change is needed. |
| **Hardware** | Open-ended | Hardware-specific gotchas, NVML quirks, MI300X / Blackwell / M-series notes. |

## When to open an issue instead

Open an issue (not a discussion) when:

- You have a reproduction of a bug — use [bug.yml](https://github.com/yobitelcomm/bench/blob/main/.github/ISSUE_TEMPLATE/bug.yml).
- You have a concrete feature proposal with a use case — use [feature.yml](https://github.com/yobitelcomm/bench/blob/main/.github/ISSUE_TEMPLATE/feature.yml).
- You want a new benchmark added — use [benchmark.yml](https://github.com/yobitelcomm/bench/blob/main/.github/ISSUE_TEMPLATE/benchmark.yml).
- You have a security vulnerability — follow [SECURITY.md](security.md). Do not file publicly.

## Etiquette

- Search first. Discussions get long; the same question gets asked twice.
- Title posts like a search query, not a greeting.
- For numerical claims, attach the signed envelope (or a Hugging Face Hub link). "My run was faster" is not actionable; an envelope is.
- For methodology criticism, propose a remediation, not just a complaint.
