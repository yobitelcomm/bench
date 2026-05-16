# Methodology

A benchmark is its methodology. The number on the leaderboard is a function of the dataset, the driver, the warm-up, the convergence gate, the metrics, and the judge. Change any one of those and the numbers move.

Every InferenceBench plugin ships a public methodology page that explains exactly how its results are produced. External researchers should be able to cite the page and reproduce a run from it alone.

## What a methodology page must answer

1. What does this benchmark measure?
2. What dataset is used, and how is it sampled?
3. How is the system warmed up before measurement?
4. What driver produces traffic? Open-loop? Closed-loop? At what arrival rate?
5. What metrics are computed, with what formulas?
6. If a judge model is used, which one, and how is it calibrated?
7. What known limitations exist?
8. How can a reader reproduce the run?

## Page template

The `docs-writer` agent ships a template. Every plugin must have a `methodology.md` that fills it in.

```markdown
# Methodology: <suite-id>

## What this benchmark measures
<one paragraph>

## Dataset
- Source: <url + citation>
- Size: <n examples>
- Hash: <sha256>
- Sampling: <strategy>
- Rotation: <how often the private held-out set rotates>

## Driver
<open-loop Poisson? closed-loop? batch? exact parameters>

## Warm-up
<three discarded runs; convergence gate at CoV < 5% over last 30 requests>

## Metrics
<formulas + motivation>

## Judge model (if applicable)
<which judge? calibration data? inter-judge agreement?>

## Known limitations
<honest list>

## How to reproduce
bench run <suite-id> --model ... --hardware ... --seed 42

## Citation
<bibtex>

## Methodology version history
| Version | Date | Change |
```

## Versioning

Methodology pages are versioned alongside the plugin. A breaking methodology change bumps the major version (e.g. `llm.inference` 1.x → 2.x); a methodology tweak bumps the minor version. The version is embedded in the envelope as `suite_version`.

## Methodology review

New benchmarks and methodology changes go through a `benchmark-validator` review. The reviewer checks for:

- A clear, plain-English description of what is measured
- A dataset hash that can be re-derived from the dataset id
- A driver specification precise enough to reproduce
- Metric formulas that match the code
- A calibrated judge (where applicable), with inter-judge agreement reported
- Honest known limitations

## Disputes

If you believe a published result is wrong, file an issue with the envelope URL and the specific claim you dispute. The reproducibility path is: re-run with the same envelope inputs, compare. If the dispute is methodological, propose a methodology change as a versioned bump.

## See also

- [Reproducibility](reproducibility.md)
- [Vendor neutrality](vendor-neutrality.md)
