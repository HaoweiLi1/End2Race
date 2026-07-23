# Validation report notes

## Required structure map

Title; technical summary; key findings with chart/table evidence; scope and definitions;
methodology; limitations and robustness; recommendation; further questions.

## Chart map

| Section | Question | Family | Fields | Supported claim |
|---|---|---|---|---|
| Target-KL mechanism | How irregular is the optimization budget? | Grouped bar | update, steps, target | Both targets create state-dependent truncation |
| Target-KL eval | Where and how large is the safety collapse? | Bar | update, collision count | 0.04 peaks at 70 and only partly recovers |

## Omitted visuals

- G3/G5 uses tables because there are only two warm-up arms and five tensor checkpoints.
- Paired scenario comparisons use a table because shared/resolved/created and unadjusted p-values require exact lookup.
